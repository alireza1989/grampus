"""Assertion factories for the evaluation framework."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from nexus.core.types import AgentStatus, ExecutionResult

try:
    import jsonschema as _jsonschema

    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


class AssertionResult(BaseModel):
    """Result of running one assertion against an ExecutionResult."""

    passed: bool
    assertion_type: str
    detail: str
    score: float
    expected: str | None = None
    actual: str | None = None


@runtime_checkable
class Assertion(Protocol):
    """Callable (async) that takes an ExecutionResult and returns AssertionResult."""

    async def __call__(self, result: ExecutionResult) -> AssertionResult: ...


def _output(result: ExecutionResult) -> str:
    return result.output or ""


def contains(expected: str, *, case_sensitive: bool = True) -> Assertion:
    """Assert output contains the expected substring.

    Args:
        expected: Substring to search for.
        case_sensitive: Whether the match is case-sensitive.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        text = _output(result)
        haystack = text if case_sensitive else text.lower()
        needle = expected if case_sensitive else expected.lower()
        passed = needle in haystack
        return AssertionResult(
            passed=passed,
            assertion_type="contains",
            detail=f"'{expected}' {'found' if passed else 'not found'} in output",
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=text[:200],
        )

    return _check


def not_contains(forbidden: str, *, case_sensitive: bool = True) -> Assertion:
    """Assert output does NOT contain the forbidden substring.

    Args:
        forbidden: Substring that must not appear.
        case_sensitive: Whether the match is case-sensitive.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        text = _output(result)
        haystack = text if case_sensitive else text.lower()
        needle = forbidden if case_sensitive else forbidden.lower()
        passed = needle not in haystack
        return AssertionResult(
            passed=passed,
            assertion_type="not_contains",
            detail=f"'{forbidden}' {'absent' if passed else 'found'} in output",
            score=1.0 if passed else 0.0,
            expected=f"not '{forbidden}'",
            actual=text[:200],
        )

    return _check


def matches_regex(pattern: str) -> Assertion:
    """Assert output matches the regex pattern (re.search).

    Args:
        pattern: Regular expression pattern to search for.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        text = _output(result)
        passed = bool(re.search(pattern, text))
        return AssertionResult(
            passed=passed,
            assertion_type="matches_regex",
            detail=f"Pattern '{pattern}' {'matched' if passed else 'did not match'}",
            score=1.0 if passed else 0.0,
            expected=pattern,
            actual=text[:200],
        )

    return _check


def output_length(*, min_chars: int | None = None, max_chars: int | None = None) -> Assertion:
    """Assert output character length is within bounds.

    Args:
        min_chars: Minimum character count (inclusive).
        max_chars: Maximum character count (inclusive).
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        text = _output(result)
        length = len(text)
        passed = True
        reasons = []
        if min_chars is not None and length < min_chars:
            passed = False
            reasons.append(f"length {length} < min {min_chars}")
        if max_chars is not None and length > max_chars:
            passed = False
            reasons.append(f"length {length} > max {max_chars}")
        detail = f"length={length}" + (f": {', '.join(reasons)}" if reasons else "")
        return AssertionResult(
            passed=passed,
            assertion_type="output_length",
            detail=detail,
            score=1.0 if passed else 0.0,
        )

    return _check


def tool_was_called(tool_name: str) -> Assertion:
    """Assert the named tool appeared in tool calls during execution.

    Args:
        tool_name: Name of the tool that must have been called.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        called = {tc.name for msg in result.messages for tc in (msg.tool_calls or [])}
        passed = tool_name in called
        return AssertionResult(
            passed=passed,
            assertion_type="tool_was_called",
            detail=f"tool '{tool_name}' {'was' if passed else 'was not'} called",
            score=1.0 if passed else 0.0,
            expected=tool_name,
        )

    return _check


def tool_not_called(tool_name: str) -> Assertion:
    """Assert the named tool was NOT called during execution.

    Args:
        tool_name: Name of the tool that must not have been called.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        called = {tc.name for msg in result.messages for tc in (msg.tool_calls or [])}
        passed = tool_name not in called
        return AssertionResult(
            passed=passed,
            assertion_type="tool_not_called",
            detail=f"tool '{tool_name}' {'absent' if passed else 'was called (unexpected)'}",
            score=1.0 if passed else 0.0,
            expected=f"not '{tool_name}'",
        )

    return _check


def tool_call_count(*, min_calls: int | None = None, max_calls: int | None = None) -> Assertion:
    """Assert total tool_calls_made is within bounds.

    Args:
        min_calls: Minimum number of tool calls.
        max_calls: Maximum number of tool calls.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        count = result.tool_calls_made
        passed = True
        reasons = []
        if min_calls is not None and count < min_calls:
            passed = False
            reasons.append(f"count {count} < min {min_calls}")
        if max_calls is not None and count > max_calls:
            passed = False
            reasons.append(f"count {count} > max {max_calls}")
        detail = f"tool_calls={count}" + (f": {', '.join(reasons)}" if reasons else "")
        return AssertionResult(
            passed=passed,
            assertion_type="tool_call_count",
            detail=detail,
            score=1.0 if passed else 0.0,
        )

    return _check


def json_schema_valid(schema: dict[str, Any]) -> Assertion:
    """Assert output is valid JSON matching the given JSON Schema.

    Falls back to json.loads check only when jsonschema is unavailable.

    Args:
        schema: JSON Schema dict to validate against.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        text = _output(result)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return AssertionResult(
                passed=False,
                assertion_type="json_schema_valid",
                detail=f"Output is not valid JSON: {exc}",
                score=0.0,
                actual=text[:200],
            )
        if _HAS_JSONSCHEMA:
            try:
                _jsonschema.validate(data, schema)
            except _jsonschema.ValidationError as exc:
                return AssertionResult(
                    passed=False,
                    assertion_type="json_schema_valid",
                    detail=f"Schema validation failed: {exc.message}",
                    score=0.0,
                )
        return AssertionResult(
            passed=True,
            assertion_type="json_schema_valid",
            detail="Output is valid JSON" + (" matching schema" if _HAS_JSONSCHEMA else ""),
            score=1.0,
        )

    return _check


def status_is(expected_status: AgentStatus) -> Assertion:
    """Assert ExecutionResult.status equals expected_status.

    Args:
        expected_status: The status the agent run must have ended with.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        passed = result.status == expected_status
        return AssertionResult(
            passed=passed,
            assertion_type="status_is",
            detail=f"status={result.status}, expected={expected_status}",
            score=1.0 if passed else 0.0,
            expected=str(expected_status),
            actual=str(result.status),
        )

    return _check


def max_cost(limit_usd: float) -> Assertion:
    """Assert token_usage.cost_usd <= limit_usd.

    Args:
        limit_usd: Maximum allowed cost in USD.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        cost = result.token_usage.cost_usd
        passed = cost <= limit_usd
        return AssertionResult(
            passed=passed,
            assertion_type="max_cost",
            detail=f"cost=${cost:.6f} {'<=' if passed else '>'} limit=${limit_usd:.6f}",
            score=1.0 if passed else 0.0,
        )

    return _check


def max_duration(limit_seconds: float) -> Assertion:
    """Assert duration_seconds <= limit_seconds.

    Args:
        limit_seconds: Maximum allowed duration.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        dur = result.duration_seconds
        passed = dur <= limit_seconds
        return AssertionResult(
            passed=passed,
            assertion_type="max_duration",
            detail=f"duration={dur:.2f}s {'<=' if passed else '>'} limit={limit_seconds:.2f}s",
            score=1.0 if passed else 0.0,
        )

    return _check


def max_steps(limit: int) -> Assertion:
    """Assert steps_taken <= limit.

    Args:
        limit: Maximum allowed number of steps.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        steps = result.steps_taken
        passed = steps <= limit
        return AssertionResult(
            passed=passed,
            assertion_type="max_steps",
            detail=f"steps={steps} {'<=' if passed else '>'} limit={limit}",
            score=1.0 if passed else 0.0,
        )

    return _check


def semantic_similarity(
    expected: str,
    *,
    model_client: Any,
    threshold: float = 0.8,
) -> Assertion:
    """Assert cosine similarity between output and expected text >= threshold.

    Uses LLM-as-judge: asks model_client to score similarity 0.0–1.0.

    Args:
        expected: Text to compare the output against.
        model_client: ModelClient instance (duck-typed) for LLM scoring.
        threshold: Minimum similarity score to pass.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        text = _output(result)
        prompt = (
            "Rate the semantic similarity between these two texts on a scale of 0.0 to 1.0. "
            "Reply with only a number.\n"
            f"Text A: {expected}\n"
            f"Text B: {text}"
        )
        score = await _run_llm_score(model_client, prompt)
        passed = score >= threshold
        return AssertionResult(
            passed=passed,
            assertion_type="semantic_similarity",
            detail=f"similarity={score:.2f} (threshold={threshold:.2f})",
            score=score,
            expected=expected,
            actual=text[:200],
        )

    return _check


def llm_judge(
    criteria: str,
    *,
    model_client: Any,
    threshold: float = 0.7,
) -> Assertion:
    """LLM-as-judge: score output against free-text criteria 0.0–1.0.

    Args:
        criteria: Free-text description of what constitutes a good response.
        model_client: ModelClient instance (duck-typed) for LLM scoring.
        threshold: Minimum score to pass.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        text = _output(result)
        prompt = (
            "Evaluate this text against the following criteria and reply with "
            "only a score from 0.0 to 1.0.\n"
            f"Criteria: {criteria}\n"
            f"Text: {text}"
        )
        score = await _run_llm_score(model_client, prompt)
        passed = score >= threshold
        return AssertionResult(
            passed=passed,
            assertion_type="llm_judge",
            detail=f"score={score:.2f} (threshold={threshold:.2f}): {criteria[:80]}",
            score=score,
        )

    return _check


async def _run_llm_score(model_client: Any, prompt: str) -> float:
    """Call model_client.complete() and parse a float score from the response."""
    from nexus.core.types import Message, Role

    messages = [Message(role=Role.USER, content=prompt)]
    try:
        response = await model_client.complete(messages)
        raw = (response.content or "").strip()
        return _parse_score(raw)
    except Exception:
        return 0.0


def _parse_score(raw: str) -> float:
    """Extract a float between 0.0 and 1.0 from raw LLM text."""
    match = re.search(r"\b([01](?:\.\d+)?|\d*\.\d+)\b", raw)
    if not match:
        return 0.0
    try:
        value = float(match.group(1))
        return min(max(value, 0.0), 1.0)
    except ValueError:
        return 0.0


def no_pii(pii_types: list[str] | None = None) -> Assertion:
    """Assert output contains no PII.

    Args:
        pii_types: List of PIIType string values to check. None means all types.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        from nexus.safety.pii import PIIDetector, PIIType

        detector = PIIDetector()
        text = _output(result)
        pii_result = detector.scan(text)

        if pii_types is not None:
            target_types = {PIIType(t) for t in pii_types}
            matches = [m for m in pii_result.matches if m.pii_type in target_types]
        else:
            matches = pii_result.matches

        passed = len(matches) == 0
        found = [m.pii_type for m in matches]
        return AssertionResult(
            passed=passed,
            assertion_type="no_pii",
            detail="No PII detected" if passed else f"PII found: {found}",
            score=1.0 if passed else 0.0,
        )

    return _check


def no_injection_patterns() -> Assertion:
    """Assert output contains no prompt injection patterns.

    Uses PromptInjectionDetector at BALANCED level.
    """

    async def _check(result: ExecutionResult) -> AssertionResult:
        from nexus.safety.injection import DetectionLevel, PromptInjectionDetector

        detector = PromptInjectionDetector(level=DetectionLevel.BALANCED)
        text = _output(result)
        inj_result = detector.check(text)
        passed = not inj_result.detected
        return AssertionResult(
            passed=passed,
            assertion_type="no_injection_patterns",
            detail=(
                "No injection patterns detected"
                if passed
                else f"Injection detected (confidence={inj_result.confidence:.2f}): {inj_result.matched_patterns}"
            ),
            score=1.0 if passed else 0.0,
        )

    return _check
