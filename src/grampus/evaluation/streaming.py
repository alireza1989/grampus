"""Streaming evaluation: StreamingResult, StreamingEvalSuite, and assertion factories."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from grampus.core.types import (
    AgentDefinition,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from grampus.evaluation.assertions import AssertionResult
from grampus.evaluation.suite import (
    CaseResult,
    SuiteResult,
    _build_suite_result,
    _empty_suite_result,
)


class StreamingResult(BaseModel):
    """Metrics and raw data collected from one streaming agent execution."""

    first_token_seconds: float | None
    total_seconds: float
    inter_chunk_delays: list[float]
    full_output: str
    output_length: int
    chunk_count: int
    chunk_deltas: list[str]
    token_usage: TokenUsage | None
    agent_ended: bool
    tool_calls_made: int
    error: str | None
    events: list[StreamEvent]

    model_config = ConfigDict(arbitrary_types_allowed=True)


@runtime_checkable
class StreamingAssertion(Protocol):
    """Callable (async) that takes a StreamingResult and returns AssertionResult."""

    async def __call__(self, result: StreamingResult) -> AssertionResult: ...


class StreamingEvalCase(BaseModel):
    """A single streaming evaluation test case."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    input: str
    tags: list[str] = Field(default_factory=list)
    assertions: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


async def _collect_stream(
    runner: Any,
    agent_def: AgentDefinition,
    user_input: str,
    session_id: str,
) -> StreamingResult:
    """Drive runner.stream() and collect all metrics into a StreamingResult."""
    start = time.monotonic()
    first_token_mono: float | None = None
    last_chunk_mono: float | None = None
    chunk_deltas: list[str] = []
    inter_chunk_delays: list[float] = []
    token_usage: TokenUsage | None = None
    agent_ended = False
    tool_calls_made = 0
    events: list[StreamEvent] = []
    error: str | None = None

    try:
        async for event in runner.stream(agent_def, user_input, session_id=session_id):
            now = time.monotonic()
            events.append(event)

            if event.event_type == StreamEventType.TOKEN and event.chunk and event.chunk.delta:
                delta = event.chunk.delta
                if first_token_mono is None:
                    first_token_mono = now
                if last_chunk_mono is not None:
                    inter_chunk_delays.append(now - last_chunk_mono)
                last_chunk_mono = now
                chunk_deltas.append(delta)

            elif event.event_type == StreamEventType.TOOL_CALL_START:
                tool_calls_made += 1

            elif event.event_type == StreamEventType.AGENT_END:
                agent_ended = True
                if event.chunk and event.chunk.token_usage:
                    token_usage = event.chunk.token_usage

    except Exception as exc:
        error = str(exc)

    total_seconds = time.monotonic() - start
    full_output = "".join(chunk_deltas)
    first_token_seconds = (first_token_mono - start) if first_token_mono is not None else None

    return StreamingResult(
        first_token_seconds=first_token_seconds,
        total_seconds=total_seconds,
        inter_chunk_delays=inter_chunk_delays,
        full_output=full_output,
        output_length=len(full_output),
        chunk_count=len(chunk_deltas),
        chunk_deltas=chunk_deltas,
        token_usage=token_usage,
        agent_ended=agent_ended,
        tool_calls_made=tool_calls_made,
        error=error,
        events=events,
    )


async def _run_streaming_assertions(
    assertions: list[Any],
    result: StreamingResult,
) -> list[AssertionResult]:
    results = []
    for assertion in assertions:
        ar = await assertion(result)
        results.append(ar)
    return results


class StreamingEvalSuite:
    """Runs StreamingEvalCases against AgentRunner.stream() and collects metrics.

    Returns the same SuiteResult / CaseResult types as EvalSuite for reporter
    compatibility. CaseResult.streaming_result carries the detailed metrics.

    Args:
        name: Suite name for reporting.
        agent_runner: The runner to test (duck-typed AgentRunner).
        agent_def: Agent definition passed to runner.stream().
        session_id_prefix: Prefix for per-case session IDs.
        concurrency: Max cases to run in parallel.
        tags: If set, only run cases whose tags intersect this set.
    """

    def __init__(
        self,
        name: str,
        *,
        agent_runner: Any,
        agent_def: AgentDefinition,
        session_id_prefix: str = "stream-eval",
        concurrency: int = 1,
        tags: list[str] | None = None,
    ) -> None:
        self._name = name
        self._runner = agent_runner
        self._agent_def = agent_def
        self._prefix = session_id_prefix
        self._concurrency = max(1, concurrency)
        self._filter_tags: set[str] | None = set(tags) if tags else None
        self._cases: list[StreamingEvalCase] = []

    def add_case(self, case: StreamingEvalCase) -> StreamingEvalSuite:
        """Register a case. Returns self for chaining."""
        self._cases.append(case)
        return self

    def add_cases(self, cases: list[StreamingEvalCase]) -> StreamingEvalSuite:
        """Register multiple cases. Returns self for chaining."""
        for case in cases:
            self._cases.append(case)
        return self

    async def run(self) -> SuiteResult:
        """Execute all registered (and tag-filtered) cases."""
        filtered = self._filter_cases()
        if not filtered:
            return _empty_suite_result(self._name)

        sem = asyncio.Semaphore(self._concurrency)

        async def _run_with_sem(case: StreamingEvalCase) -> CaseResult:
            async with sem:
                return await self.run_case(case)

        case_results = await asyncio.gather(*(_run_with_sem(c) for c in filtered))
        return _build_suite_result(self._name, list(case_results))

    async def run_case(self, case: StreamingEvalCase) -> CaseResult:
        """Run a single StreamingEvalCase."""
        session_id = f"{self._prefix}:{case.id}"
        start = time.monotonic()

        streaming_result = await _collect_stream(
            self._runner, self._agent_def, case.input, session_id
        )
        duration = time.monotonic() - start

        if streaming_result.error:
            return CaseResult(
                case_id=case.id,
                case_name=case.name,
                passed=False,
                assertion_results=[],
                streaming_result=streaming_result,
                error=streaming_result.error,
                duration_seconds=duration,
                tags=case.tags,
            )

        assertion_results = await _run_streaming_assertions(case.assertions, streaming_result)
        passed = all(ar.passed for ar in assertion_results)
        return CaseResult(
            case_id=case.id,
            case_name=case.name,
            passed=passed,
            assertion_results=assertion_results,
            streaming_result=streaming_result,
            duration_seconds=duration,
            tags=case.tags,
        )

    def _filter_cases(self) -> list[StreamingEvalCase]:
        if self._filter_tags is None:
            return list(self._cases)
        return [c for c in self._cases if set(c.tags) & self._filter_tags]


# ---------------------------------------------------------------------------
# Streaming assertion factories
# ---------------------------------------------------------------------------


def first_token_within(max_seconds: float) -> StreamingAssertion:
    """Assert Time-To-First-Token <= max_seconds.

    Args:
        max_seconds: Maximum allowed TTFT in seconds.
    """

    async def _check(result: StreamingResult) -> AssertionResult:
        if result.first_token_seconds is None:
            return AssertionResult(
                passed=False,
                assertion_type="first_token_within",
                detail="No tokens received (TTFT unmeasured)",
                score=0.0,
            )
        ttft = result.first_token_seconds
        passed = ttft <= max_seconds
        return AssertionResult(
            passed=passed,
            assertion_type="first_token_within",
            detail=f"TTFT={ttft:.2f}s {'<=' if passed else '>'} limit={max_seconds:.2f}s",
            score=1.0 if passed else 0.0,
        )

    return _check


def stream_completes() -> StreamingAssertion:
    """Assert the stream reached AGENT_END without errors."""

    async def _check(result: StreamingResult) -> AssertionResult:
        if result.error is not None:
            return AssertionResult(
                passed=False,
                assertion_type="stream_completes",
                detail=f"Stream raised an error: {result.error}",
                score=0.0,
            )
        if not result.agent_ended:
            return AssertionResult(
                passed=False,
                assertion_type="stream_completes",
                detail="Stream did not reach AGENT_END",
                score=0.0,
            )
        return AssertionResult(
            passed=True,
            assertion_type="stream_completes",
            detail="Stream completed normally",
            score=1.0,
        )

    return _check


def no_stall(max_gap_seconds: float) -> StreamingAssertion:
    """Assert no inter-chunk gap exceeds max_gap_seconds.

    Args:
        max_gap_seconds: Maximum allowed gap between consecutive TOKEN chunks.
    """

    async def _check(result: StreamingResult) -> AssertionResult:
        delays = result.inter_chunk_delays
        if not delays:
            return AssertionResult(
                passed=True,
                assertion_type="no_stall",
                detail=f"Max inter-chunk gap=0.00s <= limit={max_gap_seconds:.2f}s (0 gaps)",
                score=1.0,
            )
        for idx, gap in enumerate(delays):
            if gap > max_gap_seconds:
                return AssertionResult(
                    passed=False,
                    assertion_type="no_stall",
                    detail=(
                        f"Stall detected: gap={gap:.2f}s > limit={max_gap_seconds:.2f}s "
                        f"at chunk index {idx}"
                    ),
                    score=0.0,
                )
        max_gap = max(delays)
        return AssertionResult(
            passed=True,
            assertion_type="no_stall",
            detail=(
                f"Max inter-chunk gap={max_gap:.2f}s <= limit={max_gap_seconds:.2f}s "
                f"({len(delays)} gaps)"
            ),
            score=1.0,
        )

    return _check


def min_throughput(min_chunks_per_second: float) -> StreamingAssertion:
    """Assert chunk throughput >= min_chunks_per_second.

    Args:
        min_chunks_per_second: Minimum required streaming throughput.
    """

    async def _check(result: StreamingResult) -> AssertionResult:
        if result.chunk_count == 0:
            return AssertionResult(
                passed=False,
                assertion_type="min_throughput",
                detail="No chunks received (throughput=0)",
                score=0.0,
            )
        throughput = result.chunk_count / result.total_seconds if result.total_seconds > 0 else 0.0
        passed = throughput >= min_chunks_per_second
        return AssertionResult(
            passed=passed,
            assertion_type="min_throughput",
            detail=(
                f"Throughput={throughput:.1f} chunks/s "
                f"{'>' if passed else '<'} min={min_chunks_per_second:.1f}"
            ),
            score=1.0 if passed else 0.0,
        )

    return _check


def stream_contains(expected: str, *, case_sensitive: bool = True) -> StreamingAssertion:
    """Assert expected substring appears in streamed full_output.

    Args:
        expected: Substring to search for.
        case_sensitive: Whether the match is case-sensitive.
    """

    async def _check(result: StreamingResult) -> AssertionResult:
        haystack = result.full_output if case_sensitive else result.full_output.lower()
        needle = expected if case_sensitive else expected.lower()
        passed = needle in haystack
        return AssertionResult(
            passed=passed,
            assertion_type="stream_contains",
            detail=f"'{expected}' {'found' if passed else 'not found'} in stream output",
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=result.full_output[:200],
        )

    return _check


def stream_not_empty() -> StreamingAssertion:
    """Assert at least one non-empty TOKEN chunk was delivered."""

    async def _check(result: StreamingResult) -> AssertionResult:
        passed = result.chunk_count > 0
        return AssertionResult(
            passed=passed,
            assertion_type="stream_not_empty",
            detail=f"chunk_count={result.chunk_count} ({'>' if passed else '=='} 0)",
            score=1.0 if passed else 0.0,
        )

    return _check


def stream_output_length(
    *, min_chars: int | None = None, max_chars: int | None = None
) -> StreamingAssertion:
    """Assert streamed output character length is within bounds.

    Args:
        min_chars: Minimum character count (inclusive).
        max_chars: Maximum character count (inclusive).
    """

    async def _check(result: StreamingResult) -> AssertionResult:
        length = result.output_length
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
            assertion_type="stream_output_length",
            detail=detail,
            score=1.0 if passed else 0.0,
        )

    return _check


def _repetition_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    overlap = sum(1 for ca, cb in zip(a, b, strict=False) if ca == cb)
    return overlap / max(len(a), len(b))


def no_repetition(*, window: int = 100, threshold: float = 0.8) -> StreamingAssertion:
    """Detect degenerate output repetition (common failure in local models).

    Checks the last `window` chars against the preceding `window` chars.
    Only meaningful when output_length >= 2 * window.

    Args:
        window: Number of characters per comparison window.
        threshold: Ratio >= this value triggers a failure.
    """

    async def _check(result: StreamingResult) -> AssertionResult:
        text = result.full_output
        if len(text) < 2 * window:
            return AssertionResult(
                passed=True,
                assertion_type="no_repetition",
                detail=f"Output too short for repetition check (length={len(text)} < {2 * window})",
                score=1.0,
            )
        b = text[-window:]
        a = text[-(2 * window) : -window]
        ratio = _repetition_ratio(a, b)
        passed = ratio < threshold
        return AssertionResult(
            passed=passed,
            assertion_type="no_repetition",
            detail=(
                f"{'No repetition detected' if passed else 'Repetition detected'} "
                f"(ratio={ratio:.2f})" + (f" >= threshold={threshold:.2f}" if not passed else "")
            ),
            score=1.0 if passed else 0.0,
        )

    return _check


def chunk_count_between(
    *, min_chunks: int | None = None, max_chunks: int | None = None
) -> StreamingAssertion:
    """Assert number of TOKEN chunks is within bounds.

    Args:
        min_chunks: Minimum number of chunks (inclusive).
        max_chunks: Maximum number of chunks (inclusive).
    """

    async def _check(result: StreamingResult) -> AssertionResult:
        count = result.chunk_count
        passed = True
        reasons = []
        if min_chunks is not None and count < min_chunks:
            passed = False
            reasons.append(f"count {count} < min {min_chunks}")
        if max_chunks is not None and count > max_chunks:
            passed = False
            reasons.append(f"count {count} > max {max_chunks}")
        detail = f"chunk_count={count}" + (f": {', '.join(reasons)}" if reasons else "")
        return AssertionResult(
            passed=passed,
            assertion_type="chunk_count_between",
            detail=detail,
            score=1.0 if passed else 0.0,
        )

    return _check


def token_usage_reported() -> StreamingAssertion:
    """Assert token_usage is populated in the AGENT_END chunk.

    Required for cost tracking and billing correctness.
    """

    async def _check(result: StreamingResult) -> AssertionResult:
        passed = result.token_usage is not None
        return AssertionResult(
            passed=passed,
            assertion_type="token_usage_reported",
            detail="Token usage reported" if passed else "No token usage in AGENT_END chunk",
            score=1.0 if passed else 0.0,
        )

    return _check


# Resolve forward references: suite.py uses TYPE_CHECKING to avoid a circular
# import, so CaseResult.streaming_result's type annotation is a string at class
# definition time. Rebuilding here makes Pydantic resolve it to the real class.
CaseResult.model_rebuild()
SuiteResult.model_rebuild()
