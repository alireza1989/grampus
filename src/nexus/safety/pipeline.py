"""Unified safety middleware — wraps every agent action with safety checks."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from nexus.core.errors import SafetyError
from nexus.core.logging import get_logger
from nexus.core.types import ToolCall, ToolResult

_log = get_logger(__name__)


class SafetyViolation(BaseModel):
    """Structured record of a detected safety issue."""

    violation_type: str
    severity: str
    detail: str
    blocked: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SafetyPipelineConfig(BaseModel):
    """Which check categories the pipeline runs."""

    check_user_input: bool = True
    check_tool_results: bool = True
    check_llm_output: bool = True
    check_memory_writes: bool = True
    log_violations: bool = True


class SafetyPipeline:
    """Middleware that wraps agent actions with safety checks.

    All checks are applied in order. A BLOCK-level result raises SafetyError.
    Non-blocking detections are logged and returned as SafetyViolation records.

    Args:
        injection_detector: Optional injection detector. Skipped if None.
        pii_detector: Optional PII detector. Skipped if None.
        action_guard: Optional action guard. Skipped if None.
        config: Which check categories to enable.
    """

    def __init__(
        self,
        *,
        injection_detector: object | None = None,
        pii_detector: object | None = None,
        action_guard: object | None = None,
        config: SafetyPipelineConfig | None = None,
    ) -> None:
        self._injection_detector = injection_detector
        self._pii_detector = pii_detector
        self._action_guard = action_guard
        self._config = config or SafetyPipelineConfig()
        self._violations: list[SafetyViolation] = []

    async def check_input(self, text: str) -> tuple[str, list[SafetyViolation]]:
        """Check user input for injection then PII.

        Args:
            text: Raw user input.

        Returns:
            Tuple of (possibly-redacted text, new violations from this call).

        Raises:
            SafetyError: code="INPUT_BLOCKED" if injection is detected at threshold.
        """
        if not self._config.check_user_input:
            return text, []

        new_violations: list[SafetyViolation] = []
        text = self._run_injection_check(
            text, error_code="INPUT_BLOCKED", new_violations=new_violations
        )
        text, pii_violations = self._run_pii_check(text)
        new_violations.extend(pii_violations)
        self._record(new_violations)
        return text, new_violations

    async def check_tool_result(
        self, result: ToolResult
    ) -> tuple[ToolResult, list[SafetyViolation]]:
        """Check tool result output for injection and PII.

        Args:
            result: The ToolResult from tool execution.

        Returns:
            Tuple of (possibly-redacted result, new violations).

        Raises:
            SafetyError: code="TOOL_RESULT_BLOCKED" if injection is detected.
        """
        if not self._config.check_tool_results:
            return result, []

        if result.output is None:
            return result, []

        new_violations: list[SafetyViolation] = []
        output_str = str(result.output)
        output_str = self._run_injection_check(
            output_str, error_code="TOOL_RESULT_BLOCKED", new_violations=new_violations
        )
        output_str, pii_violations = self._run_pii_check(output_str)
        new_violations.extend(pii_violations)
        self._record(new_violations)
        updated = result.model_copy(update={"output": output_str})
        return updated, new_violations

    async def check_llm_output(self, text: str) -> tuple[str, list[SafetyViolation]]:
        """Check LLM response for PII only (injection not blocked to avoid loops).

        Args:
            text: LLM-generated text to check.

        Returns:
            Tuple of (possibly-redacted text, new violations).
        """
        if not self._config.check_llm_output:
            return text, []

        new_violations: list[SafetyViolation] = []
        # Detect injection but do NOT block — just record as violation
        inj_violations = self._detect_injection_only(text)
        new_violations.extend(inj_violations)
        text, pii_violations = self._run_pii_check(text)
        new_violations.extend(pii_violations)
        self._record(new_violations)
        return text, new_violations

    async def check_tool_call(
        self,
        tool_call: ToolCall,
        *,
        calls_this_turn: int = 0,
        consecutive_calls: int = 0,
    ) -> tuple[ToolCall, list[SafetyViolation]]:
        """Check a tool call against action guard policy.

        Args:
            tool_call: The tool call to evaluate.
            calls_this_turn: Calls already made this turn.
            consecutive_calls: Consecutive calls without a non-tool step.

        Returns:
            Tuple of (tool_call unchanged, new violations).

        Raises:
            SafetyError: code="ACTION_BLOCKED" if the action guard denies the call.
        """
        if self._action_guard is None:
            return tool_call, []

        from nexus.safety.action_guard import SafetyActionGuard

        guard: SafetyActionGuard = self._action_guard  # type: ignore[assignment]
        check_result = guard.check_tool_call(
            tool_call, calls_this_turn=calls_this_turn, consecutive_calls=consecutive_calls
        )

        if not check_result.allowed:
            v = SafetyViolation(
                violation_type="action_blocked",
                severity="high",
                detail=check_result.reason or "Tool call denied by policy",
                blocked=True,
            )
            self._violations.append(v)
            raise SafetyError(
                check_result.reason or "Tool call blocked by safety policy",
                code="ACTION_BLOCKED",
            )

        return tool_call, []

    def get_violations(self) -> list[SafetyViolation]:
        """Return all violations recorded in this pipeline instance's lifetime."""
        return list(self._violations)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_injection_check(
        self, text: str, *, error_code: str, new_violations: list[SafetyViolation]
    ) -> str:
        """Run injection check; raise SafetyError with error_code if detected."""
        if self._injection_detector is None:
            return text
        from nexus.safety.injection import PromptInjectionDetector

        detector: PromptInjectionDetector = self._injection_detector  # type: ignore[assignment]
        result = detector.check(text)
        if result.detected:
            v = SafetyViolation(
                violation_type="injection",
                severity="critical",
                detail=f"Injection detected (confidence={result.confidence:.2f}): {result.matched_patterns}",
                blocked=True,
            )
            self._violations.append(v)
            new_violations.append(v)
            _log.warning("safety.injection_blocked", confidence=result.confidence, code=error_code)
            raise SafetyError(
                f"Prompt injection detected in input (confidence={result.confidence:.2f})",
                code=error_code,
            )
        return text

    def _detect_injection_only(self, text: str) -> list[SafetyViolation]:
        """Detect injection without blocking; returns violations list."""
        if self._injection_detector is None:
            return []
        from nexus.safety.injection import PromptInjectionDetector

        detector: PromptInjectionDetector = self._injection_detector  # type: ignore[assignment]
        result = detector.check(text)
        if result.detected:
            v = SafetyViolation(
                violation_type="injection",
                severity="medium",
                detail=f"Injection-like content in LLM output (confidence={result.confidence:.2f})",
                blocked=False,
            )
            return [v]
        return []

    def _run_pii_check(self, text: str) -> tuple[str, list[SafetyViolation]]:
        """Run PII check; return (redacted_text, violations)."""
        if self._pii_detector is None:
            return text, []
        from nexus.safety.pii import PIIDetector

        detector: PIIDetector = self._pii_detector  # type: ignore[assignment]
        pii_result = detector.scan(text)
        violations: list[SafetyViolation] = []
        if pii_result.matches:
            types = list({m.pii_type for m in pii_result.matches})
            v = SafetyViolation(
                violation_type="pii",
                severity="medium",
                detail=f"PII detected and {pii_result.action_taken}: {types}",
                blocked=False,
            )
            violations.append(v)
        return pii_result.redacted, violations

    def _record(self, violations: list[SafetyViolation]) -> None:
        """Add violations to lifetime list and optionally log them."""
        for v in violations:
            if v not in self._violations:
                self._violations.append(v)
            if self._config.log_violations:
                _log.info(
                    "safety.violation",
                    type=v.violation_type,
                    severity=v.severity,
                    blocked=v.blocked,
                )
