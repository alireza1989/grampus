"""EvalSuite — runs collections of EvalCases against an AgentRunner."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nexus.core.logging import get_logger
from nexus.core.types import AgentDefinition, ExecutionResult
from nexus.evaluation.assertions import AssertionResult

logger = get_logger(__name__)


class EvalCase(BaseModel):
    """A single evaluation test case.

    Attributes:
        id: Unique identifier (auto-generated).
        name: Human-readable case name.
        description: Optional description.
        input: User input to send to the agent.
        tags: Labels for filtering (e.g. "smoke", "regression").
        assertions: List of Assertion callables to run.
        metadata: Arbitrary key-value metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    input: str
    tags: list[str] = Field(default_factory=list)
    assertions: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CaseResult(BaseModel):
    """Result of running one EvalCase.

    Attributes:
        case_id: ID of the EvalCase.
        case_name: Name of the EvalCase.
        passed: True only if all assertions passed.
        assertion_results: Per-assertion results.
        execution_result: The agent's ExecutionResult (may be None on error).
        error: Set if the agent raised an exception.
        duration_seconds: Wall-clock time for this case.
        tags: Tags copied from the EvalCase.
    """

    case_id: str
    case_name: str
    passed: bool
    assertion_results: list[AssertionResult]
    execution_result: ExecutionResult | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    tags: list[str] = Field(default_factory=list)


class SuiteResult(BaseModel):
    """Aggregate result from running a full EvalSuite.

    Attributes:
        suite_name: Name of the suite.
        total_cases: Number of cases actually run (after filtering).
        passed: Cases where all assertions passed.
        failed: Cases with one or more assertion failures.
        errors: Cases where the agent raised an exception.
        pass_rate: passed / total_cases.
        avg_duration_seconds: Mean per-case wall time.
        case_results: Ordered list of per-case results.
        run_at: UTC timestamp of run start.
        total_cost_usd: Sum of all execution costs.
        metadata: Arbitrary metadata.
    """

    suite_name: str
    total_cases: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    avg_duration_seconds: float
    case_results: list[CaseResult]
    run_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalSuite:
    """Runs a collection of EvalCases against an AgentRunner.

    Args:
        name: Suite name for reporting.
        agent_runner: The runner to test (duck-typed AgentRunner).
        agent_def: Agent definition passed to runner.run().
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
        session_id_prefix: str = "eval",
        concurrency: int = 1,
        tags: list[str] | None = None,
    ) -> None:
        self._name = name
        self._runner = agent_runner
        self._agent_def = agent_def
        self._prefix = session_id_prefix
        self._concurrency = max(1, concurrency)
        self._filter_tags: set[str] | None = set(tags) if tags else None
        self._cases: list[EvalCase] = []

    def add_case(self, case: EvalCase) -> EvalSuite:
        """Register a case. Returns self for chaining."""
        self._cases.append(case)
        return self

    def add_cases(self, cases: list[EvalCase]) -> EvalSuite:
        """Register multiple cases. Returns self for chaining."""
        for case in cases:
            self._cases.append(case)
        return self

    async def run(self) -> SuiteResult:
        """Execute all registered (and tag-filtered) cases.

        Returns:
            SuiteResult with aggregated pass/fail counts and per-case details.
        """
        filtered = self._filter_cases()
        if not filtered:
            return _empty_suite_result(self._name)

        sem = asyncio.Semaphore(self._concurrency)

        async def _run_with_sem(case: EvalCase) -> CaseResult:
            async with sem:
                return await self.run_case(case)

        case_results = await asyncio.gather(*(_run_with_sem(c) for c in filtered))
        return _build_suite_result(self._name, list(case_results))

    async def run_case(self, case: EvalCase) -> CaseResult:
        """Run a single EvalCase. Useful for debugging individual cases.

        Args:
            case: The EvalCase to execute.

        Returns:
            CaseResult with assertion details.
        """
        session_id = f"{self._prefix}:{case.id}"
        start = time.monotonic()
        try:
            exec_result: ExecutionResult = await self._runner.run(
                self._agent_def, case.input, session_id=session_id
            )
        except Exception as exc:
            duration = time.monotonic() - start
            logger.warning("eval_case_error", case=case.name, error=str(exc))
            return CaseResult(
                case_id=case.id,
                case_name=case.name,
                passed=False,
                assertion_results=[],
                error=str(exc),
                duration_seconds=duration,
                tags=case.tags,
            )

        duration = time.monotonic() - start
        assertion_results = await _run_assertions(case.assertions, exec_result)
        passed = all(ar.passed for ar in assertion_results)
        return CaseResult(
            case_id=case.id,
            case_name=case.name,
            passed=passed,
            assertion_results=assertion_results,
            execution_result=exec_result,
            duration_seconds=duration,
            tags=case.tags,
        )

    def _filter_cases(self) -> list[EvalCase]:
        if self._filter_tags is None:
            return list(self._cases)
        return [c for c in self._cases if set(c.tags) & self._filter_tags]


async def _run_assertions(assertions: list[Any], result: ExecutionResult) -> list[AssertionResult]:
    results = []
    for assertion in assertions:
        ar = await assertion(result)
        results.append(ar)
    return results


def _empty_suite_result(name: str) -> SuiteResult:
    return SuiteResult(
        suite_name=name,
        total_cases=0,
        passed=0,
        failed=0,
        errors=0,
        pass_rate=0.0,
        avg_duration_seconds=0.0,
        case_results=[],
    )


def _build_suite_result(name: str, case_results: list[CaseResult]) -> SuiteResult:
    total = len(case_results)
    errors = sum(1 for cr in case_results if cr.error is not None)
    passed = sum(1 for cr in case_results if cr.passed and cr.error is None)
    failed = total - passed - errors
    pass_rate = passed / total if total > 0 else 0.0
    avg_dur = sum(cr.duration_seconds for cr in case_results) / total if total > 0 else 0.0
    total_cost = sum(
        cr.execution_result.token_usage.cost_usd
        for cr in case_results
        if cr.execution_result is not None
    )
    return SuiteResult(
        suite_name=name,
        total_cases=total,
        passed=passed,
        failed=failed,
        errors=errors,
        pass_rate=pass_rate,
        avg_duration_seconds=avg_dur,
        case_results=case_results,
        total_cost_usd=total_cost,
    )
