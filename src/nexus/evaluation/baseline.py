"""Quality baseline tracking and regression detection."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from nexus.core.logging import get_logger
from nexus.evaluation.suite import SuiteResult

logger = get_logger(__name__)


class BaselineRun(BaseModel):
    """One recorded eval run for comparison.

    Attributes:
        id: Unique run identifier.
        suite_name: Name of the eval suite.
        pass_rate: Fraction of cases that passed.
        avg_duration_seconds: Mean per-case duration.
        total_cost_usd: Total run cost.
        case_pass_rates: Map of case_id → passed boolean.
        prompt_version: Optional prompt version tag.
        recorded_at: UTC timestamp when this run was recorded.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    suite_name: str
    pass_rate: float
    avg_duration_seconds: float
    total_cost_usd: float
    case_pass_rates: dict[str, bool]
    prompt_version: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RegressionReport(BaseModel):
    """Comparison of a new run against a pinned baseline.

    Attributes:
        baseline_id: ID of the pinned BaselineRun.
        current_pass_rate: Pass rate in the new run.
        baseline_pass_rate: Pass rate in the baseline.
        delta: current - baseline (negative means regression).
        regressed: True if delta < -regression_threshold.
        regression_threshold: The threshold used.
        newly_failing: Case IDs that were passing, now failing.
        newly_passing: Case IDs that were failing, now passing.
        cost_delta_usd: current_cost - baseline_cost.
        duration_delta_seconds: current_avg_duration - baseline_avg_duration.
    """

    baseline_id: str
    current_pass_rate: float
    baseline_pass_rate: float
    delta: float
    regressed: bool
    regression_threshold: float
    newly_failing: list[str]
    newly_passing: list[str]
    cost_delta_usd: float
    duration_delta_seconds: float


class QualityBaseline:
    """Records eval runs and detects regressions against a pinned baseline.

    Args:
        regression_threshold: Pass-rate drop that triggers a regression flag.
            e.g. 0.05 = flag if pass_rate drops by more than 5 percentage points.
    """

    def __init__(self, *, regression_threshold: float = 0.05) -> None:
        self._threshold = regression_threshold
        self._runs: list[BaselineRun] = []
        self._pinned_id: str | None = None

    def record(
        self, suite_result: SuiteResult, *, prompt_version: str | None = None
    ) -> BaselineRun:
        """Record a suite run.

        Args:
            suite_result: The completed SuiteResult to record.
            prompt_version: Optional prompt version label.

        Returns:
            The stored BaselineRun.
        """
        case_pass_rates = {cr.case_id: cr.passed for cr in suite_result.case_results}
        run = BaselineRun(
            suite_name=suite_result.suite_name,
            pass_rate=suite_result.pass_rate,
            avg_duration_seconds=suite_result.avg_duration_seconds,
            total_cost_usd=suite_result.total_cost_usd,
            case_pass_rates=case_pass_rates,
            prompt_version=prompt_version,
        )
        self._runs.append(run)
        logger.info("baseline_run_recorded", run_id=run.id, pass_rate=run.pass_rate)
        return run

    def pin(self, run_id: str) -> None:
        """Pin a specific run as the baseline for future comparisons.

        Args:
            run_id: ID of the run to pin.

        Raises:
            ValueError: If run_id not found.
        """
        if not any(r.id == run_id for r in self._runs):
            raise ValueError(f"Run '{run_id}' not found")
        self._pinned_id = run_id
        logger.info("baseline_pinned", run_id=run_id)

    def pin_latest(self) -> BaselineRun:
        """Pin the most recently recorded run as the baseline.

        Returns:
            The pinned BaselineRun.

        Raises:
            ValueError: If no runs have been recorded.
        """
        if not self._runs:
            raise ValueError("No runs recorded yet")
        latest = self._runs[-1]
        self._pinned_id = latest.id
        return latest

    def compare(self, suite_result: SuiteResult) -> RegressionReport | None:
        """Compare suite_result against the pinned baseline.

        Args:
            suite_result: New suite result to compare.

        Returns:
            RegressionReport, or None if no baseline is pinned.
        """
        if self._pinned_id is None:
            return None
        baseline = next((r for r in self._runs if r.id == self._pinned_id), None)
        if baseline is None:
            return None
        return _build_regression_report(suite_result, baseline, self._threshold)

    def history(self) -> list[BaselineRun]:
        """Return all recorded runs sorted by recorded_at ascending."""
        return sorted(self._runs, key=lambda r: r.recorded_at)

    def pinned(self) -> BaselineRun | None:
        """Return the currently pinned baseline run."""
        if self._pinned_id is None:
            return None
        return next((r for r in self._runs if r.id == self._pinned_id), None)


def _build_regression_report(
    current: SuiteResult, baseline: BaselineRun, threshold: float
) -> RegressionReport:
    """Build a RegressionReport comparing current run to baseline."""
    current_rates = {cr.case_id: cr.passed for cr in current.case_results}
    baseline_rates = baseline.case_pass_rates

    all_ids = set(current_rates) | set(baseline_rates)
    newly_failing = [
        cid
        for cid in all_ids
        if baseline_rates.get(cid, False) and not current_rates.get(cid, False)
    ]
    newly_passing = [
        cid
        for cid in all_ids
        if not baseline_rates.get(cid, False) and current_rates.get(cid, False)
    ]

    delta = current.pass_rate - baseline.pass_rate
    return RegressionReport(
        baseline_id=baseline.id,
        current_pass_rate=current.pass_rate,
        baseline_pass_rate=baseline.pass_rate,
        delta=delta,
        regressed=delta < -threshold,
        regression_threshold=threshold,
        newly_failing=newly_failing,
        newly_passing=newly_passing,
        cost_delta_usd=current.total_cost_usd - baseline.total_cost_usd,
        duration_delta_seconds=(current.avg_duration_seconds - baseline.avg_duration_seconds),
    )
