"""Per-version quality metrics tracking backed by Dapr state."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from grampus.core.logging import get_logger

_log = get_logger(__name__)


class VersionMetrics(BaseModel):
    """Accumulated quality metrics for a single agent version."""

    version_id: str
    total_runs: int = 0
    error_count: int = 0
    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0
    eval_pass_count: int = 0
    eval_total: int = 0

    @property
    def eval_pass_rate(self) -> float | None:
        """Fraction of eval cases that passed, or None when no evals recorded."""
        return self.eval_pass_count / self.eval_total if self.eval_total > 0 else None

    @property
    def avg_cost_usd(self) -> float | None:
        """Mean cost per run, or None when no runs recorded."""
        return self.total_cost_usd / self.total_runs if self.total_runs > 0 else None

    @property
    def avg_latency_seconds(self) -> float | None:
        """Mean latency per run, or None when no runs recorded."""
        return self.total_latency_seconds / self.total_runs if self.total_runs > 0 else None

    @property
    def error_rate(self) -> float | None:
        """Fraction of runs that errored, or None when no runs recorded."""
        return self.error_count / self.total_runs if self.total_runs > 0 else None


class QualityTracker:
    """Records and retrieves per-version quality metrics backed by Dapr state."""

    _ENTITY = "version_metrics"

    def __init__(self, state_store: Any) -> None:
        self._state = state_store

    async def record_run(
        self,
        version_id: str,
        *,
        cost_usd: float = 0.0,
        latency_seconds: float = 0.0,
        error: bool = False,
    ) -> None:
        """Increment run counter and accumulate cost, latency, and error count."""
        metrics = await self.get_metrics(version_id)
        updated = metrics.model_copy(
            update={
                "total_runs": metrics.total_runs + 1,
                "total_cost_usd": metrics.total_cost_usd + cost_usd,
                "total_latency_seconds": metrics.total_latency_seconds + latency_seconds,
                "error_count": metrics.error_count + (1 if error else 0),
            }
        )
        await self._state.save(self._ENTITY, version_id, updated)
        _log.debug("metrics_run_recorded", version_id=version_id)

    async def record_eval_result(self, version_id: str, *, passed: bool) -> None:
        """Record the outcome of one eval assertion."""
        metrics = await self.get_metrics(version_id)
        updated = metrics.model_copy(
            update={
                "eval_total": metrics.eval_total + 1,
                "eval_pass_count": metrics.eval_pass_count + (1 if passed else 0),
            }
        )
        await self._state.save(self._ENTITY, version_id, updated)
        _log.debug("metrics_eval_recorded", version_id=version_id, passed=passed)

    async def get_metrics(self, version_id: str) -> VersionMetrics:
        """Load metrics for a version, or return a zero-valued object when not found."""
        result, _ = await self._state.get(self._ENTITY, version_id, VersionMetrics)
        if result is None:
            return VersionMetrics(version_id=version_id)
        return cast(VersionMetrics, result)
