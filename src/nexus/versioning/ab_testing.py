"""A/B test lifecycle management and auto-promotion."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel

from nexus.core.errors import VersioningError
from nexus.core.logging import get_logger
from nexus.versioning.metrics import QualityTracker, VersionMetrics
from nexus.versioning.stats import two_proportion_z_test
from nexus.versioning.store import VersionStore
from nexus.versioning.types import ABTestConfig, ABTestResult, SuccessMetric

_log = get_logger(__name__)


class _ActiveTestIndex(BaseModel):
    """Stores the experiment_id of the currently active test for an agent."""

    experiment_id: str | None = None


class ABTestManager:
    """Manages A/B test lifecycle and auto-promotion."""

    _ENTITY = "ab_tests"

    def __init__(
        self,
        store: VersionStore,
        quality_tracker: QualityTracker,
        *,
        model_client: Any = None,
    ) -> None:
        self._store = store
        self._tracker = quality_tracker
        self._model_client = model_client

    async def start_test(
        self,
        agent_id: str,
        *,
        control_version_id: str,
        treatment_version_id: str,
        traffic_split: float = 0.1,
        success_metric: SuccessMetric = SuccessMetric.eval_pass_rate,
        auto_promote_threshold: float = 0.05,
        min_samples: int = 100,
    ) -> ABTestConfig:
        """Start a new A/B experiment for an agent.

        Raises:
            VersioningError: code="TEST_ALREADY_ACTIVE" when one already exists.
        """
        existing = await self.get_active_test(agent_id)
        if existing is not None:
            raise VersioningError(
                f"An A/B test is already active for agent '{agent_id}'",
                code="TEST_ALREADY_ACTIVE",
                details={"experiment_id": existing.experiment_id, "agent_id": agent_id},
            )

        config = ABTestConfig(
            experiment_id=str(uuid.uuid4()),
            agent_id=agent_id,
            control_version_id=control_version_id,
            treatment_version_id=treatment_version_id,
            traffic_split=traffic_split,
            success_metric=success_metric,
            auto_promote_threshold=auto_promote_threshold,
            min_samples=min_samples,
            created_at=datetime.now(UTC),
        )
        await self._save_test(config)
        await self._set_active_index(agent_id, config.experiment_id)
        _log.debug(
            "ab_test_started",
            experiment_id=config.experiment_id,
            agent_id=agent_id,
        )
        return config

    async def stop_test(self, experiment_id: str) -> ABTestConfig:
        """Mark an experiment as inactive."""
        test = await self.get_test(experiment_id)
        if test is None:
            raise VersioningError(
                f"Experiment '{experiment_id}' not found",
                code="VERSION_NOT_FOUND",
                details={"experiment_id": experiment_id},
            )
        stopped = test.model_copy(update={"active": False, "ended_at": datetime.now(UTC)})
        await self._save_test(stopped)
        await self._clear_active_index(test.agent_id)
        _log.debug("ab_test_stopped", experiment_id=experiment_id)
        return stopped

    async def get_active_test(self, agent_id: str) -> ABTestConfig | None:
        """Return the currently active experiment for an agent, or None."""
        index = await self._load_active_index(agent_id)
        if index is None or index.experiment_id is None:
            return None
        test = await self.get_test(index.experiment_id)
        if test is None or not test.active:
            return None
        return test

    async def get_test(self, experiment_id: str) -> ABTestConfig | None:
        """Load an experiment by ID."""
        state_store = self._store._state
        result, _ = await state_store.get(self._ENTITY, experiment_id, ABTestConfig)
        return cast("ABTestConfig | None", result)

    async def evaluate(self, experiment_id: str) -> ABTestResult:
        """Compute current stats and auto-promote if threshold met."""
        test = await self.get_test(experiment_id)
        if test is None:
            raise VersioningError(
                f"Experiment '{experiment_id}' not found",
                code="VERSION_NOT_FOUND",
                details={"experiment_id": experiment_id},
            )

        ctrl = await self._tracker.get_metrics(test.control_version_id)
        trt = await self._tracker.get_metrics(test.treatment_version_id)

        p_value, significant, winner_id, recommendation = self._compute_result(test, ctrl, trt)

        if (
            significant
            and winner_id is not None
            and ctrl.total_runs >= test.min_samples
            and trt.total_runs >= test.min_samples
            and test.active
        ):
            await self._auto_promote(test, winner_id)

        return ABTestResult(
            experiment_id=experiment_id,
            control_metrics=ctrl,
            treatment_metrics=trt,
            p_value=p_value,
            significant=significant,
            winner_version_id=winner_id,
            recommendation=recommendation,
        )

    async def _auto_promote(self, test: ABTestConfig, winner_id: str) -> None:
        """Deploy winner and mark test as ended."""
        from nexus.versioning.manager import VersionManager

        mgr = VersionManager(self._store, agent_id=test.agent_id)
        await mgr.deploy(winner_id)

        ended = test.model_copy(
            update={
                "active": False,
                "ended_at": datetime.now(UTC),
                "winner_version_id": winner_id,
            }
        )
        await self._save_test(ended)
        await self._clear_active_index(test.agent_id)
        _log.info(
            "ab_test_auto_promoted",
            experiment_id=test.experiment_id,
            winner=winner_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_result(
        self,
        test: ABTestConfig,
        ctrl: VersionMetrics,
        trt: VersionMetrics,
    ) -> tuple[float | None, bool, str | None, str]:
        """Return (p_value, significant, winner_id, recommendation)."""
        metric = test.success_metric

        if metric == SuccessMetric.eval_pass_rate:
            if ctrl.eval_total == 0 or trt.eval_total == 0:
                return None, False, None, "no winner yet — insufficient eval data"
            try:
                p_value = two_proportion_z_test(
                    ctrl.eval_total,
                    ctrl.eval_pass_count,
                    trt.eval_total,
                    trt.eval_pass_count,
                )
            except ValueError:
                return None, False, None, "no winner yet — insufficient data"

            significant = p_value < test.auto_promote_threshold
            ctrl_rate = ctrl.eval_pass_rate or 0.0
            trt_rate = trt.eval_pass_rate or 0.0
            winner_id = None
            if significant:
                winner_id = (
                    test.treatment_version_id if trt_rate > ctrl_rate else test.control_version_id
                )
                recommendation = (
                    "promote treatment"
                    if winner_id == test.treatment_version_id
                    else "keep control"
                )
            else:
                recommendation = "no winner yet — not statistically significant"
            return p_value, significant, winner_id, recommendation

        # Continuous metrics: simple mean comparison with 10% threshold
        ctrl_val = getattr(ctrl, metric.value)
        trt_val = getattr(trt, metric.value)

        if ctrl_val is None or trt_val is None:
            return None, False, None, "no winner yet — insufficient data"

        diff_pct = abs(trt_val - ctrl_val) / max(abs(ctrl_val), 1e-9)
        significant = diff_pct > 0.10

        if not significant:
            return None, False, None, "no winner yet — difference < 10%"

        # For cost/latency, lower is better; for error_rate, lower is better
        lower_is_better = metric in (
            SuccessMetric.avg_cost_usd,
            SuccessMetric.avg_latency_seconds,
            SuccessMetric.error_rate,
        )
        treatment_wins = (trt_val < ctrl_val) if lower_is_better else (trt_val > ctrl_val)
        winner_id = test.treatment_version_id if treatment_wins else test.control_version_id
        recommendation = (
            "promote treatment" if winner_id == test.treatment_version_id else "keep control"
        )
        return None, significant, winner_id, recommendation

    async def _save_test(self, config: ABTestConfig) -> None:
        state_store = self._store._state
        await state_store.save(self._ENTITY, config.experiment_id, config)

    async def _load_active_index(self, agent_id: str) -> _ActiveTestIndex | None:
        state_store = self._store._state
        result, _ = await state_store.get(self._ENTITY, f"active:{agent_id}", _ActiveTestIndex)
        return cast("_ActiveTestIndex | None", result)

    async def _set_active_index(self, agent_id: str, experiment_id: str) -> None:
        state_store = self._store._state
        index = _ActiveTestIndex(experiment_id=experiment_id)
        await state_store.save(self._ENTITY, f"active:{agent_id}", index)

    async def _clear_active_index(self, agent_id: str) -> None:
        state_store = self._store._state
        index = _ActiveTestIndex(experiment_id=None)
        await state_store.save(self._ENTITY, f"active:{agent_id}", index)
