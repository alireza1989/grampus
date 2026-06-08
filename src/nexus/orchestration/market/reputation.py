"""ReputationTracker — UCB-based per-agent reputation with Dapr persistence."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from nexus.core.logging import get_logger
from nexus.orchestration.market.types import ReputationRecord, TaskOutcome

_log = get_logger(__name__)

_UCB_EXPLORATION_CONSTANT = 2.0
_EMA_ALPHA = 0.2


class ReputationTracker:
    """UCB-based reputation tracking per agent (DRF, arXiv 2509.05764).

    Persists ReputationRecord per agent to Dapr state (namespace:
    "market:reputation"). Maintains an in-memory global task counter for
    the UCB denominator.

    UCB formula:
        ucb_bonus = sqrt(UCB_C * ln(max(1, total_global)) / max(1, agent_tasks))

    New agents receive maximum exploration bonus which decays as they
    accumulate history.

    Args:
        state_store: Optional DaprStateStore for persistence. When None,
            records are in-memory only.
    """

    def __init__(self, state_store: Any | None = None) -> None:
        self._store = state_store
        self._records: dict[str, ReputationRecord] = {}
        self._global_task_count: int = 0
        self._agent_mean_self_report: dict[str, list[float]] = {}

    async def get(self, agent_id: str) -> ReputationRecord:
        """Load reputation from Dapr or return defaults for a new agent.

        Args:
            agent_id: The agent whose record to retrieve.

        Returns:
            ReputationRecord — defaults (total_tasks=0, calibration_factor=1.0)
            for unknown agents.
        """
        if agent_id in self._records:
            return self._records[agent_id]
        if self._store is not None:
            stored, _ = await self._store.get("reputation", agent_id, ReputationRecord)
            if stored is not None:
                record: ReputationRecord = stored
                self._records[agent_id] = record
                return record
        default = ReputationRecord(agent_id=agent_id)
        self._records[agent_id] = default
        return default

    async def update(self, outcome: TaskOutcome) -> ReputationRecord:
        """Update reputation after a task completes.

        Updates success_rate (rolling), cost_accuracy (EMA alpha=0.2), and
        calibration_factor (EMA alpha=0.2). Recomputes UCB bonus.

        Args:
            outcome: The completed task outcome.

        Returns:
            The updated ReputationRecord.
        """
        record = await self.get(outcome.agent_id)
        self._increment_global_count()

        record.total_tasks += 1
        if outcome.actual_success:
            record.successful_tasks += 1
        record.success_rate = record.successful_tasks / record.total_tasks

        if outcome.actual_cost_usd > 0:
            bid_cost = max(outcome.actual_cost_usd, 1e-9)
            cost_ratio = outcome.actual_cost_usd / bid_cost
            record.cost_accuracy = (
                1.0 - _EMA_ALPHA
            ) * record.cost_accuracy + _EMA_ALPHA * cost_ratio

        mean_self_report = self._agent_mean_self_report.get(outcome.agent_id)
        if mean_self_report:
            mean_sr = sum(mean_self_report) / len(mean_self_report)
            if mean_sr > 0:
                new_factor = record.success_rate / mean_sr
                record.calibration_factor = (
                    1.0 - _EMA_ALPHA
                ) * record.calibration_factor + _EMA_ALPHA * new_factor
        else:
            if record.success_rate > 0:
                record.calibration_factor = (
                    1.0 - _EMA_ALPHA
                ) * record.calibration_factor + _EMA_ALPHA * record.success_rate

        record.ucb_confidence = _ucb_bonus(self._global_task_count, record.total_tasks)
        record.last_updated = datetime.now(UTC)

        self._records[outcome.agent_id] = record
        if self._store is not None:
            await self._store.save("reputation", outcome.agent_id, record)

        _log.debug(
            "reputation_updated",
            agent_id=outcome.agent_id,
            success_rate=record.success_rate,
            calibration_factor=record.calibration_factor,
        )
        return record

    async def calibration_factor(self, agent_id: str) -> float:
        """Return current calibration factor for discounting bid success estimates.

        Args:
            agent_id: The agent to look up.

        Returns:
            Float in (0, ∞); values < 1.0 indicate systematic over-reporting.
        """
        record = await self.get(agent_id)
        return record.calibration_factor

    async def ucb_bonus(self, agent_id: str) -> float:
        """Return current UCB exploration bonus for the agent.

        Args:
            agent_id: The agent to look up.

        Returns:
            Float ≥ 0; decreases as agent accumulates history.
        """
        record = await self.get(agent_id)
        return _ucb_bonus(self._global_task_count, record.total_tasks)

    def record_self_report(self, agent_id: str, self_reported_prob: float) -> None:
        """Track a self-reported success probability for calibration computation.

        Args:
            agent_id: The agent that submitted the self-report.
            self_reported_prob: The probability value reported in the bid.
        """
        if agent_id not in self._agent_mean_self_report:
            self._agent_mean_self_report[agent_id] = []
        self._agent_mean_self_report[agent_id].append(self_reported_prob)

    def _increment_global_count(self) -> None:
        """Increment global task counter for UCB denominator."""
        self._global_task_count += 1


def _ucb_bonus(global_count: int, agent_tasks: int) -> float:
    """Compute UCB1 exploration bonus.

    Args:
        global_count: Total tasks across all agents.
        agent_tasks: Tasks completed by this specific agent.

    Returns:
        sqrt(UCB_C * ln(max(2, global_count)) / max(1, agent_tasks))

    Note: min of 2 in the log ensures new agents always receive a positive
    exploration bonus even before any tasks have been completed globally.
    """
    return math.sqrt(
        _UCB_EXPLORATION_CONSTANT * math.log(max(2, global_count)) / max(1, agent_tasks)
    )
