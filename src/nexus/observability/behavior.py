"""Behavioral monitoring and anomaly detection for agents."""

from __future__ import annotations

from collections import Counter, deque
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from nexus.core.logging import get_logger

logger = get_logger(__name__)

_TurnRecord = tuple[float, list[str], int]  # (cost_usd, tool_names, error_count)


class AnomalyType(StrEnum):
    """Categories of behavioral anomalies."""

    COST_SPIKE = "cost_spike"
    TOOL_USAGE_SHIFT = "tool_usage_shift"
    ERROR_RATE_SPIKE = "error_rate_spike"
    MEMORY_ACCESS_ANOMALY = "memory_access_anomaly"


class Anomaly(BaseModel):
    """A detected behavioral anomaly."""

    anomaly_type: AnomalyType
    agent_id: str
    detected_at: datetime
    severity: float  # 0.0 – 1.0
    description: str
    context: dict[str, Any] = Field(default_factory=dict)


class AgentBehaviorProfile(BaseModel):
    """Rolling behavioral baseline for one agent."""

    agent_id: str
    window_size: int = 20
    avg_cost_per_turn: float = 0.0
    avg_tool_calls_per_turn: float = 0.0
    avg_errors_per_turn: float = 0.0
    top_tools: list[str] = Field(default_factory=list)
    turn_count: int = 0


def _severity(observed: float, baseline: float, threshold: float) -> float:
    """Compute anomaly severity capped at 1.0."""
    if baseline == 0:
        return 1.0 if observed > 0 else 0.0
    return min(observed / (baseline * threshold), 1.0)


class BehaviorMonitor:
    """Tracks per-agent behavioral patterns and detects anomalies.

    Maintains a rolling window of turn-level observations. After each turn is
    recorded, checks for anomalies against the baseline.

    Args:
        agent_id: Agent being monitored.
        cost_spike_threshold: Multiplier above avg_cost triggering COST_SPIKE.
        error_spike_threshold: Multiplier above avg_errors for ERROR_RATE_SPIKE.
        tool_shift_threshold: Fraction of new tools triggering TOOL_USAGE_SHIFT.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        cost_spike_threshold: float = 3.0,
        error_spike_threshold: float = 5.0,
        tool_shift_threshold: float = 0.5,
    ) -> None:
        self._agent_id = agent_id
        self._cost_threshold = cost_spike_threshold
        self._error_threshold = error_spike_threshold
        self._tool_threshold = tool_shift_threshold
        self._window_size = 20
        self._anomalies: list[Anomaly] = []
        self._window: deque[_TurnRecord] = deque(maxlen=self._window_size)
        self._turn_count: int = 0

    def record_turn(
        self,
        *,
        cost_usd: float,
        tool_names: list[str],
        error_count: int,
    ) -> list[Anomaly]:
        """Record one agent turn and return any anomalies detected.

        Args:
            cost_usd: Total cost incurred this turn.
            tool_names: Tools invoked during this turn.
            error_count: Number of errors encountered.

        Returns:
            List of new Anomaly objects (empty if baseline not yet established).
        """
        # Snapshot baseline before appending so new tools don't pollute top_tools
        window_was_full = len(self._window) >= self._window_size
        baseline = self._baseline() if window_was_full else None

        self._window.append((cost_usd, list(tool_names), error_count))
        self._turn_count += 1

        if not window_was_full or baseline is None:
            return []

        found = self._detect_anomalies(cost_usd, tool_names, error_count, baseline)
        self._anomalies.extend(found)
        return found

    def _baseline(self) -> tuple[float, float, list[str]]:
        """Compute (avg_cost, avg_errors, top_tools) from the window."""
        costs = [r[0] for r in self._window]
        errors = [r[2] for r in self._window]
        tool_counter: Counter[str] = Counter()
        for _, names, _ in self._window:
            tool_counter.update(names)
        top_tools = [t for t, _ in tool_counter.most_common(5)]
        avg_cost = sum(costs) / len(costs)
        avg_errors = sum(errors) / len(errors)
        return avg_cost, avg_errors, top_tools

    def _detect_anomalies(
        self,
        cost_usd: float,
        tool_names: list[str],
        error_count: int,
        baseline: tuple[float, float, list[str]],
    ) -> list[Anomaly]:
        detected: list[Anomaly] = []
        avg_cost, avg_errors, top_tools = baseline
        now = datetime.now(UTC)

        if avg_cost > 0 and cost_usd > avg_cost * self._cost_threshold:
            sev = _severity(cost_usd, avg_cost, self._cost_threshold)
            detected.append(
                Anomaly(
                    anomaly_type=AnomalyType.COST_SPIKE,
                    agent_id=self._agent_id,
                    detected_at=now,
                    severity=sev,
                    description=f"Cost {cost_usd:.4f} exceeds {self._cost_threshold}× avg {avg_cost:.4f}",
                    context={"cost_usd": cost_usd, "avg_cost": avg_cost},
                )
            )

        if avg_errors > 0 and error_count > avg_errors * self._error_threshold:
            sev = _severity(float(error_count), avg_errors, self._error_threshold)
            detected.append(
                Anomaly(
                    anomaly_type=AnomalyType.ERROR_RATE_SPIKE,
                    agent_id=self._agent_id,
                    detected_at=now,
                    severity=sev,
                    description=f"Errors {error_count} exceeds {self._error_threshold}× avg {avg_errors:.2f}",
                    context={"error_count": error_count, "avg_errors": avg_errors},
                )
            )

        if tool_names:
            top_set = set(top_tools)
            new_fraction = sum(1 for t in tool_names if t not in top_set) / len(tool_names)
            if new_fraction > self._tool_threshold:
                sev = min(new_fraction, 1.0)
                detected.append(
                    Anomaly(
                        anomaly_type=AnomalyType.TOOL_USAGE_SHIFT,
                        agent_id=self._agent_id,
                        detected_at=now,
                        severity=sev,
                        description=f"{new_fraction:.0%} tools are new (threshold {self._tool_threshold:.0%})",
                        context={"tool_names": tool_names, "top_tools": top_tools},
                    )
                )

        return detected

    def profile(self) -> AgentBehaviorProfile:
        """Return the current behavioral profile (snapshot).

        Returns:
            AgentBehaviorProfile with rolling-window statistics.
        """
        if not self._window:
            return AgentBehaviorProfile(agent_id=self._agent_id, turn_count=self._turn_count)
        avg_cost, avg_errors, top_tools = self._baseline()
        avg_tool_calls = sum(len(r[1]) for r in self._window) / len(self._window)
        return AgentBehaviorProfile(
            agent_id=self._agent_id,
            window_size=self._window_size,
            avg_cost_per_turn=avg_cost,
            avg_tool_calls_per_turn=avg_tool_calls,
            avg_errors_per_turn=avg_errors,
            top_tools=top_tools,
            turn_count=self._turn_count,
        )

    def anomalies(self) -> list[Anomaly]:
        """Return all anomalies detected so far."""
        return list(self._anomalies)

    def reset(self) -> None:
        """Reset all state. Useful for testing."""
        self._anomalies = []
        self._window.clear()
        self._turn_count = 0
