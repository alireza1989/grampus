"""Cost tracking alert rules, evaluator, and cooldown state models."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from nexus.core.logging import get_logger

_log = get_logger(__name__)


class ThresholdType(StrEnum):
    """What cost metric to measure against the threshold."""

    ABSOLUTE_USD = "absolute_usd"
    PER_SESSION_USD = "per_session_usd"
    PER_HOUR_USD = "per_hour_usd"
    PER_DAY_USD = "per_day_usd"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRule(BaseModel):
    """Configuration for a cost alert threshold."""

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent_id: str | None = None
    threshold_type: ThresholdType
    threshold_usd: float
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: int = 3600
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: dict[str, str] = Field(default_factory=dict)


class AlertEvent(BaseModel):
    """Emitted when a cost alert rule fires."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str
    rule_name: str
    agent_id: str | None
    session_id: str | None
    severity: AlertSeverity
    threshold_type: ThresholdType
    threshold_usd: float
    actual_usd: float
    fired_at: datetime = Field(default_factory=datetime.utcnow)
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlertState(BaseModel):
    """Cooldown tracking — stored in Dapr state or in-memory."""

    rule_id: str
    last_fired_at: datetime
    fire_count: int = 1


class AlertEvaluator:
    """Evaluates cost events against alert rules and dispatches notifications.

    Args:
        rules: Initial list of AlertRule configurations.
        dispatcher: NotificationDispatcher used to send fired alerts.
        state_store: Optional DaprStateStore for cooldown persistence.
            If None, an in-memory dict is used.
    """

    def __init__(
        self,
        rules: list[AlertRule],
        dispatcher: Any,
        state_store: Any | None = None,
    ) -> None:
        self._rules: list[AlertRule] = list(rules)
        self._dispatcher = dispatcher
        self._state_store = state_store
        self._mem_cooldown: dict[str, AlertState] = {}
        # Rolling window: agent_id → [(timestamp, cost_usd), ...]
        self._rolling_window: dict[str, list[tuple[datetime, float]]] = {}

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, rule_id: str) -> None:
        self._rules = [r for r in self._rules if r.rule_id != rule_id]

    def list_rules(self) -> list[AlertRule]:
        return list(self._rules)

    async def evaluate(
        self,
        agent_id: str,
        session_id: str,
        cost_event: dict[str, Any],
    ) -> list[AlertEvent]:
        """Check all matching rules; fire alerts for those triggered and not in cooldown."""
        event_ts = _parse_timestamp(cost_event.get("timestamp"))
        cost_usd = float(cost_event.get("cost_usd", 0.0))

        window = self._rolling_window.setdefault(agent_id, [])
        window.append((event_ts, cost_usd))

        fired: list[AlertEvent] = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.agent_id is not None and rule.agent_id != agent_id:
                continue

            actual = self._compute_metric(rule.threshold_type, agent_id, cost_event)
            if actual <= rule.threshold_usd:
                continue

            if await self._is_in_cooldown(rule, agent_id):
                continue

            msg = _format_message(rule, agent_id, actual)
            event = AlertEvent(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                agent_id=agent_id,
                session_id=session_id,
                severity=rule.severity,
                threshold_type=rule.threshold_type,
                threshold_usd=rule.threshold_usd,
                actual_usd=actual,
                message=msg,
            )
            await self._record_fire(rule, agent_id, event)
            await self._dispatcher.dispatch(event)
            fired.append(event)

        return fired

    def _compute_metric(
        self,
        threshold_type: ThresholdType,
        agent_id: str,
        cost_event: dict[str, Any],
    ) -> float:
        if threshold_type == ThresholdType.ABSOLUTE_USD:
            return float(cost_event.get("cumulative_agent_usd", 0.0))
        if threshold_type == ThresholdType.PER_SESSION_USD:
            return float(cost_event.get("cumulative_session_usd", 0.0))
        if threshold_type == ThresholdType.PER_HOUR_USD:
            return self._rolling_sum(agent_id, hours=1)
        if threshold_type == ThresholdType.PER_DAY_USD:
            return self._rolling_sum(agent_id, hours=24)
        return 0.0

    def _rolling_sum(self, agent_id: str, *, hours: int) -> float:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        window = self._rolling_window.get(agent_id, [])
        pruned = [(ts, cost) for ts, cost in window if ts >= cutoff]
        self._rolling_window[agent_id] = pruned
        return sum(cost for _, cost in pruned)

    async def _is_in_cooldown(self, rule: AlertRule, agent_id: str) -> bool:
        key = f"alerts:{rule.rule_id}:{agent_id}"
        state = await self._get_cooldown_state(key)
        if state is None:
            return False
        elapsed = (datetime.utcnow() - state.last_fired_at).total_seconds()
        return elapsed < rule.cooldown_seconds

    async def _record_fire(self, rule: AlertRule, agent_id: str, event: AlertEvent) -> None:
        key = f"alerts:{rule.rule_id}:{agent_id}"
        state = await self._get_cooldown_state(key)
        count = (state.fire_count + 1) if state is not None else 1
        new_state = AlertState(
            rule_id=rule.rule_id,
            last_fired_at=event.fired_at,
            fire_count=count,
        )
        await self._set_cooldown_state(key, new_state)

    async def _get_cooldown_state(self, key: str) -> AlertState | None:
        if self._state_store is not None:
            result: AlertState | None = await self._state_store.get(key, AlertState)
            return result
        return self._mem_cooldown.get(key)

    async def _set_cooldown_state(self, key: str, state: AlertState) -> None:
        if self._state_store is not None:
            await self._state_store.set(key, state)
        else:
            self._mem_cooldown[key] = state


def _parse_timestamp(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    return datetime.utcnow()


def _format_message(rule: AlertRule, agent_id: str, actual_usd: float) -> str:
    if rule.threshold_type == ThresholdType.PER_SESSION_USD:
        return f"Agent {agent_id} spent ${actual_usd:.2f} (limit ${rule.threshold_usd:.2f}/session)"
    if rule.threshold_type == ThresholdType.PER_HOUR_USD:
        return (
            f"Agent {agent_id} spent ${actual_usd:.2f} in the last hour"
            f" (limit ${rule.threshold_usd:.2f}/hr)"
        )
    if rule.threshold_type == ThresholdType.PER_DAY_USD:
        return (
            f"Agent {agent_id} spent ${actual_usd:.2f} today (limit ${rule.threshold_usd:.2f}/day)"
        )
    return (
        f"Agent {agent_id} total spend ${actual_usd:.2f} (limit ${rule.threshold_usd:.2f} lifetime)"
    )
