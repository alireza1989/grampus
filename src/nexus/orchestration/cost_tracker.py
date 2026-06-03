"""Cost tracker — per-token accounting, budget enforcement, pub/sub events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from nexus.core.errors import BudgetExceededError
from nexus.core.logging import get_logger
from nexus.core.types import TokenUsage
from nexus.orchestration.model_router import ModelSpec

_log = get_logger(__name__)


class CostEvent(BaseModel):
    """Emitted via pub/sub on every tracked usage record."""

    agent_id: str
    session_id: str
    step_name: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cumulative_session_usd: float = 0.0
    cumulative_agent_usd: float = 0.0
    timestamp: datetime


class CostSummary(BaseModel):
    """Accumulated cost and token totals for an agent session."""

    agent_id: str
    session_id: str
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    per_model: dict[str, float]
    per_step: dict[str, float]
    event_count: int


class CostTracker:
    """Tracks token usage and cost per agent/session/step/model.

    Args:
        agent_id: Agent being tracked.
        session_id: Current session.
        budget_usd: Hard spend limit. None means unlimited.
        pubsub: Optional pub/sub object; must expose ``publish(topic, event)`` coroutine.
            When None, events are not published.
        cost_topic: Topic name for cost events.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        session_id: str,
        budget_usd: float | None = None,
        pubsub: Any | None = None,
        cost_topic: str = "nexus.cost.events",
        alert_evaluator: Any | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._session_id = session_id
        self._budget_usd = budget_usd
        self._pubsub = pubsub
        self._cost_topic = cost_topic
        self._alert_evaluator = alert_evaluator
        self._reset_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record(
        self,
        usage: TokenUsage,
        *,
        step_name: str,
        model_spec: ModelSpec,
    ) -> None:
        """Record usage from one LLM call.

        Accumulates totals, publishes a CostEvent, then checks the budget.
        Budget check happens after recording so the event is always emitted.

        Raises:
            BudgetExceededError: code="BUDGET_EXCEEDED" when total exceeds budget.
        """
        self._accumulate(usage, step_name=step_name, model_id=model_spec.model_id)

        event = CostEvent(
            agent_id=self._agent_id,
            session_id=self._session_id,
            step_name=step_name,
            model_id=model_spec.model_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            cumulative_session_usd=self._total_cost,
            cumulative_agent_usd=self._total_cost,
            timestamp=datetime.now(UTC),
        )
        await self._publish(event)
        await self._evaluate_alerts(event)

        _log.debug(
            "cost_recorded",
            agent=self._agent_id,
            step=step_name,
            model=model_spec.model_id,
            cost=usage.cost_usd,
            total=self._total_cost,
        )

        if self._budget_usd is not None and self._total_cost > self._budget_usd:
            raise BudgetExceededError(
                f"Budget exceeded: {self._total_cost:.6f} USD > {self._budget_usd:.6f} USD",
                code="BUDGET_EXCEEDED",
                details={
                    "budget_usd": self._budget_usd,
                    "current_usd": self._total_cost,
                    "estimated_usd": 0.0,
                },
                hint="Raise cost_budget_usd in AgentDefinition or break the task into smaller sub-tasks.",
            )

    def summary(self) -> CostSummary:
        """Return current accumulated totals. Pure computation, no I/O."""
        return CostSummary(
            agent_id=self._agent_id,
            session_id=self._session_id,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_cost_usd=self._total_cost,
            per_model=dict(self._per_model),
            per_step=dict(self._per_step),
            event_count=self._event_count,
        )

    def check_budget(self, estimated_cost_usd: float = 0.0) -> None:
        """Pre-flight budget check before an LLM call.

        Raises:
            BudgetExceededError: When current + estimated exceeds the budget limit.
        """
        if self._budget_usd is None:
            return
        projected = self._total_cost + estimated_cost_usd
        if projected > self._budget_usd:
            raise BudgetExceededError(
                f"Pre-flight budget check failed: projected {projected:.6f} USD > {self._budget_usd:.6f} USD",
                code="BUDGET_EXCEEDED",
                details={
                    "budget_usd": self._budget_usd,
                    "current_usd": self._total_cost,
                    "estimated_usd": estimated_cost_usd,
                },
                hint="The estimated cost for this step exceeds the remaining budget. Reduce max_tokens or increase cost_budget_usd.",
            )

    def reset(self) -> None:
        """Reset all accumulators."""
        self._reset_state()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._per_model: dict[str, float] = {}
        self._per_step: dict[str, float] = {}
        self._event_count: int = 0

    def _accumulate(self, usage: TokenUsage, *, step_name: str, model_id: str) -> None:
        self._total_cost += usage.cost_usd
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._per_model[model_id] = self._per_model.get(model_id, 0.0) + usage.cost_usd
        self._per_step[step_name] = self._per_step.get(step_name, 0.0) + usage.cost_usd
        self._event_count += 1

    async def _publish(self, event: CostEvent) -> None:
        if self._pubsub is None:
            return
        await self._pubsub.publish(self._cost_topic, event)

    async def _evaluate_alerts(self, event: CostEvent) -> None:
        if self._alert_evaluator is None:
            return
        try:
            await self._alert_evaluator.evaluate(
                event.agent_id,
                event.session_id,
                event.model_dump(mode="json"),
            )
        except Exception as exc:
            _log.warning("alert_evaluation_failed", error=str(exc))
