"""Tests for CostTracker — accumulation, budget enforcement, pub/sub events."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import BudgetExceededError
from nexus.core.types import TokenUsage
from nexus.orchestration.cost_tracker import CostEvent, CostTracker
from nexus.orchestration.model_router import ModelSpec, ModelTier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spec(model_id: str = "test-model") -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        tier=ModelTier.BALANCED,
        provider="anthropic",
        input_cost_per_1k_tokens=0.003,
        output_cost_per_1k_tokens=0.015,
        context_window=200_000,
    )


def _usage(cost: float = 0.01, model: str = "test-model") -> TokenUsage:
    return TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost_usd=cost,
        model=model,
    )


def _tracker(
    *,
    budget: float | None = None,
    pubsub: object | None = None,
    agent_id: str = "agent-1",
    session_id: str = "session-1",
) -> CostTracker:
    return CostTracker(
        agent_id=agent_id,
        session_id=session_id,
        budget_usd=budget,
        pubsub=pubsub,
    )


def _mock_pubsub() -> MagicMock:
    ps = MagicMock()
    ps.publish = AsyncMock()
    return ps


# ---------------------------------------------------------------------------
# TestCostTrackerRecord
# ---------------------------------------------------------------------------


class TestCostTrackerRecord:
    @pytest.mark.asyncio
    async def test_record_accumulates_total_cost(self) -> None:
        tracker = _tracker()
        await tracker.record(_usage(0.01), step_name="step1", model_spec=_spec())
        await tracker.record(_usage(0.02), step_name="step2", model_spec=_spec())
        assert pytest.approx(tracker.summary().total_cost_usd) == 0.03

    @pytest.mark.asyncio
    async def test_record_accumulates_per_model_cost(self) -> None:
        tracker = _tracker()
        spec_a = _spec("model-a")
        spec_b = _spec("model-b")
        await tracker.record(_usage(0.01, "model-a"), step_name="s1", model_spec=spec_a)
        await tracker.record(_usage(0.05, "model-a"), step_name="s2", model_spec=spec_a)
        await tracker.record(_usage(0.02, "model-b"), step_name="s3", model_spec=spec_b)
        summary = tracker.summary()
        assert pytest.approx(summary.per_model["model-a"]) == 0.06
        assert pytest.approx(summary.per_model["model-b"]) == 0.02

    @pytest.mark.asyncio
    async def test_record_accumulates_per_step_cost(self) -> None:
        tracker = _tracker()
        spec = _spec()
        await tracker.record(_usage(0.01), step_name="plan", model_spec=spec)
        await tracker.record(_usage(0.03), step_name="plan", model_spec=spec)
        await tracker.record(_usage(0.02), step_name="execute", model_spec=spec)
        summary = tracker.summary()
        assert pytest.approx(summary.per_step["plan"]) == 0.04
        assert pytest.approx(summary.per_step["execute"]) == 0.02

    @pytest.mark.asyncio
    async def test_record_accumulates_token_counts(self) -> None:
        tracker = _tracker()
        spec = _spec()
        usage1 = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.01, model="m"
        )
        usage2 = TokenUsage(
            input_tokens=200, output_tokens=80, total_tokens=280, cost_usd=0.02, model="m"
        )
        await tracker.record(usage1, step_name="s1", model_spec=spec)
        await tracker.record(usage2, step_name="s2", model_spec=spec)
        summary = tracker.summary()
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 130

    @pytest.mark.asyncio
    async def test_record_publishes_cost_event_when_pubsub_configured(self) -> None:
        pubsub = _mock_pubsub()
        tracker = _tracker(pubsub=pubsub)
        await tracker.record(_usage(0.01), step_name="step1", model_spec=_spec("model-x"))
        pubsub.publish.assert_awaited_once()
        topic_arg = pubsub.publish.call_args[0][0]
        event_arg = pubsub.publish.call_args[0][1]
        assert topic_arg == "nexus.cost.events"
        assert isinstance(event_arg, CostEvent)
        assert event_arg.model_id == "model-x"
        assert pytest.approx(event_arg.cost_usd) == 0.01

    @pytest.mark.asyncio
    async def test_record_skips_publish_when_pubsub_none(self) -> None:
        tracker = _tracker(pubsub=None)
        # Should not raise
        await tracker.record(_usage(0.01), step_name="step1", model_spec=_spec())

    @pytest.mark.asyncio
    async def test_record_raises_budget_exceeded_after_limit_crossed(self) -> None:
        tracker = _tracker(budget=0.05)
        spec = _spec()
        await tracker.record(_usage(0.03), step_name="s1", model_spec=spec)
        with pytest.raises(BudgetExceededError) as exc_info:
            await tracker.record(_usage(0.03), step_name="s2", model_spec=spec)
        assert exc_info.value.code == "BUDGET_EXCEEDED"
        details = exc_info.value.details
        assert "budget_usd" in details
        assert "current_usd" in details

    @pytest.mark.asyncio
    async def test_record_emits_event_before_raising_budget_error(self) -> None:
        pubsub = _mock_pubsub()
        tracker = _tracker(budget=0.05, pubsub=pubsub)
        spec = _spec()
        await tracker.record(_usage(0.03), step_name="s1", model_spec=spec)
        with pytest.raises(BudgetExceededError):
            await tracker.record(_usage(0.03), step_name="s2", model_spec=spec)
        # publish called twice — once per record call, before the budget raise
        assert pubsub.publish.await_count == 2


# ---------------------------------------------------------------------------
# TestCostTrackerBudget
# ---------------------------------------------------------------------------


class TestCostTrackerBudget:
    def test_check_budget_passes_when_under_limit(self) -> None:
        tracker = _tracker(budget=1.0)
        tracker.check_budget(0.5)  # 0.0 + 0.5 < 1.0

    def test_check_budget_raises_when_over_limit(self) -> None:
        tracker = _tracker(budget=0.05)
        # Simulate existing spend via direct reset + record won't work here,
        # so we just check estimated alone exceeds budget
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget(0.06)
        assert exc_info.value.code == "BUDGET_EXCEEDED"
        assert "estimated_usd" in exc_info.value.details

    def test_check_budget_passes_when_budget_none(self) -> None:
        tracker = _tracker(budget=None)
        tracker.check_budget(9999.0)  # no limit → never raises

    @pytest.mark.asyncio
    async def test_check_budget_includes_estimated_cost(self) -> None:
        tracker = _tracker(budget=0.10)
        spec = _spec()
        # Spend 0.07
        await tracker.record(_usage(0.07), step_name="s1", model_spec=spec)
        # 0.07 + 0.04 = 0.11 > 0.10
        with pytest.raises(BudgetExceededError):
            tracker.check_budget(0.04)


# ---------------------------------------------------------------------------
# TestCostTrackerSummary
# ---------------------------------------------------------------------------


class TestCostTrackerSummary:
    @pytest.mark.asyncio
    async def test_summary_returns_correct_totals(self) -> None:
        tracker = _tracker()
        spec = _spec()
        await tracker.record(_usage(0.01), step_name="s1", model_spec=spec)
        await tracker.record(_usage(0.02), step_name="s2", model_spec=spec)
        summary = tracker.summary()
        assert pytest.approx(summary.total_cost_usd) == 0.03
        assert summary.agent_id == "agent-1"
        assert summary.session_id == "session-1"

    @pytest.mark.asyncio
    async def test_summary_per_model_breakdown(self) -> None:
        tracker = _tracker()
        spec_a = _spec("alpha")
        spec_b = _spec("beta")
        await tracker.record(_usage(0.10, "alpha"), step_name="s1", model_spec=spec_a)
        await tracker.record(_usage(0.20, "beta"), step_name="s2", model_spec=spec_b)
        summary = tracker.summary()
        assert pytest.approx(summary.per_model["alpha"]) == 0.10
        assert pytest.approx(summary.per_model["beta"]) == 0.20

    @pytest.mark.asyncio
    async def test_summary_per_step_breakdown(self) -> None:
        tracker = _tracker()
        spec = _spec()
        await tracker.record(_usage(0.05), step_name="plan", model_spec=spec)
        await tracker.record(_usage(0.05), step_name="plan", model_spec=spec)
        await tracker.record(_usage(0.10), step_name="execute", model_spec=spec)
        summary = tracker.summary()
        assert pytest.approx(summary.per_step["plan"]) == 0.10
        assert pytest.approx(summary.per_step["execute"]) == 0.10

    @pytest.mark.asyncio
    async def test_summary_zero_after_reset(self) -> None:
        tracker = _tracker()
        spec = _spec()
        await tracker.record(_usage(0.05), step_name="s1", model_spec=spec)
        tracker.reset()
        summary = tracker.summary()
        assert summary.total_cost_usd == 0.0
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0
        assert summary.per_model == {}
        assert summary.per_step == {}
        assert summary.event_count == 0

    @pytest.mark.asyncio
    async def test_event_count_increments_per_record(self) -> None:
        tracker = _tracker()
        spec = _spec()
        for i in range(5):
            await tracker.record(_usage(0.01), step_name=f"step_{i}", model_spec=spec)
        assert tracker.summary().event_count == 5
