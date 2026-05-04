"""E2E scenario: cost budget stops agent mid-execution."""

from __future__ import annotations

import pytest

from nexus.core.errors import BudgetExceededError
from nexus.core.types import AgentDefinition, TokenUsage, ToolCall, ToolParameter
from nexus.orchestration.cost_tracker import CostTracker
from nexus.orchestration.model_router import ModelSpec, ModelTier


@pytest.mark.integration
class TestBudgetEnforcementE2E:
    async def test_budget_stops_agent_mid_loop(self) -> None:
        from nexus.orchestration.runner import AgentRunner, RunnerConfig
        from nexus.tools.executor import ToolExecutor
        from nexus.tools.registry import ToolRegistry
        from tests.integration.conftest import MockModelClient

        registry = ToolRegistry()

        @registry.tool(
            name="echo",
            description="Echo",
            parameters=[ToolParameter(name="text", type="string", description="", required=True)],
        )
        async def echo(text: str) -> str:
            return text

        client = MockModelClient()
        for _ in range(10):
            client.add_response(
                text=None,
                tool_calls=[ToolCall(id=f"tc-b{_}", name="echo", arguments={"text": "x"})],
                cost_usd=0.0005,
            )

        cost_tracker = CostTracker(
            agent_id="budget-stop", session_id="bstop", budget_usd=0.001
        )
        executor = ToolExecutor(registry, timeout_seconds=5.0)
        runner = AgentRunner(
            client,
            executor,
            cost_tracker=cost_tracker,
            config=RunnerConfig(max_iterations=10, enable_memory=False),
        )

        with pytest.raises(BudgetExceededError) as exc_info:
            await runner.run(
                AgentDefinition(
                    name="budget-stop",
                    model="mock-model",
                    system_prompt="",
                    tools=["echo"],
                    max_iterations=10,
                    temperature=0.0,
                    memory_enabled=False,
                    cost_budget_usd=0.001,
                ),
                "Keep calling tools.",
                session_id="bstop",
            )
        assert exc_info.value.code == "BUDGET_EXCEEDED"

    async def test_preflight_budget_check_before_llm_call(self) -> None:

        cost_tracker = CostTracker(
            agent_id="preflight-agent", session_id="pf1", budget_usd=0.50
        )
        spec = ModelSpec(
            model_id="mock",
            tier=ModelTier.BALANCED,
            provider="mock",
            input_cost_per_1k_tokens=0.0,
            output_cost_per_1k_tokens=0.0,
            context_window=200_000,
        )
        existing = TokenUsage(
            input_tokens=100, output_tokens=100, total_tokens=200, cost_usd=0.40, model="mock"
        )
        await cost_tracker.record(existing, step_name="step1", model_spec=spec)

        with pytest.raises(BudgetExceededError):
            cost_tracker.check_budget(estimated_cost_usd=0.20)

    async def test_budget_not_exceeded_within_limit(self) -> None:
        cost_tracker = CostTracker(
            agent_id="within-budget", session_id="wb1", budget_usd=1.0
        )
        spec = ModelSpec(
            model_id="mock",
            tier=ModelTier.BALANCED,
            provider="mock",
            input_cost_per_1k_tokens=0.0,
            output_cost_per_1k_tokens=0.0,
            context_window=200_000,
        )
        usage = TokenUsage(
            input_tokens=10, output_tokens=10, total_tokens=20, cost_usd=0.001, model="mock"
        )
        await cost_tracker.record(usage, step_name="step1", model_spec=spec)
        cost_tracker.check_budget(estimated_cost_usd=0.001)
        summary = cost_tracker.summary()
        assert summary.total_cost_usd == pytest.approx(0.001)
