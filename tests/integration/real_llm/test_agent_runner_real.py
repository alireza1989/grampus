"""AgentRunner real-LLM tests — requires RUN_REAL_LLM_TESTS=true.

Note: AgentRunner.run() does not pass tool definitions to the model client
(AgentDefinition.tools contains name strings only, not ToolDefinition objects).
Tool calling therefore does not occur through the standard runner loop, so math
questions are answered by the model's own computation rather than tool execution.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from grampus.core.errors import BudgetExceededError, OrchestrationError
from grampus.core.models.anthropic import AnthropicClient
from grampus.core.types import AgentDefinition
from grampus.orchestration.cost_tracker import CostTracker
from grampus.orchestration.runner import AgentRunner, RunnerConfig
from grampus.tools.executor import ToolExecutor
from grampus.tools.registry import ToolRegistry

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _make_executor() -> ToolExecutor:
    """Build a ToolExecutor with a minimal calculator tool."""
    registry = ToolRegistry()

    @registry.tool(name="calculator", description="Evaluate a mathematical expression")
    async def calculator(expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception as exc:
            return f"Error: {exc}"

    return ToolExecutor(registry)


@pytest.mark.asyncio
async def test_agent_answers_math_question(
    anthropic_client: AnthropicClient,
    record_cost: Callable[[float], None],
) -> None:
    """AgentRunner completes a math question and returns a correct answer.

    The runner does not pass tool definitions to the LLM, so the model
    computes sqrt(144)+15=27 directly without tool calling.
    """
    executor = _make_executor()
    runner = AgentRunner(
        model_client=anthropic_client,
        tool_executor=executor,
        config=RunnerConfig(max_iterations=5),
    )
    agent_def = AgentDefinition(
        name="math-agent",
        model=ANTHROPIC_MODEL,
        system_prompt="You are a math assistant. Answer concisely with just the number.",
    )
    result = await runner.run(agent_def, "What is sqrt(144) + 15?", session_id="real-llm-math-1")
    assert result.output is not None
    assert "27" in result.output
    record_cost(result.token_usage.cost_usd)


@pytest.mark.asyncio
async def test_agent_respects_max_iterations(
    anthropic_client: AnthropicClient,
    record_cost: Callable[[float], None],
) -> None:
    """AgentRunner with max_iterations=1 terminates within one step."""
    executor = _make_executor()
    runner = AgentRunner(
        model_client=anthropic_client,
        tool_executor=executor,
        config=RunnerConfig(max_iterations=1),
    )
    agent_def = AgentDefinition(
        name="one-shot-agent",
        model=ANTHROPIC_MODEL,
        system_prompt="You are a helpful assistant. Answer briefly.",
    )
    # A complex-sounding task; without tools the model will respond directly in 1 step.
    # Either a successful result or OrchestrationError(MAX_ITERATIONS_EXCEEDED) is acceptable.
    try:
        result = await runner.run(
            agent_def,
            "Briefly summarise what Python is.",
            session_id="real-llm-max-iter-1",
        )
        assert result.steps_taken <= 1
        record_cost(result.token_usage.cost_usd)
    except OrchestrationError as exc:
        assert exc.code == "MAX_ITERATIONS_EXCEEDED"
        # No cost to record — the iteration limit was hit


@pytest.mark.asyncio
async def test_budget_enforcement(
    anthropic_client: AnthropicClient,
) -> None:
    """CostTracker with a $0.000001 budget raises BudgetExceededError after the first LLM call."""
    executor = _make_executor()
    cost_tracker = CostTracker(
        agent_id="budget-test",
        session_id="s-budget",
        budget_usd=0.000001,
    )
    runner = AgentRunner(
        model_client=anthropic_client,
        tool_executor=executor,
        cost_tracker=cost_tracker,
    )
    agent_def = AgentDefinition(
        name="budget-test",
        model=ANTHROPIC_MODEL,
        system_prompt="You are a helpful assistant.",
    )
    with pytest.raises(BudgetExceededError) as exc_info:
        await runner.run(agent_def, "Say hello.", session_id="s-budget")
    assert exc_info.value.code == "BUDGET_EXCEEDED"
