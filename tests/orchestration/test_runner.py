"""Tests for AgentRunner — ReAct loop, tool execution, memory, cost, persistence."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import BudgetExceededError, OrchestrationError
from nexus.core.models.base import ModelResponse
from nexus.core.types import (
    AgentDefinition,
    AgentState,
    AgentStatus,
    ExecutionResult,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from nexus.memory.manager import MemoryRecallResult
from nexus.orchestration.cost_tracker import CostSummary
from nexus.orchestration.runner import AgentRunner, RunnerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_def(name: str = "test-agent") -> AgentDefinition:
    return AgentDefinition(name=name, model="test-model")


def _token_usage(cost: float = 0.001) -> TokenUsage:
    return TokenUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cost_usd=cost,
        model="test-model",
    )


def _model_response(
    content: str | None = "Hello!",
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls or [],
        token_usage=_token_usage(),
        model="test-model",
        stop_reason="end_turn",
    )


def _tool_call(name: str = "my_tool", call_id: str = "tc-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={})


def _tool_result(call_id: str = "tc-1") -> ToolResult:
    return ToolResult(tool_call_id=call_id, output="tool output", duration_ms=5)


def _cost_summary(agent_id: str = "test-agent") -> CostSummary:
    return CostSummary(
        agent_id=agent_id,
        session_id="s1",
        total_input_tokens=10,
        total_output_tokens=5,
        total_cost_usd=0.001,
        per_model={},
        per_step={},
        event_count=1,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_client() -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=_model_response())
    return client


@pytest.fixture
def tool_executor() -> AsyncMock:
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_tool_result())
    return executor


@pytest.fixture
def memory_manager() -> AsyncMock:
    manager = AsyncMock()
    manager.recall = AsyncMock(return_value=MemoryRecallResult(episodic=[], semantic=[], query=""))
    manager.remember = AsyncMock()
    return manager


@pytest.fixture
def cost_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.record = AsyncMock()
    tracker.summary = MagicMock(return_value=_cost_summary())
    return tracker


@pytest.fixture
def state_store() -> AsyncMock:
    store = AsyncMock()
    store.save = AsyncMock()
    store.get = AsyncMock(return_value=(None, ""))
    return store


@pytest.fixture
def runner(model_client: AsyncMock, tool_executor: AsyncMock) -> AgentRunner:
    return AgentRunner(model_client, tool_executor)


# ---------------------------------------------------------------------------
# TestAgentRunnerBasic
# ---------------------------------------------------------------------------


class TestAgentRunnerBasic:
    async def test_run_returns_execution_result(self, runner: AgentRunner) -> None:
        result = await runner.run(_agent_def(), "Hello", session_id="s1")
        assert isinstance(result, ExecutionResult)

    async def test_run_appends_user_message_to_state(self, runner: AgentRunner) -> None:
        result = await runner.run(_agent_def(), "Hello", session_id="s1")
        roles = [m.role for m in result.messages]
        assert Role.USER in roles

    async def test_run_calls_model_client(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        r = AgentRunner(model_client, tool_executor)
        await r.run(_agent_def(), "Hello", session_id="s1")
        model_client.complete.assert_called_once()

    async def test_run_sets_output_from_final_assistant_response(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        model_client.complete.return_value = _model_response(content="Final answer")
        r = AgentRunner(model_client, tool_executor)
        result = await r.run(_agent_def(), "Hello", session_id="s1")
        assert result.output == "Final answer"

    async def test_run_populates_steps_taken(self, runner: AgentRunner) -> None:
        result = await runner.run(_agent_def(), "Hello", session_id="s1")
        assert result.steps_taken >= 1

    async def test_run_populates_duration_seconds(self, runner: AgentRunner) -> None:
        result = await runner.run(_agent_def(), "Hello", session_id="s1")
        assert result.duration_seconds >= 0.0

    async def test_run_with_no_tool_calls_completes_in_one_step(self, runner: AgentRunner) -> None:
        result = await runner.run(_agent_def(), "Hello", session_id="s1")
        assert result.steps_taken == 1


# ---------------------------------------------------------------------------
# TestAgentRunnerToolCalls
# ---------------------------------------------------------------------------


class TestAgentRunnerToolCalls:
    async def test_run_executes_tool_calls_from_llm_response(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        tc = _tool_call()
        model_client.complete.side_effect = [
            _model_response(content=None, tool_calls=[tc]),
            _model_response(content="Done"),
        ]
        r = AgentRunner(model_client, tool_executor)
        await r.run(_agent_def(), "Do something", session_id="s1")
        tool_executor.execute.assert_called_once_with(tc)

    async def test_run_appends_tool_results_to_messages(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        tc = _tool_call()
        model_client.complete.side_effect = [
            _model_response(content=None, tool_calls=[tc]),
            _model_response(content="Done"),
        ]
        r = AgentRunner(model_client, tool_executor)
        result = await r.run(_agent_def(), "Do something", session_id="s1")
        roles = [m.role for m in result.messages]
        assert Role.TOOL in roles

    async def test_run_increments_tool_calls_made(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        tc = _tool_call()
        model_client.complete.side_effect = [
            _model_response(content=None, tool_calls=[tc]),
            _model_response(content="Done"),
        ]
        r = AgentRunner(model_client, tool_executor)
        result = await r.run(_agent_def(), "Do something", session_id="s1")
        assert result.tool_calls_made == 1

    async def test_run_continues_loop_after_tool_execution(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        tc = _tool_call()
        model_client.complete.side_effect = [
            _model_response(content=None, tool_calls=[tc]),
            _model_response(content="Final"),
        ]
        r = AgentRunner(model_client, tool_executor)
        result = await r.run(_agent_def(), "Do something", session_id="s1")
        assert model_client.complete.call_count == 2
        assert result.steps_taken == 2

    async def test_run_raises_max_iterations_exceeded(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        tc = _tool_call()
        model_client.complete.return_value = _model_response(content=None, tool_calls=[tc])
        r = AgentRunner(model_client, tool_executor, config=RunnerConfig(max_iterations=3))
        with pytest.raises(OrchestrationError) as exc_info:
            await r.run(_agent_def(), "Loop forever", session_id="s1")
        assert exc_info.value.code == "MAX_ITERATIONS_EXCEEDED"


# ---------------------------------------------------------------------------
# TestAgentRunnerMemory
# ---------------------------------------------------------------------------


class TestAgentRunnerMemory:
    async def test_run_recalls_memory_when_enabled(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        memory_manager: AsyncMock,
    ) -> None:
        r = AgentRunner(model_client, tool_executor, memory_manager=memory_manager)
        await r.run(_agent_def(), "Hello", session_id="s1")
        memory_manager.recall.assert_called_once()

    async def test_run_stores_turn_in_memory_after_completion(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        memory_manager: AsyncMock,
    ) -> None:
        r = AgentRunner(model_client, tool_executor, memory_manager=memory_manager)
        await r.run(_agent_def(), "Hello", session_id="s1")
        memory_manager.remember.assert_called_once()
        assert memory_manager.remember.call_args.kwargs["session_id"] == "s1"

    async def test_run_skips_memory_when_manager_is_none(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        r = AgentRunner(model_client, tool_executor, memory_manager=None)
        result = await r.run(_agent_def(), "Hello", session_id="s1")
        assert isinstance(result, ExecutionResult)


# ---------------------------------------------------------------------------
# TestAgentRunnerCost
# ---------------------------------------------------------------------------


class TestAgentRunnerCost:
    async def test_run_records_cost_when_tracker_configured(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        cost_tracker: MagicMock,
    ) -> None:
        r = AgentRunner(model_client, tool_executor, cost_tracker=cost_tracker)
        await r.run(_agent_def(), "Hello", session_id="s1")
        cost_tracker.record.assert_called_once()

    async def test_run_propagates_budget_exceeded_error(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        tracker = MagicMock()
        tracker.record = AsyncMock(
            side_effect=BudgetExceededError("Over budget", code="BUDGET_EXCEEDED", details={})
        )
        r = AgentRunner(model_client, tool_executor, cost_tracker=tracker)
        with pytest.raises(BudgetExceededError):
            await r.run(_agent_def(), "Hello", session_id="s1")

    def test_cost_summary_returns_summary_when_tracker_set(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        cost_tracker: MagicMock,
    ) -> None:
        r = AgentRunner(model_client, tool_executor, cost_tracker=cost_tracker)
        assert r.cost_summary() is not None

    def test_cost_summary_returns_none_when_no_tracker(
        self, model_client: AsyncMock, tool_executor: AsyncMock
    ) -> None:
        r = AgentRunner(model_client, tool_executor)
        assert r.cost_summary() is None


# ---------------------------------------------------------------------------
# TestAgentRunnerPersistence
# ---------------------------------------------------------------------------


class TestAgentRunnerPersistence:
    async def test_run_persists_state_when_store_configured(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        state_store: AsyncMock,
    ) -> None:
        r = AgentRunner(model_client, tool_executor, state_store=state_store)
        await r.run(_agent_def(), "Hello", session_id="s1")
        state_store.save.assert_called_once()

    async def test_resume_loads_state_and_continues(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        state_store: AsyncMock,
    ) -> None:
        paused_state = AgentState(
            agent_id="test-agent",
            session_id="s1",
            status=AgentStatus.WAITING_FOR_HUMAN,
            messages=[Message(role=Role.USER, content="original")],
        )
        state_store.get.return_value = (paused_state, "etag1")
        r = AgentRunner(model_client, tool_executor, state_store=state_store)
        result = await r.resume("test-agent", "s1", "human reply")
        assert isinstance(result, ExecutionResult)

    async def test_resume_raises_no_state_found(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        state_store: AsyncMock,
    ) -> None:
        state_store.get.return_value = (None, "")
        r = AgentRunner(model_client, tool_executor, state_store=state_store)
        with pytest.raises(OrchestrationError) as exc_info:
            await r.resume("missing-agent", "s1", "reply")
        assert exc_info.value.code == "NO_STATE_FOUND"

    async def test_resume_raises_agent_not_waiting(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        state_store: AsyncMock,
    ) -> None:
        completed_state = AgentState(
            agent_id="test-agent",
            session_id="s1",
            status=AgentStatus.COMPLETED,
        )
        state_store.get.return_value = (completed_state, "etag1")
        r = AgentRunner(model_client, tool_executor, state_store=state_store)
        with pytest.raises(OrchestrationError) as exc_info:
            await r.resume("test-agent", "s1", "reply")
        assert exc_info.value.code == "AGENT_NOT_WAITING"
