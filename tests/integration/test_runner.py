"""Integration tests for AgentRunner: ReAct loop, memory, cost, budget."""

from __future__ import annotations

import pytest

from grampus.core.errors import BudgetExceededError, OrchestrationError
from grampus.core.types import (
    AgentDefinition,
    AgentStatus,
    ToolCall,
)
from tests.integration.conftest import FakeStateStore, MockModelClient, make_session_id


def _agent_def(
    name: str = "test-agent",
    budget: float | None = None,
    max_iter: int = 5,
) -> AgentDefinition:
    return AgentDefinition(
        name=name,
        model="mock-model",
        system_prompt="You are helpful.",
        tools=["echo"],
        max_iterations=max_iter,
        temperature=0.0,
        memory_enabled=False,
        cost_budget_usd=budget,
    )


@pytest.mark.integration
class TestAgentRunnerIntegration:
    async def test_single_shot_run_returns_execution_result(self, agent_runner: object) -> None:
        from grampus.orchestration.runner import AgentRunner

        runner: AgentRunner = agent_runner  # type: ignore[assignment]
        runner._model_client.add_response("Hello, world!")

        result = await runner.run(_agent_def(), "Say hello.", session_id=make_session_id())
        assert result.output == "Hello, world!"
        assert result.status == AgentStatus.COMPLETED
        assert result.steps_taken >= 1

    async def test_react_loop_with_tool_calls(self, mock_tool_registry: object) -> None:
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor

        client = MockModelClient()
        client.add_response(
            text=None,
            tool_calls=[ToolCall(id="tc-1", name="echo", arguments={"text": "tool output"})],
        )
        client.add_response("Final answer after tool.")

        executor = ToolExecutor(mock_tool_registry, timeout_seconds=5.0)  # type: ignore[arg-type]
        runner = AgentRunner(
            client,
            executor,
            config=RunnerConfig(max_iterations=5, enable_memory=False),
        )

        result = await runner.run(
            _agent_def(name="react-agent"),
            "Use the echo tool.",
            session_id=make_session_id(),
        )
        assert result.tool_calls_made == 1
        assert result.output == "Final answer after tool."
        assert result.steps_taken == 2

    async def test_cost_tracked_per_llm_call(self, mock_tool_registry: object) -> None:
        from grampus.orchestration.cost_tracker import CostTracker
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor

        client = MockModelClient()
        client.add_response("Done.", cost_usd=0.005)

        cost_tracker = CostTracker(
            agent_id="cost-agent",
            session_id="cs1",
            budget_usd=None,
        )
        executor = ToolExecutor(mock_tool_registry, timeout_seconds=5.0)  # type: ignore[arg-type]
        runner = AgentRunner(
            client,
            executor,
            cost_tracker=cost_tracker,
            config=RunnerConfig(max_iterations=3, enable_memory=False),
        )

        await runner.run(_agent_def("cost-agent"), "Test cost.", session_id="cs1")
        summary = runner.cost_summary()
        assert summary is not None
        assert summary.total_cost_usd == pytest.approx(0.005)

    async def test_budget_exceeded_stops_execution(self, mock_tool_registry: object) -> None:
        from grampus.orchestration.cost_tracker import CostTracker
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor

        client = MockModelClient()
        # Use tool-call responses so the loop continues past the first step.
        for i in range(5):
            client.add_response(
                text=None,
                tool_calls=[ToolCall(id=f"tc-{i}", name="echo", arguments={"text": "x"})],
                cost_usd=0.002,
            )

        cost_tracker = CostTracker(
            agent_id="budget-agent",
            session_id="bs1",
            budget_usd=0.003,
        )
        executor = ToolExecutor(mock_tool_registry, timeout_seconds=5.0)  # type: ignore[arg-type]
        runner = AgentRunner(
            client,
            executor,
            cost_tracker=cost_tracker,
            config=RunnerConfig(max_iterations=5, enable_memory=False),
        )

        with pytest.raises(BudgetExceededError):
            await runner.run(
                _agent_def("budget-agent", budget=0.003),
                "Keep going.",
                session_id="bs1",
            )

    async def test_max_iterations_raises_error(self, mock_tool_registry: object) -> None:
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor

        client = MockModelClient()
        for _ in range(10):
            client.add_response(
                text=None,
                tool_calls=[ToolCall(id=f"tc-{_}", name="echo", arguments={"text": "x"})],
            )

        executor = ToolExecutor(mock_tool_registry, timeout_seconds=5.0)  # type: ignore[arg-type]
        runner = AgentRunner(
            client,
            executor,
            config=RunnerConfig(max_iterations=2, enable_memory=False),
        )

        with pytest.raises(OrchestrationError, match="MAX_ITERATIONS_EXCEEDED"):
            await runner.run(_agent_def(max_iter=2), "Infinite loop.", session_id=make_session_id())

    async def test_state_persisted_to_store(
        self,
        fake_state_store: FakeStateStore,
        mock_tool_registry: object,
    ) -> None:
        from grampus.core.types import AgentState
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor

        client = MockModelClient()
        client.add_response("Stored.")
        executor = ToolExecutor(mock_tool_registry, timeout_seconds=5.0)  # type: ignore[arg-type]
        runner = AgentRunner(
            client,
            executor,
            state_store=fake_state_store,
            config=RunnerConfig(max_iterations=3, enable_memory=False),
        )

        session_id = make_session_id()
        await runner.run(_agent_def("state-agent"), "Persist me.", session_id=session_id)

        stored, _ = await fake_state_store.get(
            "runner", f"agent:state-agent:{session_id}", AgentState
        )
        assert stored is not None
        assert stored.agent_id == "state-agent"

    async def test_memory_recalled_before_llm_call(
        self,
        fake_state_store: FakeStateStore,
        mock_tool_registry: object,
    ) -> None:
        from grampus.memory.consolidation import ConsolidationPipeline
        from grampus.memory.episodic import EpisodicMemory
        from grampus.memory.manager import MemoryManager
        from grampus.memory.procedural import ProceduralMemory
        from grampus.memory.retriever import EpisodicRetriever
        from grampus.memory.semantic import SemanticMemory
        from grampus.memory.semantic_retriever import SemanticRetriever
        from grampus.memory.summarizer import Summarizer
        from grampus.memory.token_counter import TokenCounter
        from grampus.memory.working import WorkingMemory
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor
        from tests.integration.conftest import FakeEmbeddingService

        store = fake_state_store
        emb = FakeEmbeddingService()
        client = MockModelClient()
        client.add_response("Answer based on memory.")

        episodic = EpisodicMemory(store, emb, agent_id="mem-runner")
        semantic = SemanticMemory(store, agent_id="mem-runner")
        procedural = ProceduralMemory(store, agent_id="mem-runner")
        working = WorkingMemory(
            store, TokenCounter(), Summarizer(client), agent_id="mem-runner", session_id="ms1"
        )
        ep_retriever = EpisodicRetriever(episodic, emb)
        sem_retriever = SemanticRetriever(semantic, emb)
        consolidation = ConsolidationPipeline(episodic, semantic, client, agent_id="mem-runner")
        mm = MemoryManager(
            working,
            episodic,
            semantic,
            procedural,
            ep_retriever,
            sem_retriever,
            consolidation,
            agent_id="mem-runner",
        )

        await episodic.store("User prefers dark mode.", session_id="ms1")

        executor = ToolExecutor(mock_tool_registry, timeout_seconds=5.0)  # type: ignore[arg-type]
        runner = AgentRunner(
            client,
            executor,
            memory_manager=mm,
            config=RunnerConfig(max_iterations=3, enable_memory=True),
        )

        await runner.run(
            _agent_def("mem-runner"),
            "What do I prefer?",
            session_id="ms1",
        )

        first_call_messages = client.calls[0]
        all_content = " ".join(m.content for m in first_call_messages if m.content)
        assert "dark mode" in all_content
