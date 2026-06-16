"""E2E scenario: graph checkpoint and restore after simulated failure."""

from __future__ import annotations

import pytest

from grampus.core.types import AgentState, AgentStatus
from tests.integration.conftest import FakeStateStore, make_session_id


def _initial_state(agent_id: str = "ckpt-agent") -> AgentState:
    return AgentState(agent_id=agent_id, session_id=make_session_id())


@pytest.mark.integration
class TestCheckpointRestoreE2E:
    async def test_graph_resumes_from_checkpoint_after_failure(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from grampus.orchestration.graph import Graph

        node_a_calls = [0]
        node_b_calls = [0]
        node_c_calls = [0]
        fail_once = [True]

        async def node_a(state: AgentState) -> AgentState:
            node_a_calls[0] += 1
            return state.model_copy(update={"metadata": {**state.metadata, "a_done": True}})

        async def node_b(state: AgentState) -> AgentState:
            node_b_calls[0] += 1
            if fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("Simulated node B crash")
            return state.model_copy(update={"metadata": {**state.metadata, "b_done": True}})

        async def node_c(state: AgentState) -> AgentState:
            node_c_calls[0] += 1
            return state.model_copy(update={"metadata": {**state.metadata, "c_done": True}})

        graph_id = "ckpt-graph"

        g1 = Graph(graph_id=graph_id, state_store=fake_state_store)
        g1.add_node("A", node_a, entry=True).add_node("B", node_b).add_node("C", node_c)
        g1.add_edge("A", "B").add_edge("B", "C").add_edge("C", None)

        with pytest.raises(RuntimeError, match="Simulated node B crash"):
            await g1.execute(_initial_state())

        assert node_a_calls[0] >= 1

        g2 = Graph(graph_id=graph_id, state_store=fake_state_store)
        g2.add_node("A", node_a, entry=True).add_node("B", node_b).add_node("C", node_c)
        g2.add_edge("A", "B").add_edge("B", "C").add_edge("C", None)

        final_state = await g2.restore_and_execute(graph_id)
        assert final_state is not None
        assert node_b_calls[0] >= 1
        assert node_c_calls[0] >= 1

    async def test_restore_returns_none_with_no_checkpoint(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from grampus.orchestration.graph import Graph

        async def node(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="fresh-graph", state_store=fake_state_store)
        g.add_node("A", node, entry=True).add_edge("A", None)

        result = await g.restore_and_execute("fresh-graph")
        assert result is None

    async def test_human_node_pause_and_resume(self, fake_state_store: FakeStateStore) -> None:
        from grampus.core.types import AgentDefinition, ToolCall
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor
        from grampus.tools.registry import ToolRegistry
        from tests.integration.conftest import MockModelClient

        agent_id = "human-node-agent"
        session_id = make_session_id()

        client = MockModelClient()
        client.add_response(
            text=None,
            tool_calls=[ToolCall(id="tc-human", name="human_input", arguments={})],
        )
        client.add_response("Final answer after human input.")

        registry = ToolRegistry()
        executor = ToolExecutor(registry, timeout_seconds=5.0)

        runner = AgentRunner(
            client,
            executor,
            state_store=fake_state_store,
            config=RunnerConfig(max_iterations=5, enable_memory=False),
        )

        agent_def = AgentDefinition(
            name=agent_id,
            model="mock-model",
            system_prompt="Ask for human input when needed.",
            tools=["human_input"],
            max_iterations=5,
            temperature=0.0,
            memory_enabled=False,
            cost_budget_usd=None,
        )

        first_result = await runner.run(
            agent_def, "Do something requiring human input.", session_id=session_id
        )
        assert first_result.status in (
            AgentStatus.WAITING_FOR_HUMAN,
            AgentStatus.COMPLETED,
        )

    async def test_checkpoint_state_is_fully_restored(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from grampus.orchestration.graph import Graph, GraphCheckpoint

        async def node_a(state: AgentState) -> AgentState:
            return state.model_copy(update={"metadata": {**state.metadata, "from_a": "yes"}})

        graph_id = "restore-state-graph"
        g = Graph(graph_id=graph_id, state_store=fake_state_store)
        g.add_node("A", node_a, entry=True).add_edge("A", None)
        await g.execute(_initial_state())

        checkpoint, _ = await fake_state_store.get(
            "orchestration", f"graph:{graph_id}:checkpoint", GraphCheckpoint
        )
        assert checkpoint is not None
        assert "A" in checkpoint.completed_nodes
        assert checkpoint.state.metadata.get("from_a") == "yes"
