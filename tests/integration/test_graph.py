"""Integration tests for the Graph execution engine with FakeStateStore."""

from __future__ import annotations

import pytest

from nexus.core.errors import OrchestrationError
from nexus.core.types import AgentState, AgentStatus
from tests.integration.conftest import FakeStateStore, make_session_id


def _initial_state(agent_id: str = "graph-agent") -> AgentState:
    return AgentState(agent_id=agent_id, session_id=make_session_id())


@pytest.mark.integration
class TestGraphIntegration:
    async def test_linear_graph_completes_with_fake_store(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.orchestration.graph import Graph

        log: list[str] = []

        async def node_a(state: AgentState) -> AgentState:
            log.append("A")
            return state.model_copy(update={"metadata": {**state.metadata, "a": True}})

        async def node_b(state: AgentState) -> AgentState:
            log.append("B")
            return state.model_copy(update={"metadata": {**state.metadata, "b": True}})

        g = Graph(graph_id="test-linear", state_store=fake_state_store)
        g.add_node("A", node_a, entry=True).add_node("B", node_b)
        g.add_edge("A", "B").add_edge("B", None)

        final = await g.execute(_initial_state())
        assert log == ["A", "B"]
        assert final.metadata.get("a") is True
        assert final.metadata.get("b") is True

    async def test_checkpoint_saved_after_each_node(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.orchestration.graph import Graph, GraphCheckpoint

        async def node_a(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="chk-graph", state_store=fake_state_store)
        g.add_node("A", node_a, entry=True).add_edge("A", None)
        await g.execute(_initial_state())

        checkpoint, _ = await fake_state_store.get(
            "orchestration", "graph:chk-graph:checkpoint", GraphCheckpoint
        )
        assert checkpoint is not None
        assert "A" in checkpoint.completed_nodes

    async def test_restore_and_execute_resumes_from_checkpoint(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.orchestration.graph import Graph

        executed: list[str] = []
        fail_b = [True]

        async def node_a(state: AgentState) -> AgentState:
            executed.append("A")
            return state

        async def node_b(state: AgentState) -> AgentState:
            if fail_b[0]:
                fail_b[0] = False
                raise RuntimeError("simulated failure")
            executed.append("B")
            return state.model_copy(update={"status": AgentStatus.COMPLETED})

        async def node_c(state: AgentState) -> AgentState:
            executed.append("C")
            return state

        g1 = Graph(graph_id="restore-graph", state_store=fake_state_store)
        g1.add_node("A", node_a, entry=True).add_node("B", node_b).add_node("C", node_c)
        g1.add_edge("A", "B").add_edge("B", "C").add_edge("C", None)

        with pytest.raises(RuntimeError):
            await g1.execute(_initial_state())

        # A ran on first attempt, checkpoint saved for A
        g2 = Graph(graph_id="restore-graph", state_store=fake_state_store)
        g2.add_node("A", node_a, entry=True).add_node("B", node_b).add_node("C", node_c)
        g2.add_edge("A", "B").add_edge("B", "C").add_edge("C", None)

        final = await g2.restore_and_execute("restore-graph")
        assert final is not None
        assert "B" in executed
        assert "C" in executed

    async def test_conditional_routing_reaches_correct_terminal(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.orchestration.graph import Graph

        reached: list[str] = []

        async def router(state: AgentState) -> AgentState:
            return state.model_copy(update={"metadata": {**state.metadata, "route": "path_b"}})

        async def condition(state: AgentState) -> str:
            return state.metadata.get("route", "path_a")

        async def path_a(state: AgentState) -> AgentState:
            reached.append("path_a")
            return state

        async def path_b(state: AgentState) -> AgentState:
            reached.append("path_b")
            return state

        g = Graph(graph_id="cond-graph", state_store=fake_state_store)
        g.add_node("router", router, entry=True)
        g.add_node("path_a", path_a)
        g.add_node("path_b", path_b)
        g.add_conditional_edge("router", condition, {"path_a": "path_a", "path_b": "path_b"})
        g.add_edge("path_a", None)
        g.add_edge("path_b", None)

        await g.execute(_initial_state())
        assert reached == ["path_b"]

    async def test_parallel_nodes_both_execute(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.orchestration.graph import Graph

        executed: set[str] = set()

        async def entry(state: AgentState) -> AgentState:
            return state

        async def branch_x(state: AgentState) -> AgentState:
            executed.add("X")
            return state

        async def branch_y(state: AgentState) -> AgentState:
            executed.add("Y")
            return state

        g = Graph(graph_id="par-graph", state_store=fake_state_store)
        g.add_node("entry", entry, entry=True)
        g.add_node("branch_x", branch_x)
        g.add_node("branch_y", branch_y)
        g.add_edge("entry", "branch_x")
        g.add_edge("entry", "branch_y")

        await g.execute(_initial_state())
        assert "X" in executed
        assert "Y" in executed

    async def test_max_steps_exceeded_raises_error(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.orchestration.graph import Graph

        async def loop_node(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="loop-graph", state_store=fake_state_store, max_steps=3)
        g.add_node("loop", loop_node, entry=True)
        g.add_edge("loop", "loop")

        with pytest.raises(OrchestrationError, match="MAX_STEPS_EXCEEDED"):
            await g.execute(_initial_state())

    async def test_restore_returns_none_with_no_checkpoint(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.orchestration.graph import Graph

        async def node(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="no-chk-graph", state_store=fake_state_store)
        g.add_node("A", node, entry=True).add_edge("A", None)

        result = await g.restore_and_execute("no-chk-graph")
        assert result is None
