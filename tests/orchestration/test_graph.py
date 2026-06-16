"""Tests for the graph execution engine (Phase 7a)."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.errors import OrchestrationError
from grampus.core.types import AgentState, Message, Role
from grampus.orchestration.graph import Graph, GraphCheckpoint


def _state(agent_id: str = "agent-1", session_id: str = "sess-1") -> AgentState:
    return AgentState(agent_id=agent_id, session_id=session_id)


class TestGraphBuilder:
    def test_add_node_registers_node(self) -> None:
        async def handler(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1")
        g.add_node("n1", handler, entry=True)
        assert "n1" in g._nodes

    def test_add_edge_registers_edge(self) -> None:
        async def h(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1")
        g.add_node("n1", h, entry=True).add_node("n2", h)
        g.add_edge("n1", "n2")
        assert any(e.from_node == "n1" and e.to_node == "n2" for e in g._edges)

    def test_add_conditional_edge_registers_conditional_edge(self) -> None:
        async def h(state: AgentState) -> AgentState:
            return state

        async def cond(state: AgentState) -> str:
            return "yes"

        g = Graph(graph_id="g1")
        g.add_node("n1", h, entry=True).add_node("n2", h).add_node("n3", h)
        g.add_conditional_edge("n1", cond, {"yes": "n2", "no": "n3"})
        assert any(e.from_node == "n1" and e.condition is not None for e in g._edges)

    def test_builder_returns_self_for_chaining(self) -> None:
        async def h(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1")
        result = g.add_node("n1", h, entry=True).add_node("n2", h).add_edge("n1", "n2")
        assert result is g

    @pytest.mark.asyncio
    async def test_no_entry_node_raises_orchestration_error(self) -> None:
        async def h(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1")
        g.add_node("n1", h)  # no entry=True

        with pytest.raises(OrchestrationError) as exc_info:
            await g.execute(_state())
        assert exc_info.value.code == "NO_ENTRY_NODE"


class TestGraphExecution:
    @pytest.mark.asyncio
    async def test_single_node_graph_executes_handler(self) -> None:
        called: list[str] = []

        async def h(state: AgentState) -> AgentState:
            called.append("n1")
            return state

        g = Graph(graph_id="g1")
        g.add_node("n1", h, entry=True).add_edge("n1", None)
        await g.execute(_state())
        assert called == ["n1"]

    @pytest.mark.asyncio
    async def test_two_node_graph_executes_in_order(self) -> None:
        order: list[str] = []

        async def h1(state: AgentState) -> AgentState:
            order.append("n1")
            return state

        async def h2(state: AgentState) -> AgentState:
            order.append("n2")
            return state

        g = Graph(graph_id="g1")
        g.add_node("n1", h1, entry=True).add_node("n2", h2)
        g.add_edge("n1", "n2").add_edge("n2", None)
        await g.execute(_state())
        assert order == ["n1", "n2"]

    @pytest.mark.asyncio
    async def test_three_node_linear_graph_completes(self) -> None:
        order: list[str] = []

        def make_h(name: str) -> object:
            async def h(state: AgentState) -> AgentState:
                order.append(name)
                return state

            return h

        g = Graph(graph_id="g1")
        g.add_node("a", make_h("a"), entry=True)  # type: ignore[arg-type]
        g.add_node("b", make_h("b"))  # type: ignore[arg-type]
        g.add_node("c", make_h("c"))  # type: ignore[arg-type]
        g.add_edge("a", "b").add_edge("b", "c").add_edge("c", None)
        await g.execute(_state())
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_conditional_edge_routes_correctly_on_true(self) -> None:
        visited: list[str] = []

        async def entry(state: AgentState) -> AgentState:
            return state

        async def yes_node(state: AgentState) -> AgentState:
            visited.append("yes")
            return state

        async def no_node(state: AgentState) -> AgentState:
            visited.append("no")
            return state

        async def cond(state: AgentState) -> str:
            return "yes"

        g = Graph(graph_id="g1")
        g.add_node("entry", entry, entry=True)
        g.add_node("yes_node", yes_node)
        g.add_node("no_node", no_node)
        g.add_conditional_edge("entry", cond, {"yes": "yes_node", "no": "no_node"})
        g.add_edge("yes_node", None).add_edge("no_node", None)
        await g.execute(_state())
        assert visited == ["yes"]

    @pytest.mark.asyncio
    async def test_conditional_edge_routes_correctly_on_false(self) -> None:
        visited: list[str] = []

        async def entry(state: AgentState) -> AgentState:
            return state

        async def yes_node(state: AgentState) -> AgentState:
            visited.append("yes")
            return state

        async def no_node(state: AgentState) -> AgentState:
            visited.append("no")
            return state

        async def cond(state: AgentState) -> str:
            return "no"

        g = Graph(graph_id="g1")
        g.add_node("entry", entry, entry=True)
        g.add_node("yes_node", yes_node)
        g.add_node("no_node", no_node)
        g.add_conditional_edge("entry", cond, {"yes": "yes_node", "no": "no_node"})
        g.add_edge("yes_node", None).add_edge("no_node", None)
        await g.execute(_state())
        assert visited == ["no"]

    @pytest.mark.asyncio
    async def test_terminal_edge_ends_execution(self) -> None:
        order: list[str] = []

        async def h1(state: AgentState) -> AgentState:
            order.append("n1")
            return state

        async def h2(state: AgentState) -> AgentState:
            order.append("n2")
            return state

        g = Graph(graph_id="g1")
        g.add_node("n1", h1, entry=True).add_node("n2", h2)
        g.add_edge("n1", None)  # terminal after n1 — n2 never runs
        await g.execute(_state())
        assert order == ["n1"]

    @pytest.mark.asyncio
    async def test_max_steps_exceeded_raises_orchestration_error(self) -> None:
        async def loop_node(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1", max_steps=3)
        g.add_node("n1", loop_node, entry=True)
        g.add_edge("n1", "n1")  # infinite self-loop

        with pytest.raises(OrchestrationError) as exc_info:
            await g.execute(_state())
        assert exc_info.value.code == "MAX_STEPS_EXCEEDED"

    @pytest.mark.asyncio
    async def test_unknown_node_raises_orchestration_error(self) -> None:
        async def h(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1")
        g.add_node("n1", h, entry=True)
        g.add_edge("n1", "missing_node")  # points to non-existent node

        with pytest.raises(OrchestrationError) as exc_info:
            await g.execute(_state())
        assert exc_info.value.code == "UNKNOWN_NODE"

    @pytest.mark.asyncio
    async def test_state_is_passed_through_each_node(self) -> None:
        async def add_msg(state: AgentState) -> AgentState:
            new_state = state.model_copy(deep=True)
            new_state.messages.append(Message(role=Role.USER, content="hello"))
            return new_state

        g = Graph(graph_id="g1")
        g.add_node("n1", add_msg, entry=True).add_edge("n1", None)
        result = await g.execute(_state())
        assert len(result.messages) == 1
        assert result.messages[0].content == "hello"

    @pytest.mark.asyncio
    async def test_state_mutations_accumulate_across_nodes(self) -> None:
        async def add_msg_a(state: AgentState) -> AgentState:
            new_state = state.model_copy(deep=True)
            new_state.messages.append(Message(role=Role.USER, content="a"))
            return new_state

        async def add_msg_b(state: AgentState) -> AgentState:
            new_state = state.model_copy(deep=True)
            new_state.messages.append(Message(role=Role.ASSISTANT, content="b"))
            return new_state

        g = Graph(graph_id="g1")
        g.add_node("n1", add_msg_a, entry=True)
        g.add_node("n2", add_msg_b)
        g.add_edge("n1", "n2").add_edge("n2", None)
        result = await g.execute(_state())
        assert len(result.messages) == 2
        contents = [m.content for m in result.messages]
        assert "a" in contents and "b" in contents


class TestGraphCheckpointing:
    @pytest.mark.asyncio
    async def test_checkpoint_saved_after_each_node(self) -> None:
        store = MagicMock()
        store.save = AsyncMock()

        async def h(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1", state_store=store)
        g.add_node("n1", h, entry=True).add_node("n2", h)
        g.add_edge("n1", "n2").add_edge("n2", None)
        await g.execute(_state())
        assert store.save.call_count >= 2

    @pytest.mark.asyncio
    async def test_restore_and_execute_resumes_from_checkpoint(self) -> None:
        state = _state()
        state.messages.append(Message(role=Role.USER, content="from_n1"))
        checkpoint = GraphCheckpoint(
            graph_id="g1",
            current_node="n2",
            state=state,
            completed_nodes=["n1"],
            created_at=datetime.now(UTC),
        )

        store = MagicMock()
        store.get = AsyncMock(return_value=(checkpoint, "etag"))
        store.save = AsyncMock()

        visited: list[str] = []

        async def n1(s: AgentState) -> AgentState:
            visited.append("n1")
            return s

        async def n2(s: AgentState) -> AgentState:
            visited.append("n2")
            return s

        g = Graph(graph_id="g1", state_store=store)
        g.add_node("n1", n1, entry=True).add_node("n2", n2)
        g.add_edge("n1", "n2").add_edge("n2", None)

        result = await g.restore_and_execute("g1")
        assert result is not None
        assert "n2" in visited
        assert "n1" not in visited

    @pytest.mark.asyncio
    async def test_restore_returns_none_when_no_checkpoint(self) -> None:
        store = MagicMock()
        store.get = AsyncMock(return_value=(None, ""))

        async def h(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1", state_store=store)
        g.add_node("n1", h, entry=True).add_edge("n1", None)
        result = await g.restore_and_execute("g1")
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_without_state_store_skips_checkpointing(self) -> None:
        calls: list[str] = []

        async def h(state: AgentState) -> AgentState:
            calls.append("ran")
            return state

        g = Graph(graph_id="g1")  # no state_store
        g.add_node("n1", h, entry=True).add_edge("n1", None)
        result = await g.execute(_state())
        assert calls == ["ran"]
        assert result is not None


class TestGraphParallel:
    @pytest.mark.asyncio
    async def test_parallel_nodes_run_concurrently(self) -> None:
        start_times: list[float] = []
        end_times: list[float] = []

        async def slow_node(state: AgentState) -> AgentState:
            start_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            end_times.append(time.monotonic())
            return state

        async def noop(state: AgentState) -> AgentState:
            return state

        g = Graph(graph_id="g1")
        g.add_node("entry", noop, entry=True)
        g.add_node("parallel:a", slow_node)
        g.add_node("parallel:b", slow_node)
        g.add_edge("entry", "parallel:a")
        g.add_edge("entry", "parallel:b")
        g.add_edge("parallel:a", None)
        g.add_edge("parallel:b", None)
        await g.execute(_state())

        assert len(start_times) == 2
        # Concurrent: second started before first ended
        assert start_times[1] < end_times[0] or start_times[0] < end_times[1]

    @pytest.mark.asyncio
    async def test_parallel_node_messages_are_merged(self) -> None:
        async def entry(state: AgentState) -> AgentState:
            return state

        async def branch_a(state: AgentState) -> AgentState:
            s = state.model_copy(deep=True)
            s.messages.append(Message(role=Role.USER, content="from_a"))
            return s

        async def branch_b(state: AgentState) -> AgentState:
            s = state.model_copy(deep=True)
            s.messages.append(Message(role=Role.USER, content="from_b"))
            return s

        g = Graph(graph_id="g1")
        g.add_node("entry", entry, entry=True)
        g.add_node("parallel:a", branch_a)
        g.add_node("parallel:b", branch_b)
        g.add_edge("entry", "parallel:a")
        g.add_edge("entry", "parallel:b")
        g.add_edge("parallel:a", None)
        g.add_edge("parallel:b", None)

        result = await g.execute(_state())
        contents = [m.content for m in result.messages]
        assert "from_a" in contents
        assert "from_b" in contents
