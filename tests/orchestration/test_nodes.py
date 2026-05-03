"""Tests for pre-built graph node handler factories (Phase 7a)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.models.base import ModelResponse
from nexus.core.types import AgentState, AgentStatus, Message, Role, TokenUsage, ToolCall
from nexus.orchestration.graph import Graph
from nexus.orchestration.nodes import human_node, llm_node, subgraph_node, tool_node


def _state() -> AgentState:
    return AgentState(agent_id="agent-1", session_id="sess-1")


def _token_usage(model: str = "claude-3") -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model=model
    )


class TestLLMNode:
    @pytest.mark.asyncio
    async def test_llm_node_calls_model_client(self) -> None:
        client = MagicMock()
        response = ModelResponse(
            content="Hello",
            tool_calls=[],
            token_usage=_token_usage(),
            model="claude-3",
            stop_reason="end_turn",
        )
        client.complete = AsyncMock(return_value=response)

        handler = llm_node(client, model="claude-3")
        await handler(_state())
        client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_node_appends_assistant_message_to_state(self) -> None:
        client = MagicMock()
        response = ModelResponse(
            content="Hi there",
            tool_calls=[],
            token_usage=_token_usage(),
            model="claude-3",
            stop_reason="end_turn",
        )
        client.complete = AsyncMock(return_value=response)

        handler = llm_node(client, model="claude-3")
        result = await handler(_state())

        assert len(result.messages) == 1
        assert result.messages[0].role == Role.ASSISTANT
        assert result.messages[0].content == "Hi there"

    @pytest.mark.asyncio
    async def test_llm_node_accumulates_token_usage(self) -> None:
        client = MagicMock()
        usage = _token_usage()
        response = ModelResponse(
            content="Reply",
            tool_calls=[],
            token_usage=usage,
            model="claude-3",
            stop_reason="end_turn",
        )
        client.complete = AsyncMock(return_value=response)

        handler = llm_node(client, model="claude-3")
        result = await handler(_state())

        assert result.total_token_usage is not None
        assert result.total_token_usage.input_tokens == usage.input_tokens
        assert result.total_token_usage.output_tokens == usage.output_tokens

    @pytest.mark.asyncio
    async def test_llm_node_sets_status_running(self) -> None:
        client = MagicMock()
        client.complete = AsyncMock(
            return_value=ModelResponse(
                content="ok",
                tool_calls=[],
                token_usage=_token_usage(),
                model="claude-3",
                stop_reason="end_turn",
            )
        )

        handler = llm_node(client, model="claude-3")
        result = await handler(_state())
        assert result.status == AgentStatus.RUNNING


class TestToolNode:
    @pytest.mark.asyncio
    async def test_tool_node_executes_pending_tool_calls(self) -> None:
        from nexus.core.types import ToolResult

        tool_call = ToolCall(id="tc1", name="search", arguments={"q": "hello"})
        state = _state()
        state.messages.append(Message(role=Role.ASSISTANT, content=None, tool_calls=[tool_call]))

        executor = MagicMock()
        result_obj = ToolResult(tool_call_id="tc1", output="found it", error=None)
        executor.execute = AsyncMock(return_value=result_obj)

        handler = tool_node(executor)
        await handler(state)
        executor.execute.assert_called_once_with(tool_call)

    @pytest.mark.asyncio
    async def test_tool_node_appends_tool_results(self) -> None:
        from nexus.core.types import ToolResult

        tool_call = ToolCall(id="tc1", name="search", arguments={"q": "hello"})
        state = _state()
        state.messages.append(Message(role=Role.ASSISTANT, content=None, tool_calls=[tool_call]))

        executor = MagicMock()
        executor.execute = AsyncMock(return_value=ToolResult(tool_call_id="tc1", output="found it"))

        handler = tool_node(executor)
        result_state = await handler(state)

        tool_msgs = [m for m in result_state.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_results[0].output == "found it"

    @pytest.mark.asyncio
    async def test_tool_node_passes_through_when_no_tool_calls(self) -> None:
        state = _state()
        state.messages.append(Message(role=Role.ASSISTANT, content="Just a response"))

        executor = MagicMock()
        executor.execute = AsyncMock()

        handler = tool_node(executor)
        result_state = await handler(state)
        executor.execute.assert_not_called()
        assert result_state.messages == state.messages

    @pytest.mark.asyncio
    async def test_tool_node_sets_status_running(self) -> None:
        from nexus.core.types import ToolResult

        tool_call = ToolCall(id="tc1", name="search", arguments={})
        state = _state()
        state.messages.append(Message(role=Role.ASSISTANT, content=None, tool_calls=[tool_call]))

        executor = MagicMock()
        executor.execute = AsyncMock(return_value=ToolResult(tool_call_id="tc1", output="ok"))

        handler = tool_node(executor)
        result_state = await handler(state)
        assert result_state.status == AgentStatus.RUNNING


class TestHumanNode:
    @pytest.mark.asyncio
    async def test_human_node_sets_status_waiting_for_human(self) -> None:
        handler = human_node()
        result = await handler(_state())
        assert result.status == AgentStatus.WAITING_FOR_HUMAN

    @pytest.mark.asyncio
    async def test_human_node_appends_system_message(self) -> None:
        handler = human_node(prompt="Please review this.")
        result = await handler(_state())
        system_msgs = [m for m in result.messages if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1
        assert "Please review this." in (system_msgs[0].content or "")

    @pytest.mark.asyncio
    async def test_human_node_returns_immediately(self) -> None:
        handler = human_node()
        start = time.monotonic()
        await handler(_state())
        elapsed = time.monotonic() - start
        assert elapsed < 0.05


class TestSubgraphNode:
    @pytest.mark.asyncio
    async def test_subgraph_node_executes_nested_graph(self) -> None:
        visited: list[str] = []

        async def inner_handler(state: AgentState) -> AgentState:
            visited.append("inner")
            return state

        subgraph = Graph(graph_id="sub1")
        subgraph.add_node("inner", inner_handler, entry=True).add_edge("inner", None)

        handler = subgraph_node(subgraph)
        await handler(_state())
        assert "inner" in visited

    @pytest.mark.asyncio
    async def test_subgraph_node_returns_final_state(self) -> None:
        async def inner_handler(state: AgentState) -> AgentState:
            s = state.model_copy(deep=True)
            s.metadata["from_subgraph"] = True
            return s

        subgraph = Graph(graph_id="sub1")
        subgraph.add_node("inner", inner_handler, entry=True).add_edge("inner", None)

        handler = subgraph_node(subgraph)
        result = await handler(_state())
        assert result.metadata.get("from_subgraph") is True
