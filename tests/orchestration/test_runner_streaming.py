"""Tests for AgentRunner.stream() — streaming ReAct loop with structured events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.models.base import ModelResponse
from grampus.core.types import (
    AgentDefinition,
    StreamChunk,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from grampus.memory.manager import MemoryRecallResult
from grampus.orchestration.runner import AgentRunner, RunnerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_def(name: str = "test-agent") -> AgentDefinition:
    return AgentDefinition(name=name, model="test-model")


def _token_usage(input_t: int = 10, output_t: int = 5, cost: float = 0.001) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_t,
        output_tokens=output_t,
        total_tokens=input_t + output_t,
        cost_usd=cost,
        model="test-model",
    )


def _model_response(
    content: str | None = "Hello!",
    tool_calls: list[ToolCall] | None = None,
    usage: TokenUsage | None = None,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls or [],
        token_usage=usage or _token_usage(),
        model="test-model",
        stop_reason="end_turn" if not tool_calls else "tool_use",
    )


def _text_chunks(text: str, usage: TokenUsage | None = None) -> list[StreamChunk]:
    chunks: list[StreamChunk] = []
    words = text.split()
    for i, word in enumerate(words):
        delta = word + (" " if i < len(words) - 1 else "")
        chunks.append(StreamChunk(delta=delta, model="test-model"))
    chunks.append(
        StreamChunk(
            delta="",
            is_final=True,
            finish_reason="end_turn",
            token_usage=usage or _token_usage(),
            model="test-model",
        )
    )
    return chunks


def _tool_use_chunks(usage: TokenUsage | None = None) -> list[StreamChunk]:
    return [
        StreamChunk(
            delta="",
            is_final=True,
            finish_reason="tool_use",
            token_usage=usage or _token_usage(),
            model="test-model",
        )
    ]


def _make_text_client(text: str = "Hello!") -> MagicMock:
    client = MagicMock()
    chunks = _text_chunks(text)

    async def _stream(**kwargs: Any) -> AsyncIterator[StreamChunk]:
        for chunk in chunks:
            yield chunk

    client.stream = _stream
    client.complete = AsyncMock(return_value=_model_response(content=text))
    return client


def _make_tool_call_client(
    tool_call: ToolCall,
    final_text: str = "Done!",
    tool_usage: TokenUsage | None = None,
    text_usage: TokenUsage | None = None,
) -> MagicMock:
    client = MagicMock()
    tool_chunks = _tool_use_chunks(tool_usage)
    text_chunks = _text_chunks(final_text, text_usage)
    call_count: dict[str, int] = {"stream": 0, "complete": 0}

    async def _stream(**kwargs: Any) -> AsyncIterator[StreamChunk]:
        chunks = tool_chunks if call_count["stream"] == 0 else text_chunks
        call_count["stream"] += 1
        for chunk in chunks:
            yield chunk

    tc_response = _model_response(content=None, tool_calls=[tool_call])
    text_response = _model_response(content=final_text)

    async def _complete(**kwargs: Any) -> ModelResponse:
        responses = [tc_response, text_response]
        idx = min(call_count["complete"], len(responses) - 1)
        call_count["complete"] += 1
        return responses[idx]

    client.stream = _stream
    client.complete = _complete
    return client


async def _collect(
    runner: AgentRunner,
    text: str = "Hello",
    *,
    name: str = "test-agent",
    session_id: str = "s1",
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for event in runner.stream(_agent_def(name), text, session_id=session_id):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_executor() -> AsyncMock:
    executor = AsyncMock()
    executor.execute = AsyncMock(
        return_value=ToolResult(tool_call_id="tc-1", output="tool output", duration_ms=5)
    )
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
    return tracker


# ---------------------------------------------------------------------------
# TestStreamAgentStartAndEnd
# ---------------------------------------------------------------------------


class TestStreamAgentStartAndEnd:
    async def test_stream_yields_agent_start_and_end(self, tool_executor: AsyncMock) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor)
        events = await _collect(runner)
        types = [e.event_type for e in events]
        assert StreamEventType.AGENT_START in types
        assert StreamEventType.AGENT_END in types

    async def test_stream_agent_start_is_first_event(self, tool_executor: AsyncMock) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor)
        events = await _collect(runner)
        assert events[0].event_type == StreamEventType.AGENT_START

    async def test_stream_agent_end_is_last_event(self, tool_executor: AsyncMock) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor)
        events = await _collect(runner)
        assert events[-1].event_type == StreamEventType.AGENT_END

    async def test_stream_agent_start_carries_agent_name(self, tool_executor: AsyncMock) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor)
        events = await _collect(runner, name="my-agent")
        start = events[0]
        assert start.message == "my-agent"

    async def test_stream_agent_end_has_final_chunk(self, tool_executor: AsyncMock) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor)
        events = await _collect(runner)
        end = events[-1]
        assert end.chunk is not None
        assert end.chunk.is_final is True

    async def test_stream_yields_iteration_start(self, tool_executor: AsyncMock) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor)
        events = await _collect(runner)
        types = [e.event_type for e in events]
        assert StreamEventType.ITERATION_START in types

    async def test_stream_iteration_start_has_iteration_number(
        self, tool_executor: AsyncMock
    ) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor)
        events = await _collect(runner)
        iter_event = next(e for e in events if e.event_type == StreamEventType.ITERATION_START)
        assert iter_event.iteration == 1


# ---------------------------------------------------------------------------
# TestStreamTokenEvents
# ---------------------------------------------------------------------------


class TestStreamTokenEvents:
    async def test_stream_token_events_contain_delta(self, tool_executor: AsyncMock) -> None:
        runner = AgentRunner(_make_text_client("Hello world"), tool_executor)
        events = await _collect(runner, "Hi")
        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        assert len(token_events) > 0
        assert all(e.chunk is not None for e in token_events)

    async def test_stream_token_events_reconstruct_full_text(
        self, tool_executor: AsyncMock
    ) -> None:
        runner = AgentRunner(_make_text_client("Hello world"), tool_executor)
        events = await _collect(runner, "Hi")
        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        combined = "".join(e.chunk.delta for e in token_events if e.chunk)
        assert "Hello" in combined
        assert "world" in combined

    async def test_stream_empty_content_yields_no_token_events(
        self, tool_executor: AsyncMock
    ) -> None:
        tc = ToolCall(id="tc-1", name="my_tool", arguments={})
        client = _make_tool_call_client(tc, final_text="Done!")
        runner = AgentRunner(client, tool_executor)
        events = await _collect(runner, "Do")
        # First iteration has no text (tool_use), second has "Done!"
        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        combined = "".join(e.chunk.delta for e in token_events if e.chunk)
        assert "Done" in combined


# ---------------------------------------------------------------------------
# TestStreamToolCallEvents
# ---------------------------------------------------------------------------


class TestStreamToolCallEvents:
    async def test_stream_tool_call_events(self, tool_executor: AsyncMock) -> None:
        tc = ToolCall(id="tc-1", name="my_tool", arguments={})
        runner = AgentRunner(_make_tool_call_client(tc), tool_executor)
        events = await _collect(runner, "Do something")
        types = [e.event_type for e in events]
        assert StreamEventType.TOOL_CALL_START in types
        assert StreamEventType.TOOL_CALL_END in types

    async def test_stream_tool_call_start_has_tool_call(self, tool_executor: AsyncMock) -> None:
        tc = ToolCall(id="tc-1", name="search_web", arguments={"q": "test"})
        runner = AgentRunner(_make_tool_call_client(tc), tool_executor)
        events = await _collect(runner, "Search")
        start_events = [e for e in events if e.event_type == StreamEventType.TOOL_CALL_START]
        assert len(start_events) == 1
        assert start_events[0].tool_call is not None
        assert start_events[0].tool_call.name == "search_web"

    async def test_stream_tool_call_end_has_tool_result(self, tool_executor: AsyncMock) -> None:
        tc = ToolCall(id="tc-1", name="my_tool", arguments={})
        runner = AgentRunner(_make_tool_call_client(tc), tool_executor)
        events = await _collect(runner, "Do")
        end_events = [e for e in events if e.event_type == StreamEventType.TOOL_CALL_END]
        assert len(end_events) == 1
        assert end_events[0].tool_result is not None

    async def test_stream_tool_call_end_carries_same_tool_call(
        self, tool_executor: AsyncMock
    ) -> None:
        tc = ToolCall(id="tc-1", name="my_tool", arguments={"x": 1})
        runner = AgentRunner(_make_tool_call_client(tc), tool_executor)
        events = await _collect(runner, "Do")
        end_events = [e for e in events if e.event_type == StreamEventType.TOOL_CALL_END]
        assert end_events[0].tool_call is not None
        assert end_events[0].tool_call.name == "my_tool"

    async def test_stream_executor_called_for_tool(self, tool_executor: AsyncMock) -> None:
        tc = ToolCall(id="tc-1", name="my_tool", arguments={})
        runner = AgentRunner(_make_tool_call_client(tc), tool_executor)
        await _collect(runner, "Do")
        tool_executor.execute.assert_called_once_with(tc)


# ---------------------------------------------------------------------------
# TestStreamMaxIterations
# ---------------------------------------------------------------------------


class TestStreamMaxIterations:
    async def test_stream_max_iterations_respected(self, tool_executor: AsyncMock) -> None:
        call_count: dict[str, int] = {"n": 0}

        async def _stream(**kwargs: Any) -> AsyncIterator[StreamChunk]:
            call_count["n"] += 1
            yield StreamChunk(
                delta="",
                is_final=True,
                finish_reason="tool_use",
                token_usage=_token_usage(),
                model="test-model",
            )

        client = MagicMock()
        client.stream = _stream
        tc = ToolCall(id="tc-1", name="loop_tool", arguments={})
        client.complete = AsyncMock(return_value=_model_response(content=None, tool_calls=[tc]))

        runner = AgentRunner(client, tool_executor, config=RunnerConfig(max_iterations=3))
        await _collect(runner, "Loop forever")
        assert call_count["n"] == 3

    async def test_stream_agent_end_emitted_at_max_iterations(
        self, tool_executor: AsyncMock
    ) -> None:
        async def _stream(**kwargs: Any) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(
                delta="",
                is_final=True,
                finish_reason="tool_use",
                token_usage=_token_usage(),
                model="test-model",
            )

        client = MagicMock()
        client.stream = _stream
        tc = ToolCall(id="tc-1", name="loop_tool", arguments={})
        client.complete = AsyncMock(return_value=_model_response(content=None, tool_calls=[tc]))

        runner = AgentRunner(client, tool_executor, config=RunnerConfig(max_iterations=2))
        events = await _collect(runner, "Loop")
        types = [e.event_type for e in events]
        assert StreamEventType.AGENT_END in types


# ---------------------------------------------------------------------------
# TestStreamAccumulatesTokenUsage
# ---------------------------------------------------------------------------


class TestStreamAccumulatesTokenUsage:
    async def test_stream_accumulates_token_usage(self, tool_executor: AsyncMock) -> None:
        usage1 = _token_usage(input_t=10, output_t=5, cost=0.001)
        usage2 = _token_usage(input_t=20, output_t=10, cost=0.002)
        tc = ToolCall(id="tc-1", name="my_tool", arguments={})
        client = _make_tool_call_client(
            tc, final_text="Done!", tool_usage=usage1, text_usage=usage2
        )

        runner = AgentRunner(client, tool_executor)
        events = await _collect(runner, "Do")
        end = events[-1]
        assert end.chunk is not None
        assert end.chunk.token_usage is not None
        assert end.chunk.token_usage.input_tokens == 30  # 10 + 20
        assert end.chunk.token_usage.output_tokens == 15  # 5 + 10

    async def test_stream_agent_end_token_usage_set_for_single_turn(
        self, tool_executor: AsyncMock
    ) -> None:
        runner = AgentRunner(_make_text_client("Hello!"), tool_executor)
        events = await _collect(runner)
        end = events[-1]
        assert end.chunk is not None
        assert end.chunk.token_usage is not None
        assert end.chunk.token_usage.input_tokens > 0


# ---------------------------------------------------------------------------
# TestStreamMemoryIntegration
# ---------------------------------------------------------------------------


class TestStreamMemoryIntegration:
    async def test_stream_memory_still_written(
        self, tool_executor: AsyncMock, memory_manager: AsyncMock
    ) -> None:
        runner = AgentRunner(
            _make_text_client("Hello!"), tool_executor, memory_manager=memory_manager
        )
        await _collect(runner, "Hello", session_id="s42")
        memory_manager.remember.assert_called_once()
        assert memory_manager.remember.call_args.kwargs["session_id"] == "s42"

    async def test_stream_memory_recall_before_loop(
        self, tool_executor: AsyncMock, memory_manager: AsyncMock
    ) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor, memory_manager=memory_manager)
        await _collect(runner, "Hello")
        memory_manager.recall.assert_called_once()

    async def test_stream_skips_memory_when_manager_none(self, tool_executor: AsyncMock) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor, memory_manager=None)
        events = await _collect(runner, "Hello")
        assert events[-1].event_type == StreamEventType.AGENT_END

    async def test_stream_cost_tracker_called(
        self, tool_executor: AsyncMock, cost_tracker: MagicMock
    ) -> None:
        runner = AgentRunner(_make_text_client(), tool_executor, cost_tracker=cost_tracker)
        await _collect(runner)
        cost_tracker.record.assert_called_once()
