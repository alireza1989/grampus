"""Tests for streaming types and model client stream() implementations."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nexus.core.errors import ModelError
from nexus.core.models.anthropic import AnthropicClient
from nexus.core.models.openai import OpenAIClient
from nexus.core.types import (
    Message,
    Role,
    StreamChunk,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Fake streaming helpers
# ---------------------------------------------------------------------------


class FakeAnthropicStream:
    """Fake async context manager mimicking anthropic client.messages.stream()."""

    def __init__(
        self,
        text_chunks: list[str],
        stop_reason: str = "end_turn",
        raise_on_enter: BaseException | None = None,
    ) -> None:
        self._chunks = text_chunks
        self._stop_reason = stop_reason
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self) -> FakeAnthropicStream:
        if self._raise_on_enter:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    @property
    def text_stream(self) -> Any:
        chunks = self._chunks

        async def _gen() -> Any:
            for chunk in chunks:
                yield chunk

        return _gen()

    async def get_final_message(self) -> MagicMock:
        msg = MagicMock()
        msg.stop_reason = self._stop_reason
        msg.usage = MagicMock()
        msg.usage.input_tokens = 100
        msg.usage.output_tokens = 50
        msg.content = []
        return msg


class FakeOpenAIStream:
    """Fake async context manager mimicking openai client.chat.completions.stream()."""

    def __init__(
        self,
        text_chunks: list[str],
        finish_reason: str = "stop",
        raise_on_enter: BaseException | None = None,
    ) -> None:
        self._text_chunks = text_chunks
        self._finish_reason = finish_reason
        self._raise_on_enter = raise_on_enter
        self._iter: Any = None

    async def __aenter__(self) -> FakeOpenAIStream:
        if self._raise_on_enter:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def __aiter__(self) -> FakeOpenAIStream:
        text_chunks = self._text_chunks

        async def _gen() -> Any:
            for text in text_chunks:
                chunk = MagicMock()
                choice = MagicMock()
                choice.delta = MagicMock()
                choice.delta.content = text
                chunk.choices = [choice]
                yield chunk

        self._iter = _gen().__aiter__()
        return self

    async def __anext__(self) -> MagicMock:
        return await self._iter.__anext__()

    async def get_final_completion(self) -> MagicMock:
        completion = MagicMock()
        choice = MagicMock()
        choice.finish_reason = self._finish_reason
        completion.choices = [choice]
        completion.usage = MagicMock()
        completion.usage.prompt_tokens = 100
        completion.usage.completion_tokens = 50
        completion.usage.total_tokens = 150
        return completion


# ---------------------------------------------------------------------------
# StreamChunk tests
# ---------------------------------------------------------------------------


class TestStreamChunk:
    def test_stream_chunk_default_values(self) -> None:
        chunk = StreamChunk()
        assert chunk.delta == ""
        assert chunk.finish_reason is None
        assert chunk.token_usage is None
        assert chunk.model == ""
        assert chunk.is_final is False

    def test_stream_chunk_is_final_flag(self) -> None:
        chunk = StreamChunk(is_final=True, delta="", model="claude")
        assert chunk.is_final is True

    def test_stream_chunk_with_token_usage(self) -> None:
        usage = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.001, model="m"
        )
        chunk = StreamChunk(delta="", is_final=True, token_usage=usage, model="m")
        assert chunk.token_usage is not None
        assert chunk.token_usage.input_tokens == 100
        assert chunk.token_usage.output_tokens == 50

    def test_stream_chunk_serializes_to_json(self) -> None:
        chunk = StreamChunk(delta="hello", is_final=False, model="m")
        restored = StreamChunk.model_validate_json(chunk.model_dump_json())
        assert restored.delta == "hello"
        assert restored.is_final is False

        usage = TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0, model="m"
        )
        chunk_final = StreamChunk(
            delta="", is_final=True, finish_reason="end_turn", token_usage=usage, model="m"
        )
        restored_final = StreamChunk.model_validate_json(chunk_final.model_dump_json())
        assert restored_final.is_final is True
        assert restored_final.finish_reason == "end_turn"
        assert restored_final.token_usage is not None


# ---------------------------------------------------------------------------
# StreamEvent tests
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_stream_event_token_type(self) -> None:
        chunk = StreamChunk(delta="hello", model="m")
        event = StreamEvent(event_type=StreamEventType.TOKEN, chunk=chunk)
        assert event.event_type == StreamEventType.TOKEN
        assert event.chunk is not None
        assert event.chunk.delta == "hello"

    def test_stream_event_tool_call_start_type(self) -> None:
        tc = ToolCall(id="call_1", name="search", arguments={"q": "test"})
        event = StreamEvent(event_type=StreamEventType.TOOL_CALL_START, tool_call=tc)
        assert event.event_type == StreamEventType.TOOL_CALL_START
        assert event.tool_call is not None
        assert event.tool_call.name == "search"

    def test_stream_event_tool_call_end_type(self) -> None:
        tr = ToolResult(tool_call_id="call_1", output="result text", duration_ms=50)
        event = StreamEvent(event_type=StreamEventType.TOOL_CALL_END, tool_result=tr)
        assert event.event_type == StreamEventType.TOOL_CALL_END
        assert event.tool_result is not None
        assert event.tool_result.output == "result text"

    def test_stream_event_agent_lifecycle_types(self) -> None:
        for et in [
            StreamEventType.AGENT_START,
            StreamEventType.AGENT_END,
            StreamEventType.ITERATION_START,
            StreamEventType.ERROR,
        ]:
            event = StreamEvent(event_type=et, message=f"event {et}")
            assert event.event_type == et
            assert event.message == f"event {et}"

    def test_stream_event_serializes_to_json(self) -> None:
        chunk = StreamChunk(delta="word ", model="m")
        event = StreamEvent(event_type=StreamEventType.TOKEN, chunk=chunk, iteration=2)
        restored = StreamEvent.model_validate_json(event.model_dump_json())
        assert restored.event_type == StreamEventType.TOKEN
        assert restored.chunk is not None
        assert restored.chunk.delta == "word "
        assert restored.iteration == 2


# ---------------------------------------------------------------------------
# AnthropicClient.stream() tests
# ---------------------------------------------------------------------------


class TestAnthropicClientStream:
    def _make_client(self, stream_obj: Any) -> AnthropicClient:
        mock_sdk = MagicMock()
        mock_sdk.messages.stream = MagicMock(return_value=stream_obj)
        return AnthropicClient(api_key="sk-test", _client=mock_sdk)

    async def test_stream_yields_stream_chunks(self) -> None:
        client = self._make_client(FakeAnthropicStream(["hello", " world"]))
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="claude-3-5-haiku-20241022"
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, StreamChunk) for c in chunks)

    async def test_stream_chunks_are_stream_chunk_instances(self) -> None:
        client = self._make_client(FakeAnthropicStream(["tok1", "tok2", "tok3"]))
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="claude-3-5-haiku-20241022"
        ):
            chunks.append(chunk)
        for c in chunks:
            assert isinstance(c, StreamChunk)

    async def test_stream_final_chunk_has_is_final_true(self) -> None:
        client = self._make_client(FakeAnthropicStream(["hello", " world"]))
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="claude-3-5-haiku-20241022"
        ):
            chunks.append(chunk)
        assert chunks[-1].is_final is True

    async def test_stream_final_chunk_has_token_usage(self) -> None:
        client = self._make_client(FakeAnthropicStream(["hello"]))
        final_chunk = None
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="claude-3-5-haiku-20241022"
        ):
            if chunk.is_final:
                final_chunk = chunk
        assert final_chunk is not None
        assert final_chunk.token_usage is not None
        assert final_chunk.token_usage.input_tokens == 100
        assert final_chunk.token_usage.output_tokens == 50
        assert final_chunk.token_usage.total_tokens == 150

    async def test_stream_empty_response_yields_final_chunk(self) -> None:
        client = self._make_client(FakeAnthropicStream([]))
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="claude-3-5-haiku-20241022"
        ):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].is_final is True

    async def test_stream_api_error_raises_model_error(self) -> None:
        client = self._make_client(FakeAnthropicStream([], raise_on_enter=Exception("API failure")))
        with pytest.raises(ModelError) as exc_info:
            async for _ in client.stream(
                messages=[Message(role=Role.USER, content="hi")], model="claude-3-5-haiku-20241022"
            ):
                pass
        assert exc_info.value.code == "MODEL_API_ERROR"


# ---------------------------------------------------------------------------
# OpenAIClient.stream() tests
# ---------------------------------------------------------------------------


class TestOpenAIClientStream:
    def _make_client(self, stream_obj: Any) -> OpenAIClient:
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.stream = MagicMock(return_value=stream_obj)
        return OpenAIClient(api_key="sk-test", _client=mock_sdk)

    async def test_stream_yields_stream_chunks(self) -> None:
        client = self._make_client(FakeOpenAIStream(["hello", " world"]))
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="gpt-4o-mini"
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, StreamChunk) for c in chunks)

    async def test_stream_final_chunk_has_is_final_true(self) -> None:
        client = self._make_client(FakeOpenAIStream(["hello", " world"]))
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="gpt-4o-mini"
        ):
            chunks.append(chunk)
        assert chunks[-1].is_final is True

    async def test_stream_final_chunk_has_token_usage(self) -> None:
        client = self._make_client(FakeOpenAIStream(["hello"]))
        final_chunk = None
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="gpt-4o-mini"
        ):
            if chunk.is_final:
                final_chunk = chunk
        assert final_chunk is not None
        assert final_chunk.token_usage is not None
        assert final_chunk.token_usage.input_tokens == 100
        assert final_chunk.token_usage.output_tokens == 50

    async def test_stream_api_error_raises_model_error(self) -> None:
        client = self._make_client(FakeOpenAIStream([], raise_on_enter=Exception("OpenAI error")))
        with pytest.raises(ModelError) as exc_info:
            async for _ in client.stream(
                messages=[Message(role=Role.USER, content="hi")], model="gpt-4o-mini"
            ):
                pass
        assert exc_info.value.code == "MODEL_API_ERROR"


# ---------------------------------------------------------------------------
# MockModelClient.stream() tests
# ---------------------------------------------------------------------------


class TestMockClientStream:
    async def test_mock_stream_yields_words_from_response(self) -> None:
        from demos.deep_research._mock_client import MockModelClient

        client = MockModelClient()
        # "Worker results:" triggers the supervisor to return a text report directly.
        messages = [Message(role=Role.USER, content="Worker results: Research Report: test topic")]
        non_final_chunks = []
        async for chunk in client.stream(messages=messages, model="mock"):
            if not chunk.is_final:
                non_final_chunks.append(chunk)
        assert len(non_final_chunks) > 0

    async def test_mock_stream_final_chunk_is_final(self) -> None:
        from demos.deep_research._mock_client import MockModelClient

        client = MockModelClient()
        messages = [Message(role=Role.USER, content="Worker results: Research Report: test topic")]
        final = None
        async for chunk in client.stream(messages=messages, model="mock"):
            if chunk.is_final:
                final = chunk
        assert final is not None
        assert final.is_final is True
        assert final.token_usage is not None

    async def test_mock_stream_total_text_matches_complete_output(self) -> None:
        from demos.deep_research._mock_client import MockModelClient

        client = MockModelClient()
        messages = [Message(role=Role.USER, content="Worker results: Research Report: test topic")]

        streamed_text = ""
        async for chunk in client.stream(messages=messages, model="mock"):
            if not chunk.is_final:
                streamed_text += chunk.delta

        response = await client.complete(messages=messages, model="mock")
        expected_text = response.content or ""

        # Word-by-word streaming collapses whitespace, so compare word lists.
        assert streamed_text.split() == expected_text.split()
