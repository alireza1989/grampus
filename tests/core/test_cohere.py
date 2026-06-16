"""Tests for grampus.core.models.cohere — CohereClient."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.errors import ModelError
from grampus.core.models.base import ModelClient, ModelResponse
from grampus.core.models.cohere import (
    CohereClient,
    _normalise_stop_reason,
    _to_cohere_messages,
    _to_cohere_tools,
)
from grampus.core.types import (
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_chat_response(
    text: str = "hello",
    tool_calls: list[Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
    finish_reason: str = "COMPLETE",
) -> MagicMock:
    """Minimal mock of cohere v2 AsyncClientV2.chat() response."""
    resp = MagicMock()
    resp.finish_reason = finish_reason

    # response.message.content = [MagicMock(type="text", text=text)]
    content_item = MagicMock()
    content_item.type = "text"
    content_item.text = text
    resp.message.content = [content_item]

    resp.message.tool_calls = tool_calls or []

    resp.usage.billed_units.input_tokens = input_tokens
    resp.usage.billed_units.output_tokens = output_tokens
    return resp


async def _make_stream_events(
    text_chunks: list[str],
    finish_reason: str = "COMPLETE",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> AsyncIterator[MagicMock]:
    """Yield content-delta events then a single message-end event."""
    for text in text_chunks:
        event = MagicMock()
        event.type = "content-delta"
        content_item = MagicMock()
        content_item.text = text
        event.delta.message.content = [content_item]
        yield event

    end_event = MagicMock()
    end_event.type = "message-end"
    end_event.delta.usage.billed_units.input_tokens = input_tokens
    end_event.delta.usage.billed_units.output_tokens = output_tokens
    end_event.delta.finish_reason = finish_reason
    yield end_event


@pytest.fixture
def mock_sdk() -> MagicMock:
    return MagicMock()


def _make_client(mock_sdk: MagicMock) -> CohereClient:
    return CohereClient(api_key="test-key", _client=mock_sdk)


# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------


class TestToCohereMessages:
    def test_system_inline(self) -> None:
        messages = [Message(role=Role.SYSTEM, content="Be helpful.")]
        result = _to_cohere_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful."

    def test_user_and_assistant(self) -> None:
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
        ]
        result = _to_cohere_messages(messages)
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_tool_result(self) -> None:
        tr = ToolResult(tool_call_id="call-1", output="42")
        msg = Message(role=Role.TOOL, tool_results=[tr])
        result = _to_cohere_messages([msg])
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call-1"
        assert result[0]["content"] == "42"

    def test_tool_result_error_fallback(self) -> None:
        tr = ToolResult(tool_call_id="call-2", output=None, error="failed")
        msg = Message(role=Role.TOOL, tool_results=[tr])
        result = _to_cohere_messages([msg])
        assert result[0]["content"] == "failed"

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "grampus"})
        msg = Message(role=Role.ASSISTANT, content="Searching…", tool_calls=[tc])
        result = _to_cohere_messages([msg])
        assert len(result) == 1
        entry = result[0]
        assert entry["role"] == "assistant"
        assert entry["content"] == "Searching…"
        assert len(entry["tool_calls"]) == 1
        tc_out = entry["tool_calls"][0]
        assert tc_out["id"] == "tc-1"
        assert tc_out["type"] == "function"
        assert tc_out["function"]["name"] == "search"
        assert json.loads(tc_out["function"]["arguments"]) == {"q": "grampus"}

    def test_system_not_skipped(self) -> None:
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
        ]
        result = _to_cohere_messages(messages)
        roles = [m["role"] for m in result]
        assert "system" in roles

    def test_multiple_tool_results_expand(self) -> None:
        tr1 = ToolResult(tool_call_id="c1", output="res1")
        tr2 = ToolResult(tool_call_id="c2", output="res2")
        msg = Message(role=Role.TOOL, tool_results=[tr1, tr2])
        result = _to_cohere_messages([msg])
        assert len(result) == 2
        assert result[0]["tool_call_id"] == "c1"
        assert result[1]["tool_call_id"] == "c2"


# ---------------------------------------------------------------------------
# Tool conversion tests
# ---------------------------------------------------------------------------


class TestToCohereTools:
    def test_wraps_in_function_type(self) -> None:
        tool = ToolDefinition(name="search", description="search the web")
        result = _to_cohere_tools([tool])
        assert result[0]["type"] == "function"

    def test_uses_to_function_schema(self) -> None:
        tool = ToolDefinition(
            name="calculate",
            description="do math",
            parameters=[ToolParameter(name="expr", type="string", description="expression")],
        )
        result = _to_cohere_tools([tool])
        func = result[0]["function"]
        assert func["name"] == "calculate"
        assert "parameters" in func


# ---------------------------------------------------------------------------
# Stop reason normaliser tests
# ---------------------------------------------------------------------------


class TestNormaliseStopReason:
    def test_complete_to_end_turn(self) -> None:
        assert _normalise_stop_reason("COMPLETE") == "end_turn"

    def test_tool_call_to_tool_use(self) -> None:
        assert _normalise_stop_reason("TOOL_CALL") == "tool_use"

    def test_max_tokens(self) -> None:
        assert _normalise_stop_reason("MAX_TOKENS") == "max_tokens"

    def test_error_maps(self) -> None:
        assert _normalise_stop_reason("ERROR") == "error"

    def test_unknown_defaults_to_end_turn(self) -> None:
        assert _normalise_stop_reason("SOME_FUTURE_REASON") == "end_turn"

    def test_none_defaults_to_end_turn(self) -> None:
        assert _normalise_stop_reason(None) == "end_turn"


# ---------------------------------------------------------------------------
# complete() tests
# ---------------------------------------------------------------------------


class TestCohereClientComplete:
    @pytest.mark.asyncio
    async def test_returns_model_response(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat = AsyncMock(return_value=_make_chat_response("hello world"))
        client = _make_client(mock_sdk)
        result = await client.complete(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        )
        assert isinstance(result, ModelResponse)
        assert result.content == "hello world"
        assert result.model == "command-a-03-2025"
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_token_usage_from_billed_units(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat = AsyncMock(
            return_value=_make_chat_response(input_tokens=100, output_tokens=50)
        )
        client = _make_client(mock_sdk)
        result = await client.complete(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        )
        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 50
        assert result.token_usage.total_tokens == 150

    @pytest.mark.asyncio
    async def test_with_tools_passes_tools_to_sdk(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat = AsyncMock(return_value=_make_chat_response("ok"))
        client = _make_client(mock_sdk)
        tool = ToolDefinition(name="search", description="search")
        await client.complete(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
            tools=[tool],
        )
        call_kwargs = mock_sdk.chat.call_args.kwargs
        assert "tools" in call_kwargs

    @pytest.mark.asyncio
    async def test_tool_call_in_response(self, mock_sdk: MagicMock) -> None:
        tc_mock = MagicMock()
        tc_mock.id = "tc-abc"
        fn_mock = MagicMock()
        fn_mock.name = "search"
        fn_mock.arguments = '{"q": "grampus"}'
        tc_mock.function = fn_mock

        mock_sdk.chat = AsyncMock(return_value=_make_chat_response(tool_calls=[tc_mock]))
        client = _make_client(mock_sdk)
        result = await client.complete(
            messages=[Message(role=Role.USER, content="search")],
            model="command-a-03-2025",
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-abc"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "grampus"}

    @pytest.mark.asyncio
    async def test_no_text_content_returns_none(self, mock_sdk: MagicMock) -> None:
        resp = MagicMock()
        resp.finish_reason = "COMPLETE"
        resp.message.content = []  # no content items
        resp.message.tool_calls = []
        resp.usage.billed_units.input_tokens = 5
        resp.usage.billed_units.output_tokens = 5
        mock_sdk.chat = AsyncMock(return_value=resp)
        client = _make_client(mock_sdk)
        result = await client.complete(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        )
        assert result.content is None

    @pytest.mark.asyncio
    async def test_api_error_raises_model_error(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat = AsyncMock(side_effect=Exception("500 internal server error"))
        client = _make_client(mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="command-a-03-2025",
            )
        assert exc_info.value.code == "MODEL_API_ERROR"
        assert exc_info.value.details["provider"] == "cohere"

    @pytest.mark.asyncio
    async def test_auth_error_hint_mentions_api_key(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat = AsyncMock(side_effect=Exception("401 Unauthorized"))
        client = _make_client(mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="command-a-03-2025",
            )
        assert "COHERE_API_KEY" in exc_info.value.hint

    @pytest.mark.asyncio
    async def test_non_auth_error_hint_mentions_status_page(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat = AsyncMock(side_effect=Exception("rate limit exceeded"))
        client = _make_client(mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="command-a-03-2025",
            )
        assert "status page" in exc_info.value.hint

    @pytest.mark.asyncio
    async def test_cost_computed_from_pricing(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat = AsyncMock(
            return_value=_make_chat_response(input_tokens=1_000_000, output_tokens=1_000_000)
        )
        client = _make_client(mock_sdk)
        result = await client.complete(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        )
        # command-a-03-2025: $2.50/M input + $10.00/M output = $12.50
        assert result.token_usage.cost_usd == pytest.approx(12.50)

    @pytest.mark.asyncio
    async def test_cheap_model_pricing(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat = AsyncMock(
            return_value=_make_chat_response(input_tokens=1_000_000, output_tokens=1_000_000)
        )
        client = _make_client(mock_sdk)
        result = await client.complete(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-r7b-12-2024",
        )
        # command-r7b-12-2024: $0.0375/M + $0.15/M = $0.1875
        assert result.token_usage.cost_usd == pytest.approx(0.1875)

    def test_is_subclass_of_model_client(self) -> None:
        assert issubclass(CohereClient, ModelClient)


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


class TestCohereClientStream:
    @pytest.mark.asyncio
    async def test_yields_text_chunks(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat_stream = MagicMock(return_value=_make_stream_events(["Hello", " world"]))
        client = _make_client(mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        ):
            chunks.append(chunk)
        non_final = [c for c in chunks if not c.is_final]
        assert len(non_final) == 2
        assert non_final[0].delta == "Hello"
        assert non_final[1].delta == " world"

    @pytest.mark.asyncio
    async def test_final_chunk_has_usage(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat_stream = MagicMock(
            return_value=_make_stream_events(["hi"], input_tokens=15, output_tokens=30)
        )
        client = _make_client(mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        ):
            chunks.append(chunk)
        final = next(c for c in chunks if c.is_final)
        assert final.token_usage is not None
        assert final.token_usage.input_tokens == 15
        assert final.token_usage.output_tokens == 30
        assert final.token_usage.total_tokens == 45

    @pytest.mark.asyncio
    async def test_final_chunk_cost_computed(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat_stream = MagicMock(
            return_value=_make_stream_events(
                ["hi"], input_tokens=1_000_000, output_tokens=1_000_000
            )
        )
        client = _make_client(mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        ):
            chunks.append(chunk)
        final = next(c for c in chunks if c.is_final)
        assert final.token_usage is not None
        assert final.token_usage.cost_usd == pytest.approx(12.50)

    @pytest.mark.asyncio
    async def test_skips_non_content_events(self, mock_sdk: MagicMock) -> None:
        async def _mixed_stream() -> AsyncIterator[MagicMock]:
            # stream-start event (should be ignored)
            start = MagicMock()
            start.type = "stream-start"
            yield start
            # tool-plan-delta (should be ignored)
            plan = MagicMock()
            plan.type = "tool-plan-delta"
            yield plan
            # real content
            content_event = MagicMock()
            content_event.type = "content-delta"
            item = MagicMock()
            item.text = "actual text"
            content_event.delta.message.content = [item]
            yield content_event
            # end
            end = MagicMock()
            end.type = "message-end"
            end.delta.usage.billed_units.input_tokens = 5
            end.delta.usage.billed_units.output_tokens = 5
            end.delta.finish_reason = "COMPLETE"
            yield end

        mock_sdk.chat_stream = MagicMock(return_value=_mixed_stream())
        client = _make_client(mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        ):
            chunks.append(chunk)
        non_final = [c for c in chunks if not c.is_final]
        assert len(non_final) == 1
        assert non_final[0].delta == "actual text"

    @pytest.mark.asyncio
    async def test_empty_delta_text_not_yielded(self, mock_sdk: MagicMock) -> None:
        async def _empty_delta_stream() -> AsyncIterator[MagicMock]:
            event = MagicMock()
            event.type = "content-delta"
            item = MagicMock()
            item.text = ""
            event.delta.message.content = [item]
            yield event
            end = MagicMock()
            end.type = "message-end"
            end.delta.usage.billed_units.input_tokens = 1
            end.delta.usage.billed_units.output_tokens = 1
            end.delta.finish_reason = "COMPLETE"
            yield end

        mock_sdk.chat_stream = MagicMock(return_value=_empty_delta_stream())
        client = _make_client(mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="command-a-03-2025",
        ):
            chunks.append(chunk)
        non_final = [c for c in chunks if not c.is_final]
        assert len(non_final) == 0

    @pytest.mark.asyncio
    async def test_api_error_raises_model_error(self, mock_sdk: MagicMock) -> None:
        mock_sdk.chat_stream = MagicMock(side_effect=Exception("network error"))
        client = _make_client(mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            async for _ in client.stream(
                messages=[Message(role=Role.USER, content="hi")],
                model="command-a-03-2025",
            ):
                pass
        assert exc_info.value.code == "MODEL_API_ERROR"

    def test_is_async_generator(self) -> None:
        assert inspect.isasyncgenfunction(CohereClient.stream)
