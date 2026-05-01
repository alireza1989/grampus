"""Tests for nexus.core.models — base ABC and provider clients."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import ModelError
from nexus.core.models.anthropic import AnthropicClient
from nexus.core.models.base import ModelClient, ModelResponse
from nexus.core.models.openai import OpenAIClient
from nexus.core.types import Message, Role, TokenUsage, ToolCall, ToolDefinition, ToolParameter

# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_basic(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0, model="m")
        resp = ModelResponse(
            content="Hello",
            tool_calls=[],
            token_usage=usage,
            model="claude-3-haiku",
            stop_reason="end_turn",
        )
        assert resp.content == "Hello"
        assert resp.stop_reason == "end_turn"

    def test_tool_calls_field(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments={"q": "test"})
        usage = TokenUsage(input_tokens=5, output_tokens=2, total_tokens=7, cost_usd=0.0, model="m")
        resp = ModelResponse(
            content=None,
            tool_calls=[tc],
            token_usage=usage,
            model="m",
            stop_reason="tool_use",
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"

    def test_json_round_trip(self) -> None:
        usage = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="m")
        resp = ModelResponse(content="hi", tool_calls=[], token_usage=usage, model="m", stop_reason="end_turn")
        restored = ModelResponse.model_validate_json(resp.model_dump_json())
        assert restored.content == resp.content


# ---------------------------------------------------------------------------
# ModelClient ABC
# ---------------------------------------------------------------------------


class TestModelClientABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ModelClient()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_complete(self) -> None:
        class Incomplete(ModelClient):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


def _make_anthropic_mock_response(
    content_text: str = "hello",
    tool_name: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a minimal mock of anthropic.types.Message."""
    msg = MagicMock()
    msg.stop_reason = "end_turn" if not tool_name else "tool_use"
    msg.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    msg.model = "claude-3-5-haiku-20241022"

    if tool_name:
        block = MagicMock()
        block.type = "tool_use"
        block.id = "toolu_123"
        block.name = tool_name
        block.input = {"query": "test"}
        msg.content = [block]
    else:
        block = MagicMock()
        block.type = "text"
        block.text = content_text
        msg.content = [block]

    return msg


class TestAnthropicClient:
    def _make_client(self) -> AnthropicClient:
        mock_sdk = MagicMock()
        mock_sdk.messages = MagicMock()
        mock_sdk.messages.create = AsyncMock(
            return_value=_make_anthropic_mock_response("hi there")
        )
        return AnthropicClient(api_key="sk-test", _client=mock_sdk)

    def test_instantiate(self) -> None:
        client = self._make_client()
        assert isinstance(client, ModelClient)

    async def test_complete_returns_model_response(self) -> None:
        client = self._make_client()
        messages = [Message(role=Role.USER, content="Hello")]
        response = await client.complete(messages=messages, model="claude-3-5-haiku-20241022")
        assert isinstance(response, ModelResponse)
        assert response.content == "hi there"

    async def test_complete_extracts_token_usage(self) -> None:
        client = self._make_client()
        messages = [Message(role=Role.USER, content="hi")]
        response = await client.complete(messages=messages, model="claude-3-5-haiku-20241022")
        assert response.token_usage.input_tokens == 10
        assert response.token_usage.output_tokens == 5
        assert response.token_usage.total_tokens == 15

    async def test_complete_with_tool_use(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.messages = MagicMock()
        mock_sdk.messages.create = AsyncMock(
            return_value=_make_anthropic_mock_response(tool_name="web_search")
        )
        client = AnthropicClient(api_key="sk-test", _client=mock_sdk)
        tool = ToolDefinition(
            name="web_search",
            description="search",
            parameters=[ToolParameter(name="query", type="string", description="q", required=True)],
        )
        messages = [Message(role=Role.USER, content="search for nexus")]
        response = await client.complete(
            messages=messages, model="claude-3-5-haiku-20241022", tools=[tool]
        )
        assert response.stop_reason == "tool_use"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "web_search"

    async def test_api_error_raises_model_error(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.messages = MagicMock()
        mock_sdk.messages.create = AsyncMock(side_effect=Exception("API error"))
        client = AnthropicClient(api_key="sk-test", _client=mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="claude-3-5-haiku-20241022",
            )
        assert exc_info.value.code == "MODEL_API_ERROR"

    async def test_converts_system_message(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.messages = MagicMock()
        mock_sdk.messages.create = AsyncMock(
            return_value=_make_anthropic_mock_response("ok")
        )
        client = AnthropicClient(api_key="sk-test", _client=mock_sdk)
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="hello"),
        ]
        await client.complete(messages=messages, model="claude-3-5-haiku-20241022")
        call_kwargs = mock_sdk.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "You are helpful."

    async def test_stream_yields_chunks(self) -> None:
        mock_sdk = MagicMock()

        class FakeAnthropicStream:
            async def __aenter__(self) -> "FakeAnthropicStream":
                return self

            async def __aexit__(self, *args: object) -> None:
                pass

            def __aiter__(self) -> "FakeAnthropicStream":
                self._events = iter([
                    MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="chunk1")),
                    MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="chunk2")),
                    MagicMock(type="message_stop"),
                ])
                return self

            async def __anext__(self) -> MagicMock:
                try:
                    return next(self._events)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        mock_sdk.messages = MagicMock()
        mock_sdk.messages.stream = MagicMock(return_value=FakeAnthropicStream())

        client = AnthropicClient(api_key="sk-test", _client=mock_sdk)
        messages = [Message(role=Role.USER, content="hi")]
        chunks = []
        async for chunk in client.stream(messages=messages, model="claude-3-5-haiku-20241022"):
            chunks.append(chunk)
        assert len(chunks) > 0

    async def test_cost_calculation(self) -> None:
        client = self._make_client()
        messages = [Message(role=Role.USER, content="hi")]
        response = await client.complete(messages=messages, model="claude-3-5-haiku-20241022")
        assert response.token_usage.cost_usd >= 0.0


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


def _make_openai_mock_response(
    content_text: str = "hello",
    tool_name: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a minimal mock of openai.types.chat.ChatCompletion."""
    resp = MagicMock()
    resp.model = "gpt-4o-mini"
    resp.usage = MagicMock(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )

    choice = MagicMock()
    choice.finish_reason = "stop" if not tool_name else "tool_calls"

    if tool_name:
        func_mock = MagicMock()
        func_mock.name = tool_name  # must set after construction; MagicMock(name=x) sets repr name
        func_mock.arguments = '{"query": "test"}'
        tc = MagicMock()
        tc.id = "call_abc"
        tc.function = func_mock
        choice.message = MagicMock(content=None, tool_calls=[tc])
    else:
        choice.message = MagicMock(content=content_text, tool_calls=None)

    resp.choices = [choice]
    return resp


class TestOpenAIClient:
    def _make_client(self) -> OpenAIClient:
        mock_sdk = MagicMock()
        mock_sdk.chat = MagicMock()
        mock_sdk.chat.completions = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_make_openai_mock_response("hello")
        )
        return OpenAIClient(api_key="sk-test", _client=mock_sdk)

    def test_instantiate(self) -> None:
        client = self._make_client()
        assert isinstance(client, ModelClient)

    async def test_complete_returns_model_response(self) -> None:
        client = self._make_client()
        messages = [Message(role=Role.USER, content="Hello")]
        response = await client.complete(messages=messages, model="gpt-4o-mini")
        assert isinstance(response, ModelResponse)
        assert response.content == "hello"

    async def test_complete_extracts_token_usage(self) -> None:
        client = self._make_client()
        messages = [Message(role=Role.USER, content="hi")]
        response = await client.complete(messages=messages, model="gpt-4o-mini")
        assert response.token_usage.input_tokens == 10
        assert response.token_usage.output_tokens == 5
        assert response.token_usage.total_tokens == 15

    async def test_complete_with_tool_use(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = MagicMock()
        mock_sdk.chat.completions = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_make_openai_mock_response(tool_name="web_search")
        )
        client = OpenAIClient(api_key="sk-test", _client=mock_sdk)
        tool = ToolDefinition(
            name="web_search",
            description="search",
            parameters=[ToolParameter(name="query", type="string", description="q", required=True)],
        )
        messages = [Message(role=Role.USER, content="search for nexus")]
        response = await client.complete(messages=messages, model="gpt-4o-mini", tools=[tool])
        assert response.stop_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "web_search"

    async def test_api_error_raises_model_error(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = MagicMock()
        mock_sdk.chat.completions = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        client = OpenAIClient(api_key="sk-test", _client=mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="gpt-4o-mini",
            )
        assert exc_info.value.code == "MODEL_API_ERROR"

    async def test_system_message_included(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = MagicMock()
        mock_sdk.chat.completions = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_make_openai_mock_response("ok")
        )
        client = OpenAIClient(api_key="sk-test", _client=mock_sdk)
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="hello"),
        ]
        await client.complete(messages=messages, model="gpt-4o-mini")
        call_kwargs = mock_sdk.chat.completions.create.call_args
        oai_messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else []
        if not oai_messages:
            oai_messages = call_kwargs.kwargs.get("messages", [])
        system_msgs = [m for m in oai_messages if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful."

    async def test_stream_yields_chunks(self) -> None:
        mock_sdk = MagicMock()

        async def fake_chunks() -> None:
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock(delta=MagicMock(content="hello", tool_calls=None), finish_reason=None)]
            yield chunk1
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock(delta=MagicMock(content=" world", tool_calls=None), finish_reason="stop")]
            yield chunk2

        mock_sdk.chat = MagicMock()
        mock_sdk.chat.completions = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=fake_chunks())
        client = OpenAIClient(api_key="sk-test", _client=mock_sdk)
        chunks = []
        async for chunk in client.stream(messages=[Message(role=Role.USER, content="hi")], model="gpt-4o-mini"):
            chunks.append(chunk)
        assert len(chunks) > 0

    async def test_cost_calculation(self) -> None:
        client = self._make_client()
        messages = [Message(role=Role.USER, content="hi")]
        response = await client.complete(messages=messages, model="gpt-4o-mini")
        assert response.token_usage.cost_usd >= 0.0
