"""Tests for nexus.core.models — base ABC and provider clients."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import ModelError
from nexus.core.models.anthropic import AnthropicClient
from nexus.core.models.base import ModelClient, ModelResponse
from nexus.core.models.gemini import GeminiClient, _stop_reason, _to_gemini_contents
from nexus.core.models.openai import OpenAIClient
from nexus.core.types import (
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_basic(self) -> None:
        usage = TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0, model="m"
        )
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
        resp = ModelResponse(
            content="hi", tool_calls=[], token_usage=usage, model="m", stop_reason="end_turn"
        )
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
        mock_sdk.messages.create = AsyncMock(return_value=_make_anthropic_mock_response("hi there"))
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
        mock_sdk.messages.create = AsyncMock(return_value=_make_anthropic_mock_response("ok"))
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

            @property
            def text_stream(self) -> Any:
                async def _gen() -> Any:
                    yield "chunk1"
                    yield "chunk2"

                return _gen()

            async def get_final_message(self) -> MagicMock:
                msg = MagicMock()
                msg.stop_reason = "end_turn"
                msg.usage = MagicMock()
                msg.usage.input_tokens = 10
                msg.usage.output_tokens = 5
                return msg

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
        mock_sdk.chat.completions.create = AsyncMock(return_value=_make_openai_mock_response("ok"))
        client = OpenAIClient(api_key="sk-test", _client=mock_sdk)
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="hello"),
        ]
        await client.complete(messages=messages, model="gpt-4o-mini")
        call_kwargs = mock_sdk.chat.completions.create.call_args
        oai_messages = (
            call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else []
        )
        if not oai_messages:
            oai_messages = call_kwargs.kwargs.get("messages", [])
        system_msgs = [m for m in oai_messages if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful."

    async def test_stream_yields_chunks(self) -> None:
        mock_sdk = MagicMock()

        class FakeOpenAIStream:
            def __init__(self) -> None:
                self._iter: Any = None

            async def __aenter__(self) -> "FakeOpenAIStream":
                return self

            async def __aexit__(self, *args: object) -> None:
                pass

            def __aiter__(self) -> "FakeOpenAIStream":
                async def _gen() -> Any:
                    for text in ["hello", " world"]:
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
                choice.finish_reason = "stop"
                completion.choices = [choice]
                completion.usage = MagicMock()
                completion.usage.prompt_tokens = 10
                completion.usage.completion_tokens = 5
                completion.usage.total_tokens = 15
                return completion

        mock_sdk.chat = MagicMock()
        mock_sdk.chat.completions = MagicMock()
        mock_sdk.chat.completions.stream = MagicMock(return_value=FakeOpenAIStream())
        client = OpenAIClient(api_key="sk-test", _client=mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")], model="gpt-4o-mini"
        ):
            chunks.append(chunk)
        assert len(chunks) > 0

    async def test_cost_calculation(self) -> None:
        client = self._make_client()
        messages = [Message(role=Role.USER, content="hi")]
        response = await client.complete(messages=messages, model="gpt-4o-mini")
        assert response.token_usage.cost_usd >= 0.0


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------


def _make_gemini_mock_part(text: str | None = None, function_call: Any = None) -> MagicMock:
    part = MagicMock()
    part.text = text
    part.function_call = function_call
    return part


def _make_gemini_mock_response(
    text: str = "hello",
    tool_name: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    finish_reason: str = "STOP",
) -> MagicMock:
    resp = MagicMock()
    resp.usage_metadata = MagicMock(
        prompt_token_count=input_tokens,
        candidates_token_count=output_tokens,
    )
    if tool_name:
        fc = MagicMock()
        fc.name = tool_name
        fc.args = {"query": "test"}
        part = _make_gemini_mock_part(function_call=fc)
        part.text = None
    else:
        part = _make_gemini_mock_part(text=text)
        part.function_call = None
    resp.candidates = [
        MagicMock(
            content=MagicMock(parts=[part]),
            finish_reason=finish_reason,
        )
    ]
    return resp


def _make_gemini_chunk(
    text: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    finish_reason: str | None = None,
) -> MagicMock:
    chunk = MagicMock()
    chunk.usage_metadata = MagicMock(
        prompt_token_count=input_tokens,
        candidates_token_count=output_tokens,
    )
    part = MagicMock()
    part.text = text if text else None
    candidate = MagicMock(
        content=MagicMock(parts=[part] if text else []),
        finish_reason=finish_reason,
    )
    chunk.candidates = [candidate]
    return chunk


class TestGeminiClient:
    def _make_client(self, mock_response: MagicMock | None = None) -> GeminiClient:
        mock_sdk = MagicMock()
        if mock_response is None:
            mock_response = _make_gemini_mock_response("hello")
        mock_sdk.aio = MagicMock()
        mock_sdk.aio.models = MagicMock()
        mock_sdk.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # Stub for GenerateContentConfig used in complete/stream
        import sys

        types_mod = MagicMock()
        types_mod.GenerateContentConfig = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        sys.modules.setdefault("google", MagicMock())
        sys.modules.setdefault("google.genai", MagicMock())
        sys.modules["google.genai.types"] = types_mod

        return GeminiClient(api_key="test-key", _client=mock_sdk)

    async def test_complete_returns_model_response(self) -> None:
        client = self._make_client()
        result = await client.complete(
            messages=[Message(role=Role.USER, content="Hello")],
            model="gemini-2.0-flash-001",
        )
        assert isinstance(result, ModelResponse)
        assert result.content == "hello"
        assert result.token_usage.input_tokens == 10
        assert result.stop_reason == "end_turn"

    async def test_complete_with_tool_call(self) -> None:
        client = self._make_client(_make_gemini_mock_response(tool_name="search"))
        result = await client.complete(
            messages=[Message(role=Role.USER, content="search something")],
            model="gemini-2.0-flash-001",
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "test"}
        assert result.content is None

    async def test_complete_system_message_extracted(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.aio = MagicMock()
        mock_sdk.aio.models = MagicMock()
        mock_sdk.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_mock_response("ok")
        )
        import sys

        types_mod = MagicMock()
        captured: dict[str, Any] = {}

        def capture_config(**kw: Any) -> MagicMock:
            captured.update(kw)
            m = MagicMock()
            for k, v in kw.items():
                setattr(m, k, v)
            return m

        types_mod.GenerateContentConfig = MagicMock(side_effect=capture_config)
        sys.modules.setdefault("google.genai", MagicMock())
        sys.modules["google.genai"].types = types_mod
        sys.modules["google.genai.types"] = types_mod

        client = GeminiClient(api_key="test-key", _client=mock_sdk)
        system_msg = Message(role=Role.SYSTEM, content="You are helpful.")
        user_msg = Message(role=Role.USER, content="hello")
        await client.complete(
            messages=[system_msg, user_msg],
            model="gemini-2.0-flash-001",
        )

        assert captured.get("system_instruction") == "You are helpful."
        call_args = mock_sdk.aio.models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        assert not any(c.get("role") == "system" for c in contents if isinstance(c, dict))

    async def test_complete_with_tools(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.aio = MagicMock()
        mock_sdk.aio.models = MagicMock()
        mock_sdk.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_mock_response("ok")
        )
        import sys

        types_mod = MagicMock()
        captured: dict[str, Any] = {}

        def capture_config(**kw: Any) -> MagicMock:
            captured.update(kw)
            m = MagicMock()
            for k, v in kw.items():
                setattr(m, k, v)
            return m

        types_mod.GenerateContentConfig = MagicMock(side_effect=capture_config)
        sys.modules.setdefault("google.genai", MagicMock())
        sys.modules["google.genai"].types = types_mod
        sys.modules["google.genai.types"] = types_mod

        client = GeminiClient(api_key="test-key", _client=mock_sdk)
        tool = ToolDefinition(
            name="search",
            description="search the web",
            parameters=[ToolParameter(name="query", type="string", description="q", required=True)],
        )
        await client.complete(
            messages=[Message(role=Role.USER, content="search something")],
            model="gemini-2.0-flash-001",
            tools=[tool],
        )
        assert captured.get("tools") is not None

    async def test_complete_api_error_raises_model_error(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.aio = MagicMock()
        mock_sdk.aio.models = MagicMock()
        mock_sdk.aio.models.generate_content = AsyncMock(side_effect=Exception("api_key invalid"))
        import sys

        types_mod = MagicMock()
        types_mod.GenerateContentConfig = MagicMock(return_value=MagicMock())
        sys.modules["google.genai.types"] = types_mod

        client = GeminiClient(api_key="test-key", _client=mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="gemini-2.0-flash-001",
            )
        assert exc_info.value.code == "MODEL_API_ERROR"
        assert "GOOGLE_API_KEY" in (exc_info.value.hint or "")

    async def test_complete_api_error_non_auth(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.aio = MagicMock()
        mock_sdk.aio.models = MagicMock()
        mock_sdk.aio.models.generate_content = AsyncMock(
            side_effect=Exception("rate limit exceeded")
        )
        import sys

        types_mod = MagicMock()
        types_mod.GenerateContentConfig = MagicMock(return_value=MagicMock())
        sys.modules["google.genai.types"] = types_mod

        client = GeminiClient(api_key="test-key", _client=mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="gemini-2.0-flash-001",
            )
        assert "status page" in (exc_info.value.hint or "")

    async def test_complete_cost_calculated(self) -> None:
        client = self._make_client(
            _make_gemini_mock_response("ok", input_tokens=1000, output_tokens=500)
        )
        result = await client.complete(
            messages=[Message(role=Role.USER, content="hi")],
            model="gemini-2.0-flash-001",
        )
        expected_cost = (1000 * 0.075 + 500 * 0.30) / 1_000_000
        assert result.token_usage.cost_usd == pytest.approx(expected_cost)

    async def test_stream_yields_chunks(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.aio = MagicMock()
        mock_sdk.aio.models = MagicMock()

        async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
            yield _make_gemini_chunk("Hello", input_tokens=0, output_tokens=0)
            yield _make_gemini_chunk(" world", input_tokens=0, output_tokens=0)
            yield _make_gemini_chunk("", input_tokens=10, output_tokens=5, finish_reason="STOP")

        mock_sdk.aio.models.generate_content_stream = AsyncMock(return_value=_fake_stream())

        import sys

        types_mod = MagicMock()
        types_mod.GenerateContentConfig = MagicMock(return_value=MagicMock())
        sys.modules["google.genai.types"] = types_mod

        client = GeminiClient(api_key="test-key", _client=mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="gemini-2.0-flash-001",
        ):
            chunks.append(chunk)

        non_final = [c for c in chunks if not c.is_final]
        assert len(non_final) >= 1
        final_chunks = [c for c in chunks if c.is_final]
        assert len(final_chunks) == 1

    async def test_stream_final_chunk_has_usage(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.aio = MagicMock()
        mock_sdk.aio.models = MagicMock()

        async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
            yield _make_gemini_chunk("text", input_tokens=0, output_tokens=0)
            yield _make_gemini_chunk("", input_tokens=10, output_tokens=5, finish_reason="STOP")

        mock_sdk.aio.models.generate_content_stream = AsyncMock(return_value=_fake_stream())

        import sys

        types_mod = MagicMock()
        types_mod.GenerateContentConfig = MagicMock(return_value=MagicMock())
        sys.modules["google.genai.types"] = types_mod

        client = GeminiClient(api_key="test-key", _client=mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="gemini-2.0-flash-001",
        ):
            chunks.append(chunk)

        final = next(c for c in chunks if c.is_final)
        assert final.token_usage is not None
        assert final.token_usage.total_tokens == 15

    async def test_stream_api_error_raises_model_error(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.aio = MagicMock()
        mock_sdk.aio.models = MagicMock()
        mock_sdk.aio.models.generate_content_stream = AsyncMock(
            side_effect=Exception("stream failed")
        )

        import sys

        types_mod = MagicMock()
        types_mod.GenerateContentConfig = MagicMock(return_value=MagicMock())
        sys.modules["google.genai.types"] = types_mod

        client = GeminiClient(api_key="test-key", _client=mock_sdk)
        with pytest.raises(ModelError):
            async for _ in client.stream(
                messages=[Message(role=Role.USER, content="hi")],
                model="gemini-2.0-flash-001",
            ):
                pass

    def test_is_subclass_of_model_client(self) -> None:
        assert issubclass(GeminiClient, ModelClient)

    def test_tool_result_message_conversion(self) -> None:
        tc = ToolCall(id="call-001", name="search", arguments={"q": "test"})
        tr = ToolResult(tool_call_id="call-001", output="result text")
        messages = [
            Message(role=Role.USER, content="search for me"),
            Message(role=Role.ASSISTANT, tool_calls=[tc]),
            Message(role=Role.TOOL, tool_results=[tr]),
        ]
        contents = _to_gemini_contents(messages)

        last = contents[-1]
        assert last["role"] == "user"
        fn_resp = last["parts"][0]["function_response"]
        assert fn_resp["name"] == "search"
        # id must be echoed back for Gemini 3+ models
        assert fn_resp["id"] == "call-001"

    def test_function_call_id_in_history(self) -> None:
        # function_call parts in assistant history should carry the id for Gemini 3+
        tc = ToolCall(id="gemini-fc-id-abc123", name="search", arguments={"q": "x"})
        messages = [
            Message(role=Role.USER, content="go"),
            Message(role=Role.ASSISTANT, tool_calls=[tc]),
        ]
        contents = _to_gemini_contents(messages)
        assistant_entry = next(c for c in contents if c["role"] == "model")
        fc_part = assistant_entry["parts"][0]["function_call"]
        assert fc_part["id"] == "gemini-fc-id-abc123"

    async def test_gemini3_sdk_id_preserved_as_tool_call_id(self) -> None:
        # When the SDK returns a real string id on function_call (Gemini 3),
        # it should be stored as ToolCall.id instead of generating a UUID.
        fc = MagicMock()
        fc.name = "search"
        fc.args = {"query": "test"}
        fc.id = "fc-real-id-from-sdk"  # string id as Gemini 3 provides

        part = MagicMock()
        part.text = None
        part.function_call = fc

        resp = MagicMock()
        resp.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=3)
        resp.candidates = [MagicMock(content=MagicMock(parts=[part]), finish_reason="STOP")]

        client = self._make_client(resp)
        result = await client.complete(
            messages=[Message(role=Role.USER, content="search")],
            model="gemini-3.5-flash",
        )
        assert result.tool_calls[0].id == "fc-real-id-from-sdk"

    def test_stop_reason_mapping(self) -> None:
        assert _stop_reason("STOP") == "end_turn"
        assert _stop_reason("MAX_TOKENS") == "max_tokens"
        assert _stop_reason("SAFETY") == "safety"
        assert _stop_reason("UNKNOWN") == "stop"


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

from nexus.core.models.ollama import OllamaClient, _to_ollama_messages  # noqa: E402
from nexus.core.models.ollama import _stop_reason as _ollama_stop_reason  # noqa: E402


def _make_ollama_response(
    content: str = "hello",
    tool_name: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    done_reason: str = "stop",
) -> MagicMock:
    resp = MagicMock()
    resp.prompt_eval_count = input_tokens
    resp.eval_count = output_tokens
    resp.done_reason = done_reason
    msg = MagicMock()
    msg.content = content if not tool_name else ""
    if tool_name:
        tc = MagicMock()
        tc.function.name = tool_name
        tc.function.arguments = {"query": "test"}
        msg.tool_calls = [tc]
    else:
        msg.tool_calls = None
    resp.message = msg
    return resp


async def _fake_ollama_stream() -> AsyncIterator[MagicMock]:
    for text in ["Hello", " world"]:
        chunk = MagicMock()
        chunk.done = False
        chunk.message.content = text
        yield chunk
    final = MagicMock()
    final.done = True
    final.done_reason = "stop"
    final.prompt_eval_count = 10
    final.eval_count = 5
    yield final


class TestOllamaClient:
    def _make_client(self, mock_response: MagicMock | None = None) -> OllamaClient:
        mock_sdk = MagicMock()
        if mock_response is None:
            mock_response = _make_ollama_response()
        mock_sdk.chat = AsyncMock(return_value=mock_response)
        return OllamaClient(_client=mock_sdk)

    async def test_complete_returns_model_response(self) -> None:
        client = self._make_client()
        result = await client.complete(
            messages=[Message(role=Role.USER, content="Hello")],
            model="llama3.2",
        )
        assert isinstance(result, ModelResponse)
        assert result.content == "hello"
        assert result.token_usage.input_tokens == 10
        assert result.token_usage.output_tokens == 5
        assert result.stop_reason == "end_turn"

    async def test_complete_with_tool_call(self) -> None:
        client = self._make_client(_make_ollama_response(tool_name="search"))
        result = await client.complete(
            messages=[Message(role=Role.USER, content="search something")],
            model="llama3.2",
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "test"}

    async def test_complete_includes_system_message(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(return_value=_make_ollama_response("ok"))
        client = OllamaClient(_client=mock_sdk)
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="hello"),
        ]
        await client.complete(messages=messages, model="llama3.2")
        call_kwargs = mock_sdk.chat.call_args.kwargs
        sent_messages = call_kwargs["messages"]
        roles = [m["role"] for m in sent_messages]
        assert "system" in roles

    async def test_complete_with_tools(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(return_value=_make_ollama_response("ok"))
        client = OllamaClient(_client=mock_sdk)
        tool = ToolDefinition(
            name="search",
            description="search the web",
            parameters=[ToolParameter(name="query", type="string", description="q", required=True)],
        )
        await client.complete(
            messages=[Message(role=Role.USER, content="search something")],
            model="llama3.2",
            tools=[tool],
        )
        call_kwargs = mock_sdk.chat.call_args.kwargs
        assert "tools" in call_kwargs

    async def test_complete_connection_error_raises_model_error(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(side_effect=Exception("connection refused"))
        client = OllamaClient(_client=mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="llama3.2",
            )
        assert exc_info.value.code == "MODEL_API_ERROR"
        assert "ollama serve" in (exc_info.value.hint or "")

    async def test_complete_model_not_found_raises_model_error(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(side_effect=Exception("model not found"))
        client = OllamaClient(_client=mock_sdk)
        with pytest.raises(ModelError) as exc_info:
            await client.complete(
                messages=[Message(role=Role.USER, content="hi")],
                model="llama3.2",
            )
        assert exc_info.value.code == "MODEL_API_ERROR"
        assert "ollama pull" in (exc_info.value.hint or "")

    async def test_complete_zero_cost(self) -> None:
        client = self._make_client()
        result = await client.complete(
            messages=[Message(role=Role.USER, content="hi")],
            model="llama3.2",
        )
        assert result.token_usage.cost_usd == 0.0

    async def test_stream_yields_chunks(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(return_value=_fake_ollama_stream())
        client = OllamaClient(_client=mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="llama3.2",
        ):
            chunks.append(chunk)
        non_final = [c for c in chunks if not c.is_final]
        assert len(non_final) >= 1
        assert any(c.delta for c in non_final)
        final_chunks = [c for c in chunks if c.is_final]
        assert len(final_chunks) == 1
        assert final_chunks[0].token_usage is not None

    async def test_stream_final_chunk_has_zero_cost(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(return_value=_fake_ollama_stream())
        client = OllamaClient(_client=mock_sdk)
        chunks = []
        async for chunk in client.stream(
            messages=[Message(role=Role.USER, content="hi")],
            model="llama3.2",
        ):
            chunks.append(chunk)
        final = next(c for c in chunks if c.is_final)
        assert final.token_usage is not None
        assert final.token_usage.cost_usd == 0.0

    async def test_stream_api_error_raises_model_error(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(side_effect=Exception("stream failed"))
        client = OllamaClient(_client=mock_sdk)
        with pytest.raises(ModelError):
            async for _ in client.stream(
                messages=[Message(role=Role.USER, content="hi")],
                model="llama3.2",
            ):
                pass

    def test_is_subclass_of_model_client(self) -> None:
        assert issubclass(OllamaClient, ModelClient)

    def test_tool_result_message_uses_tool_name(self) -> None:
        tc = ToolCall(id="call-001", name="search", arguments={"q": "test"})
        tr = ToolResult(tool_call_id="call-001", output="result text")
        messages = [
            Message(role=Role.USER, content="search for me"),
            Message(role=Role.ASSISTANT, tool_calls=[tc]),
            Message(role=Role.TOOL, tool_results=[tr]),
        ]
        result = _to_ollama_messages(messages)
        tool_entry = next(m for m in result if m["role"] == "tool")
        assert "tool_name" in tool_entry
        assert "name" not in tool_entry
        assert tool_entry["tool_name"] == "search"

    def test_stop_reason_mapping(self) -> None:
        assert _ollama_stop_reason("stop") == "end_turn"
        assert _ollama_stop_reason("length") == "max_tokens"
        assert _ollama_stop_reason("tool_calls") == "tool_use"
        assert _ollama_stop_reason(None) == "stop"
        assert _ollama_stop_reason("unknown") == "stop"

    def test_custom_host_stored(self) -> None:
        mock_sdk = MagicMock()
        client = OllamaClient(host="http://myserver:11434", _client=mock_sdk)
        assert client._host == "http://myserver:11434"
