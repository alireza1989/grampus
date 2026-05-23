"""Tests for the OpenAI-compatible /v1 router."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nexus.core.types import (
    AgentDefinition,
    AgentStatus,
    ExecutionResult,
    StreamChunk,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
)
from nexus.server.openai_compat import (
    OAIChatChunk,
    OAIChatRequest,
    OAIChatResponse,
    OAIMessage,
    OAIUsage,
    _extract_system_prompt,
    _extract_user_input,
    _finish_reason,
    _nexus_usage_to_oai,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model="test"
    )


def _make_result(tool_calls_made: int = 0) -> ExecutionResult:
    return ExecutionResult(
        output="Hello from agent",
        messages=[],
        tool_calls_made=tool_calls_made,
        token_usage=_make_usage(),
        duration_seconds=0.5,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


async def _default_stream(
    agent_def: AgentDefinition, user_input: str, *, session_id: str
) -> AsyncIterator[StreamEvent]:
    yield StreamEvent(event_type=StreamEventType.AGENT_START, message=agent_def.name)
    yield StreamEvent(
        event_type=StreamEventType.TOKEN,
        chunk=StreamChunk(delta="Hello", is_final=False),
    )
    yield StreamEvent(
        event_type=StreamEventType.TOOL_CALL_START,
        tool_call=ToolCall(id="tc-1", name="search"),
    )
    yield StreamEvent(
        event_type=StreamEventType.AGENT_END,
        chunk=StreamChunk(is_final=True, token_usage=_make_usage()),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runner() -> MagicMock:
    runner = MagicMock()
    runner.run = AsyncMock(return_value=_make_result())
    runner.stream = _default_stream
    return runner


@pytest.fixture
def mock_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-sonnet-4-6")


@pytest.fixture
def client(mock_runner: MagicMock, mock_agent_def: AgentDefinition) -> TestClient:
    from nexus.server.app import create_app

    app = create_app(mock_runner, mock_agent_def)
    return TestClient(app)


# ---------------------------------------------------------------------------
# TestOAIModels — Pydantic model shapes and helper functions
# ---------------------------------------------------------------------------


class TestOAIModels:
    def test_oai_chat_request_defaults(self) -> None:
        req = OAIChatRequest(
            model="gpt-4",
            messages=[OAIMessage(role="user", content="hi")],
        )
        assert req.stream is False
        assert req.stream_options is None

    def test_oai_chat_request_extra_fields_allowed(self) -> None:
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "unknown_future_field": True,
            "store": True,
        }
        req = OAIChatRequest(**data)
        assert req.model == "gpt-4"

    def test_oai_chat_request_max_completion_tokens_accepted(self) -> None:
        req = OAIChatRequest(
            model="gpt-4",
            messages=[OAIMessage(role="user", content="hi")],
            max_completion_tokens=100,
        )
        assert req.max_completion_tokens == 100

    def test_oai_chat_response_has_system_fingerprint(self) -> None:
        resp = OAIChatResponse(
            id="chatcmpl-abc",
            created=1234,
            model="gpt-4",
            choices=[],
            usage=OAIUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        assert resp.system_fingerprint == "nexus-v0.1"

    def test_oai_chat_chunk_usage_field_optional(self) -> None:
        chunk = OAIChatChunk(
            id="chatcmpl-abc",
            created=1234,
            model="gpt-4",
            choices=[],
        )
        assert chunk.usage is None

    def test_extract_user_input_returns_last_user_message(self) -> None:
        messages = [
            OAIMessage(role="system", content="You are helpful"),
            OAIMessage(role="user", content="First"),
            OAIMessage(role="assistant", content="Reply"),
            OAIMessage(role="user", content="Last"),
        ]
        assert _extract_user_input(messages) == "Last"

    def test_extract_user_input_raises_when_no_user_message(self) -> None:
        messages = [OAIMessage(role="system", content="sys")]
        with pytest.raises(ValueError, match="No user message"):
            _extract_user_input(messages)

    def test_extract_system_prompt_returns_first_system(self) -> None:
        messages = [
            OAIMessage(role="system", content="Be concise"),
            OAIMessage(role="system", content="Second system"),
            OAIMessage(role="user", content="hi"),
        ]
        assert _extract_system_prompt(messages) == "Be concise"

    def test_extract_system_prompt_returns_none_when_absent(self) -> None:
        messages = [OAIMessage(role="user", content="hi")]
        assert _extract_system_prompt(messages) is None

    def test_nexus_usage_to_oai_mapping(self) -> None:
        usage = TokenUsage(
            input_tokens=5, output_tokens=15, total_tokens=20, cost_usd=0.001, model="m"
        )
        oai = _nexus_usage_to_oai(usage)
        assert oai.prompt_tokens == 5
        assert oai.completion_tokens == 15
        assert oai.total_tokens == 20

    def test_finish_reason_stop_when_no_tools(self) -> None:
        assert _finish_reason(0) == "stop"

    def test_finish_reason_tool_calls_when_tools_used(self) -> None:
        assert _finish_reason(3) == "tool_calls"


# ---------------------------------------------------------------------------
# TestModelsEndpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_get_models_returns_200(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_get_models_returns_list_object(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.json()["object"] == "list"

    def test_get_models_contains_agent_name(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert "test-agent" in ids


# ---------------------------------------------------------------------------
# TestChatCompletionsNonStreaming
# ---------------------------------------------------------------------------


class TestChatCompletionsNonStreaming:
    def _post(self, client: TestClient, extra: dict | None = None) -> object:
        payload: dict = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        if extra:
            payload.update(extra)
        return client.post("/v1/chat/completions", json=payload)

    def test_chat_completions_returns_200(self, client: TestClient) -> None:
        assert self._post(client).status_code == 200

    def test_chat_completions_object_is_chat_completion(self, client: TestClient) -> None:
        assert self._post(client).json()["object"] == "chat.completion"

    def test_chat_completions_choices_has_one_item(self, client: TestClient) -> None:
        assert len(self._post(client).json()["choices"]) == 1

    def test_chat_completions_choice_role_is_assistant(self, client: TestClient) -> None:
        choice = self._post(client).json()["choices"][0]
        assert choice["message"]["role"] == "assistant"

    def test_chat_completions_content_matches_agent_output(self, client: TestClient) -> None:
        choice = self._post(client).json()["choices"][0]
        assert choice["message"]["content"] == "Hello from agent"

    def test_chat_completions_model_echoed_from_request(self, client: TestClient) -> None:
        assert self._post(client).json()["model"] == "gpt-4"

    def test_chat_completions_usage_populated(self, client: TestClient) -> None:
        usage = self._post(client).json()["usage"]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_chat_completions_has_id_and_created_fields(self, client: TestClient) -> None:
        body = self._post(client).json()
        assert body["id"].startswith("chatcmpl-")
        assert isinstance(body["created"], int)

    def test_chat_completions_has_system_fingerprint(self, client: TestClient) -> None:
        assert self._post(client).json()["system_fingerprint"] == "nexus-v0.1"

    def test_chat_completions_system_message_applied(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from nexus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        c = TestClient(app)
        c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "Be terse"},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        called_def: AgentDefinition = mock_runner.run.call_args[0][0]
        assert called_def.system_prompt == "Be terse"

    def test_chat_completions_temperature_override_applied(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from nexus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        c = TestClient(app)
        c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
            },
        )
        called_def: AgentDefinition = mock_runner.run.call_args[0][0]
        assert called_def.temperature == pytest.approx(0.7)

    def test_chat_completions_no_user_message_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "system", "content": "sys"}]},
        )
        assert resp.status_code == 400

    def test_chat_completions_bearer_token_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status_code == 200

    def test_chat_completions_no_auth_header_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestChatCompletionsStreaming
# ---------------------------------------------------------------------------


def _stream_lines(client: TestClient, extra: dict | None = None) -> list[str]:
    payload: dict = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    if extra:
        payload.update(extra)
    resp = client.post("/v1/chat/completions", json=payload)
    return resp.text.splitlines()


def _data_chunks(lines: list[str]) -> list[dict]:
    result = []
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            result.append(json.loads(line[6:]))
    return result


class TestChatCompletionsStreaming:
    def test_stream_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

    def test_stream_content_type_is_event_stream(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_first_chunk_has_role_assistant(self, client: TestClient) -> None:
        lines = _stream_lines(client)
        chunks = _data_chunks(lines)
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

    def test_stream_token_chunks_have_content(self, client: TestClient) -> None:
        lines = _stream_lines(client)
        chunks = _data_chunks(lines)
        contents = [
            c["choices"][0]["delta"].get("content")
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("content")
        ]
        assert "Hello" in contents

    def test_stream_final_chunk_has_finish_reason_stop(self, client: TestClient) -> None:
        lines = _stream_lines(client)
        chunks = _data_chunks(lines)
        finish_chunks = [
            c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason") == "stop"
        ]
        assert len(finish_chunks) >= 1

    def test_stream_ends_with_done_sentinel(self, client: TestClient) -> None:
        lines = _stream_lines(client)
        non_empty = [ln for ln in lines if ln.strip()]
        assert non_empty[-1] == "data: [DONE]"

    def test_stream_each_data_line_is_valid_json(self, client: TestClient) -> None:
        lines = _stream_lines(client)
        data_lines = [ln[6:] for ln in lines if ln.startswith("data: ") and ln != "data: [DONE]"]
        assert len(data_lines) > 0
        for line in data_lines:
            parsed = json.loads(line)
            assert "id" in parsed

    def test_stream_all_chunks_share_same_id(self, client: TestClient) -> None:
        lines = _stream_lines(client)
        chunks = _data_chunks(lines)
        ids = {c["id"] for c in chunks}
        assert len(ids) == 1

    def test_stream_tool_events_not_in_output(self, client: TestClient) -> None:
        lines = _stream_lines(client)
        chunks = _data_chunks(lines)
        # TOOL_CALL_START from _default_stream must not produce extra SSE lines —
        # only role chunk + token chunk + finish chunk (+ optional usage chunk)
        # The finish chunk has choices=[...] with finish_reason=stop
        finish_count = sum(
            1 for c in chunks if c["choices"] and c["choices"][0].get("finish_reason") == "stop"
        )
        assert finish_count == 1

    def test_stream_include_usage_emits_usage_chunk(self, client: TestClient) -> None:
        lines = _stream_lines(client, extra={"stream_options": {"include_usage": True}})
        chunks = _data_chunks(lines)
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        assert len(usage_chunks) == 1

    def test_stream_no_include_usage_omits_usage_chunk(self, client: TestClient) -> None:
        lines = _stream_lines(client)
        chunks = _data_chunks(lines)
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        assert len(usage_chunks) == 0

    def test_stream_usage_chunk_has_empty_choices(self, client: TestClient) -> None:
        lines = _stream_lines(client, extra={"stream_options": {"include_usage": True}})
        chunks = _data_chunks(lines)
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        assert usage_chunks[0]["choices"] == []


# ---------------------------------------------------------------------------
# TestCORSHeaders
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    def test_cors_allows_all_origins(self, client: TestClient) -> None:
        resp = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# TestOpenAISDKCompat
# ---------------------------------------------------------------------------


class TestOpenAISDKCompat:
    def test_full_request_shape_matches_sdk_output(self, client: TestClient) -> None:
        # POST a raw dict matching what the OpenAI SDK sends
        sdk_payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"},
            ],
            "temperature": 0.5,
            "max_tokens": 256,
            "stream": False,
            "seed": 42,
            "logprobs": False,
            "store": True,
        }
        resp = client.post(
            "/v1/chat/completions",
            json=sdk_payload,
            headers={"Authorization": "Bearer sk-test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Validate shape matches what OpenAI SDK expects
        assert body["object"] == "chat.completion"
        assert isinstance(body["id"], str) and body["id"].startswith("chatcmpl-")
        assert isinstance(body["created"], int)
        assert body["model"] == "gpt-4o"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)
        assert body["choices"][0]["finish_reason"] in ("stop", "tool_calls")
        assert body["usage"]["total_tokens"] > 0
        assert "system_fingerprint" in body
