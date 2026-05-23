"""Tests for nexus.server.routes endpoint handlers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nexus.core.errors import OrchestrationError
from nexus.core.types import (
    AgentDefinition,
    AgentStatus,
    ExecutionResult,
    StreamChunk,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model="test"
    )


def _make_result() -> ExecutionResult:
    return ExecutionResult(
        output="Hello from agent",
        messages=[],
        tool_calls_made=2,
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
        event_type=StreamEventType.AGENT_END,
        chunk=StreamChunk(is_final=True, token_usage=_make_usage()),
    )


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
def app(mock_runner: MagicMock, mock_agent_def: AgentDefinition) -> object:
    from nexus.server.app import create_app

    return create_app(mock_runner, mock_agent_def)


@pytest.fixture
def client(app: object) -> TestClient:
    return TestClient(app)  # type: ignore[arg-type]


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    mm = MagicMock()
    recall_result = MagicMock()
    recall_result.episodic = []
    recall_result.semantic = []
    mm.recall = AsyncMock(return_value=recall_result)
    mm.forget = AsyncMock()
    return mm


@pytest.fixture
def app_with_memory(
    mock_runner: MagicMock,
    mock_agent_def: AgentDefinition,
    mock_memory_manager: MagicMock,
) -> object:
    from nexus.server.app import create_app

    return create_app(mock_runner, mock_agent_def, memory_manager=mock_memory_manager)


@pytest.fixture
def client_with_memory(app_with_memory: object) -> TestClient:
    return TestClient(app_with_memory)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_returns_agent_name(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json()["agent_name"] == "test-agent"

    def test_health_returns_version(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# /run endpoint
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    def test_run_returns_200(self, client: TestClient) -> None:
        resp = client.post("/run", json={"input": "hello"})
        assert resp.status_code == 200

    def test_run_returns_output(self, client: TestClient) -> None:
        resp = client.post("/run", json={"input": "hello"})
        assert resp.json()["output"] == "Hello from agent"

    def test_run_generates_session_id_when_omitted(self, client: TestClient) -> None:
        resp = client.post("/run", json={"input": "hello"})
        sid = resp.json()["session_id"]
        assert sid.startswith("api-")
        assert len(sid) > 4

    def test_run_uses_provided_session_id(self, client: TestClient) -> None:
        resp = client.post("/run", json={"input": "hello", "session_id": "my-session"})
        assert resp.json()["session_id"] == "my-session"

    def test_run_temperature_override_applied(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from nexus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        c = TestClient(app)
        c.post("/run", json={"input": "hi", "temperature": 0.9})
        call_args = mock_runner.run.call_args
        called_def: AgentDefinition = call_args[0][0]
        assert called_def.temperature == pytest.approx(0.9)

    def test_run_max_iterations_override_applied(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from nexus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        c = TestClient(app)
        c.post("/run", json={"input": "hi", "max_iterations": 3})
        call_args = mock_runner.run.call_args
        called_def: AgentDefinition = call_args[0][0]
        assert called_def.max_iterations == 3

    def test_run_nexus_error_returns_400(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        mock_runner.run = AsyncMock(
            side_effect=OrchestrationError("limit hit", code="MAX_ITER", hint="increase it")
        )
        from nexus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post("/run", json={"input": "hi"})
        assert resp.status_code == 400

    def test_run_nexus_error_includes_code(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        mock_runner.run = AsyncMock(
            side_effect=OrchestrationError("limit hit", code="MAX_ITER", hint="increase it")
        )
        from nexus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post("/run", json={"input": "hi"})
        assert resp.json()["code"] == "MAX_ITER"

    def test_run_nexus_error_includes_hint(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        mock_runner.run = AsyncMock(
            side_effect=OrchestrationError("limit hit", code="MAX_ITER", hint="increase it")
        )
        from nexus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post("/run", json={"input": "hi"})
        assert resp.json()["hint"] == "increase it"


# ---------------------------------------------------------------------------
# /stream endpoint
# ---------------------------------------------------------------------------


class TestStreamEndpoint:
    def test_stream_returns_200(self, client: TestClient) -> None:
        resp = client.post("/stream", json={"input": "hello"})
        assert resp.status_code == 200

    def test_stream_content_type_is_event_stream(self, client: TestClient) -> None:
        resp = client.post("/stream", json={"input": "hello"})
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_yields_token_events(self, client: TestClient) -> None:
        resp = client.post("/stream", json={"input": "hello"})
        lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
        events = [json.loads(ln[6:]) for ln in lines]
        token_events = [e for e in events if e["event_type"] == "token"]
        assert len(token_events) >= 1
        assert any(e["delta"] == "Hello" for e in token_events)

    def test_stream_yields_agent_end_event(self, client: TestClient) -> None:
        resp = client.post("/stream", json={"input": "hello"})
        lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
        events = [json.loads(ln[6:]) for ln in lines]
        end_events = [e for e in events if e["event_type"] == "agent_end"]
        assert len(end_events) == 1

    def test_stream_response_is_valid_json_per_line(self, client: TestClient) -> None:
        resp = client.post("/stream", json={"input": "hello"})
        data_lines = [ln[6:] for ln in resp.text.splitlines() if ln.startswith("data: ")]
        assert len(data_lines) > 0
        for line in data_lines:
            parsed = json.loads(line)
            assert "event_type" in parsed


# ---------------------------------------------------------------------------
# /memory endpoints
# ---------------------------------------------------------------------------


class TestMemoryEndpoint:
    def test_memory_recall_returns_200(self, client_with_memory: TestClient) -> None:
        resp = client_with_memory.post("/memory/recall", json={"query": "what happened"})
        assert resp.status_code == 200

    def test_memory_recall_without_manager_returns_404(self, client: TestClient) -> None:
        resp = client.post("/memory/recall", json={"query": "what happened"})
        assert resp.status_code == 404

    def test_memory_recall_response_shape(self, client_with_memory: TestClient) -> None:
        resp = client_with_memory.post("/memory/recall", json={"query": "q"})
        body = resp.json()
        assert "episodic" in body
        assert "semantic" in body
        assert body["query"] == "q"

    def test_memory_delete_calls_forget(
        self, client_with_memory: TestClient, mock_memory_manager: MagicMock
    ) -> None:
        resp = client_with_memory.delete("/memory/rec-123?memory_type=episodic")
        assert resp.status_code == 200
        mock_memory_manager.forget.assert_awaited_once()

    def test_memory_delete_without_manager_returns_404(self, client: TestClient) -> None:
        resp = client.delete("/memory/rec-123?memory_type=episodic")
        assert resp.status_code == 404

    def test_memory_delete_response_shape(self, client_with_memory: TestClient) -> None:
        resp = client_with_memory.delete("/memory/rec-abc?memory_type=semantic")
        body = resp.json()
        assert body["deleted"] is True
        assert body["record_id"] == "rec-abc"
