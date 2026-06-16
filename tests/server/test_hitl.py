"""Tests for Human-in-the-loop endpoints and models."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from grampus.core.errors import OrchestrationError
from grampus.core.types import (
    AgentDefinition,
    AgentState,
    AgentStatus,
    ExecutionResult,
    Message,
    Role,
    TokenUsage,
)
from grampus.server.models import (
    PendingSession,
    PendingSessionsResponse,
    ResumeRequest,
    ResumeResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_usage() -> TokenUsage:
    return TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15, cost_usd=0.0, model="test")


def _make_state(
    session_id: str = "sess-abc",
    agent_id: str = "test-agent",
    status: AgentStatus = AgentStatus.WAITING_FOR_HUMAN,
    messages: list[Message] | None = None,
) -> AgentState:
    if messages is None:
        messages = [Message(role=Role.USER, content="help me")]
    return AgentState(
        agent_id=agent_id,
        session_id=session_id,
        messages=messages,
        status=status,
        updated_at=datetime.now(UTC),
    )


def _make_result(
    status: AgentStatus = AgentStatus.COMPLETED,
    output: str | None = "Done",
) -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[],
        tool_calls_made=0,
        token_usage=_make_usage(),
        duration_seconds=0.1,
        steps_taken=1,
        status=status,
    )


def _make_app(runner: MagicMock) -> object:
    from grampus.server.app import create_app

    agent_def = AgentDefinition(name="test-agent", model="claude-sonnet-4-6")
    return create_app(runner, agent_def)


def _make_runner(
    *,
    pending: list[str] | None = None,
    state: AgentState | None = None,
    state_error: Exception | None = None,
    resume_result: ExecutionResult | None = None,
) -> MagicMock:
    runner = MagicMock()
    runner.list_pending_sessions = MagicMock(return_value=pending or [])
    if state_error is not None:
        runner.get_state = AsyncMock(side_effect=state_error)
    else:
        runner.get_state = AsyncMock(return_value=state)
    runner.resume = AsyncMock(return_value=resume_result or _make_result())
    return runner


# ---------------------------------------------------------------------------
# GET /agents/pending
# ---------------------------------------------------------------------------


class TestPendingSessions:
    def test_pending_sessions_empty(self) -> None:
        runner = _make_runner(pending=[])
        client = TestClient(_make_app(runner))  # type: ignore[arg-type]
        resp = client.get("/agents/pending")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sessions"] == []
        assert body["count"] == 0

    def test_pending_sessions_with_session(self) -> None:
        state = _make_state(
            session_id="sess-abc",
            messages=[Message(role=Role.USER, content="help me")],
        )
        runner = _make_runner(pending=["sess-abc"], state=state)
        client = TestClient(_make_app(runner))  # type: ignore[arg-type]
        resp = client.get("/agents/pending")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["sessions"][0]["session_id"] == "sess-abc"
        assert body["sessions"][0]["last_message"] == "help me"


# ---------------------------------------------------------------------------
# GET /agents/{session_id}/state
# ---------------------------------------------------------------------------


class TestAgentStateEndpoint:
    def test_agent_state_endpoint_found(self) -> None:
        state = _make_state(
            session_id="sess-123",
            messages=[
                Message(role=Role.USER, content="question"),
                Message(role=Role.ASSISTANT, content="answer"),
            ],
        )
        runner = _make_runner(state=state)
        client = TestClient(_make_app(runner))  # type: ignore[arg-type]
        resp = client.get("/agents/sess-123/state")
        assert resp.status_code == 200
        body = resp.json()
        assert body["message_count"] == 2
        assert body["status"] == "waiting_for_human"

    def test_agent_state_endpoint_not_found(self) -> None:
        runner = _make_runner(state_error=OrchestrationError("not found", code="NO_STATE_FOUND"))
        client = TestClient(_make_app(runner), raise_server_exceptions=False)  # type: ignore[arg-type]
        resp = client.get("/agents/missing/state")
        assert resp.status_code == 404

    def test_agent_state_excludes_system_messages(self) -> None:
        state = _make_state(
            session_id="sess-sys",
            messages=[
                Message(role=Role.SYSTEM, content="You are helpful."),
                Message(role=Role.USER, content="hello"),
                Message(role=Role.ASSISTANT, content="hi there"),
            ],
        )
        runner = _make_runner(state=state)
        client = TestClient(_make_app(runner))  # type: ignore[arg-type]
        resp = client.get("/agents/sess-sys/state")
        assert resp.status_code == 200
        body = resp.json()
        roles = [m["role"] for m in body["messages"]]
        assert "system" not in roles
        assert body["message_count"] == 2


# ---------------------------------------------------------------------------
# POST /agents/{session_id}/resume
# ---------------------------------------------------------------------------


class TestResumeEndpoint:
    def test_resume_endpoint_success(self) -> None:
        runner = _make_runner(
            resume_result=_make_result(status=AgentStatus.COMPLETED, output="Done")
        )
        client = TestClient(_make_app(runner))  # type: ignore[arg-type]
        resp = client.post("/agents/sess-1/resume", json={"input": "proceed"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["output"] == "Done"
        assert body["still_waiting"] is False

    def test_resume_endpoint_still_waiting(self) -> None:
        runner = _make_runner(
            resume_result=_make_result(status=AgentStatus.WAITING_FOR_HUMAN, output=None)
        )
        client = TestClient(_make_app(runner))  # type: ignore[arg-type]
        resp = client.post("/agents/sess-1/resume", json={"input": "ok"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["still_waiting"] is True


# ---------------------------------------------------------------------------
# UI endpoints
# ---------------------------------------------------------------------------


class TestUIEndpoints:
    def test_ui_endpoint_returns_html(self) -> None:
        runner = _make_runner()
        client = TestClient(_make_app(runner))  # type: ignore[arg-type]
        resp = client.get("/ui")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Nexus" in resp.text

    async def test_ui_events_returns_sse(self) -> None:
        # Drive the ASGI app directly so we control disconnect timing.
        # httpx.ASGITransport deadlocks on infinite SSE (waits for response_complete
        # before sending disconnect, but the generator waits for disconnect first).
        runner = _make_runner()
        app = _make_app(runner)

        captured_status: list[int] = []
        captured_headers: list[tuple[bytes, bytes]] = []
        response_started = asyncio.Event()

        async def receive() -> dict[str, Any]:
            # Block until headers are sent, then signal disconnect.
            await response_started.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                captured_status.append(message["status"])
                captured_headers.extend(message.get("headers", []))
                response_started.set()

        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "headers": [],
            "path": "/ui/events",
            "raw_path": b"/ui/events",
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 12345),
        }

        await app(scope, receive, send)

        assert captured_status == [200]
        cts = [v.decode() for k, v in captured_headers if k == b"content-type"]
        assert any("text/event-stream" in ct for ct in cts)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_resume_request_model_validates(self) -> None:
        req = ResumeRequest(input="hello")
        assert req.input == "hello"

    def test_resume_request_requires_input(self) -> None:
        with pytest.raises(ValidationError):
            ResumeRequest()  # type: ignore[call-arg]

    def test_pending_session_model(self) -> None:
        ps = PendingSession(
            session_id="s1",
            agent_id="a1",
            last_message="hi",
            waiting_since="2025-01-01T00:00:00",
        )
        dumped = ps.model_dump()
        restored = PendingSession(**dumped)
        assert restored.session_id == "s1"

    def test_resume_response_still_waiting_true(self) -> None:
        rr = ResumeResponse(
            session_id="s",
            output=None,
            status="waiting_for_human",
            steps_taken=1,
            still_waiting=True,
        )
        assert rr.still_waiting is True

    def test_pending_sessions_response_shape(self) -> None:
        resp = PendingSessionsResponse(sessions=[], count=0)
        assert resp.count == 0
        assert resp.sessions == []
