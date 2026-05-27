"""Tests for Phase C3 — execution trace viewer endpoints and runner pub/sub."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nexus.core.types import AgentDefinition
from nexus.observability.events import AgentEvent, EventLog, EventType
from nexus.server.trace_ui import TRACE_HTML

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-sonnet-4-6")


def _make_runner() -> MagicMock:
    runner = MagicMock()
    runner._state_store = None
    runner.subscribe_trace = MagicMock(return_value=asyncio.Queue())
    runner.unsubscribe_trace = MagicMock()
    return runner


def _make_app(runner: Any, agent_def: AgentDefinition | None = None) -> Any:
    from nexus.server.app import create_app

    return create_app(runner, agent_def or _make_agent_def())


# ---------------------------------------------------------------------------
# Bare runner fixture (AgentRunner unit tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def bare_runner() -> Any:
    from nexus.orchestration.runner import AgentRunner
    from nexus.tools.executor import ToolExecutor

    return AgentRunner(model_client=MagicMock(), tool_executor=MagicMock(spec=ToolExecutor))


# ---------------------------------------------------------------------------
# GET /trace — serve the trace viewer page
# ---------------------------------------------------------------------------


class TestTraceUIEndpoint:
    def test_trace_ui_endpoint_returns_html(self) -> None:
        client = TestClient(_make_app(_make_runner()))
        resp = client.get("/trace")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Nexus" in resp.text


# ---------------------------------------------------------------------------
# GET /trace/{session_id}/history
# ---------------------------------------------------------------------------


class TestTraceHistoryEndpoint:
    def test_trace_history_no_state_store(self) -> None:
        client = TestClient(_make_app(_make_runner()))
        resp = client.get("/trace/sess-1/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["events"] == []
        assert body["count"] == 0
        assert body["session_id"] == "sess-1"

    def test_trace_history_with_events(self) -> None:
        log = EventLog(agent_id="test-agent", session_id="sess-1", state_store=None)
        log._events = [
            AgentEvent(
                event_type=EventType.AGENT_STARTED,
                agent_id="test-agent",
                session_id="sess-1",
                sequence_number=0,
                payload={},
            ),
            AgentEvent(
                event_type=EventType.LLM_CALLED,
                agent_id="test-agent",
                session_id="sess-1",
                sequence_number=1,
                payload={},
            ),
            AgentEvent(
                event_type=EventType.AGENT_COMPLETED,
                agent_id="test-agent",
                session_id="sess-1",
                sequence_number=2,
                payload={},
            ),
        ]
        log._next_seq = 3

        with patch("nexus.server.routes.EventLog") as mock_cls:
            mock_cls.open = AsyncMock(return_value=log)
            app = _make_app(_make_runner())
            client = TestClient(app)
            resp = client.get("/trace/sess-1/history")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["events"]) == 3


# ---------------------------------------------------------------------------
# GET /trace/{session_id}/stream — SSE
# ---------------------------------------------------------------------------


class TestTraceStreamEndpoint:
    async def test_trace_stream_returns_sse_content_type(self) -> None:
        runner = _make_runner()
        app = _make_app(runner)

        captured_status: list[int] = []
        captured_headers: list[tuple[bytes, bytes]] = []
        response_started = asyncio.Event()

        async def receive() -> dict[str, Any]:
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
            "path": "/trace/sess-1/stream",
            "raw_path": b"/trace/sess-1/stream",
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
# AgentRunner.subscribe_trace / unsubscribe_trace / _publish_trace
# ---------------------------------------------------------------------------


class TestRunnerTracePubSub:
    def test_subscribe_trace_returns_queue(self, bare_runner: Any) -> None:
        q = bare_runner.subscribe_trace("sess-1")
        assert isinstance(q, asyncio.Queue)

    def test_unsubscribe_trace_removes_queue(self, bare_runner: Any) -> None:
        q = bare_runner.subscribe_trace("sess-1")
        bare_runner.unsubscribe_trace("sess-1", q)
        assert bare_runner._trace_queues["sess-1"] == []

    def test_publish_trace_puts_on_queue(self, bare_runner: Any) -> None:
        q = bare_runner.subscribe_trace("sess-1")
        mock_event = MagicMock(spec=AgentEvent)
        bare_runner._publish_trace("sess-1", mock_event)
        assert q.get_nowait() is mock_event

    def test_publish_trace_sends_none_sentinel(self, bare_runner: Any) -> None:
        q = bare_runner.subscribe_trace("sess-1")
        bare_runner._publish_trace("sess-1", None)
        assert q.get_nowait() is None

    def test_publish_trace_no_subscriber_no_error(self, bare_runner: Any) -> None:
        bare_runner._publish_trace("unknown-session", None)  # must not raise

    def test_publish_trace_full_queue_drops_silently(self, bare_runner: Any) -> None:
        q: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=1)
        bare_runner._trace_queues["sess-fill"].append(q)
        q.put_nowait(None)  # fill the queue
        bare_runner._publish_trace("sess-fill", None)  # must not raise


# ---------------------------------------------------------------------------
# TRACE_HTML content
# ---------------------------------------------------------------------------


class TestTraceHTMLContent:
    def test_trace_html_contains_session_input(self) -> None:
        assert "session" in TRACE_HTML

    def test_trace_html_contains_watch_button(self) -> None:
        assert "Watch" in TRACE_HTML


# ---------------------------------------------------------------------------
# EventLog.open with None store returns empty log
# ---------------------------------------------------------------------------


async def test_event_log_open_with_none_store_empty() -> None:
    log = await EventLog.open(agent_id="a", session_id="s", state_store=None)
    events = await log.replay()
    assert events == []
    assert log._next_seq == 0
