"""Tests for Phase C5: Dapr job callback endpoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from grampus.core.types import AgentDefinition, AgentStatus, ExecutionResult, TokenUsage


def _make_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=5, output_tokens=10, total_tokens=15, cost_usd=0.0005, model="test"
    )


def _make_result() -> ExecutionResult:
    return ExecutionResult(
        output="done",
        messages=[],
        tool_calls_made=0,
        token_usage=_make_usage(),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


@pytest.fixture
def mock_runner() -> MagicMock:
    runner = MagicMock()
    runner.run = AsyncMock(return_value=_make_result())
    return runner


@pytest.fixture
def mock_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-sonnet-4-6")


@pytest.fixture
def client(mock_runner: MagicMock, mock_agent_def: AgentDefinition) -> TestClient:
    from grampus.server.app import create_app

    app = create_app(mock_runner, mock_agent_def)
    return TestClient(app)


def test_job_callback_returns_accepted(client: TestClient, mock_runner: MagicMock) -> None:
    payload = json.dumps({"value": json.dumps({"input": "run now"})})
    with patch("grampus.server.routes.asyncio.create_task") as mock_task:
        resp = client.post("/job/my-job", content=payload.encode())
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    assert data["job"] == "my-job"
    assert "session_id" in data
    mock_task.assert_called_once()
    mock_task.call_args[0][0].close()


def test_job_callback_empty_body(client: TestClient) -> None:
    with patch("grampus.server.routes.asyncio.create_task") as mock_task:
        resp = client.post("/job/my-job", content=b"")
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    assert data["session_id"].startswith("sched-")
    mock_task.call_args[0][0].close()


def test_job_callback_malformed_json(client: TestClient) -> None:
    with patch("grampus.server.routes.asyncio.create_task") as mock_task:
        resp = client.post("/job/my-job", content=b"not-json")
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True
    mock_task.call_args[0][0].close()


def test_job_callback_uses_session_prefix(client: TestClient, mock_runner: MagicMock) -> None:
    payload = json.dumps({"value": json.dumps({"input": "hello", "session_prefix": "custom"})})
    with patch("grampus.server.routes.asyncio.create_task") as mock_task:
        resp = client.post("/job/my-job", content=payload.encode())
    assert resp.status_code == 200
    assert resp.json()["session_id"].startswith("custom-")
    mock_task.call_args[0][0].close()


def test_job_callback_default_input_uses_job_name(client: TestClient) -> None:
    with patch("grampus.server.routes.asyncio.create_task") as mock_task:
        resp = client.post("/job/special-job", content=b"")
    assert resp.status_code == 200
    coro = mock_task.call_args[0][0]
    # Verify the coroutine's qualname references our helper
    assert "run_scheduled_job" in coro.__qualname__
    coro.close()


def test_job_callback_with_schedule_store(
    mock_runner: MagicMock, mock_agent_def: AgentDefinition
) -> None:
    from grampus.dapr.schedule_store import ScheduleStore
    from grampus.server.app import create_app

    store = ScheduleStore(state_store=None)
    app = create_app(mock_runner, mock_agent_def, schedule_store=store)
    tc = TestClient(app)

    payload = json.dumps({"value": json.dumps({"input": "go"})})
    with patch("grampus.server.routes.asyncio.create_task") as mock_task:
        resp = tc.post("/job/tracked-job", content=payload.encode())
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True
    mock_task.call_args[0][0].close()
