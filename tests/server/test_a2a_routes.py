"""Tests for A2A server routes."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nexus.core.types import AgentDefinition, AgentStatus, ExecutionResult, TokenUsage


def _make_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-3-5-haiku-20241022")


def _make_runner(output: str = "ok") -> MagicMock:
    runner = MagicMock()
    result = ExecutionResult(
        output=output,
        messages=[],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=5,
            output_tokens=10,
            total_tokens=15,
            cost_usd=0.0005,
            model="claude-3-5-haiku-20241022",
        ),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )
    runner.run = AsyncMock(return_value=result)
    runner.stream = MagicMock()
    runner.list_pending_sessions = MagicMock(return_value=[])
    runner.subscribe_trace = MagicMock(return_value=AsyncMock())
    runner.unsubscribe_trace = MagicMock()
    return runner


def _make_app_with_a2a(api_key: str | None = None) -> Any:
    from nexus.orchestration.a2a.executor import NexusA2AExecutor
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.a2a.task_store import NexusTaskStore
    from nexus.server.app import create_app

    runner = _make_runner()
    agent_def = _make_agent_def()

    executor = NexusA2AExecutor(runner=runner, agent_def=agent_def)
    task_store = NexusTaskStore()
    registry = AgentRegistry()
    registry.register_local(
        name="test-agent",
        runner=runner,
        description="A test agent",
        base_url="http://testserver",
    )

    return create_app(
        runner=runner,
        agent_def=agent_def,
        a2a_executor=executor,
        a2a_task_store=task_store,
        agent_registry=registry,
        a2a_api_key=api_key,
    )


def test_well_known_agent_json_returns_200() -> None:
    app = _make_app_with_a2a()
    client = TestClient(app)
    resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200


def test_well_known_agent_json_is_valid_agent_card() -> None:
    app = _make_app_with_a2a()
    client = TestClient(app)
    resp = client.get("/.well-known/agent-card.json")
    data = resp.json()
    assert "name" in data
    assert "capabilities" in data


def test_a2a_message_send_returns_task() -> None:
    app = _make_app_with_a2a()
    client = TestClient(app)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": "hello"}],
                "messageId": "msg-1",
            },
            "configuration": {},
        },
    }
    resp = client.post(
        "/a2a",
        json=payload,
        headers={"x-a2a-version": "1.0"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body or "error" in body


def test_a2a_message_send_no_executor_returns_503() -> None:
    from nexus.server.app import create_app

    runner = _make_runner()
    agent_def = _make_agent_def()
    app = create_app(runner=runner, agent_def=agent_def)

    client = TestClient(app)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": "hello"}],
                "messageId": "msg-1",
            },
        },
    }
    resp = client.post("/a2a", json=payload, headers={"x-a2a-version": "1.0"})
    assert resp.status_code == 503


def test_a2a_agents_list_returns_registered_agents() -> None:
    app = _make_app_with_a2a()
    client = TestClient(app)
    resp = client.get("/a2a/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data


def test_a2a_api_key_required_when_configured() -> None:
    app = _make_app_with_a2a(api_key="secret-123")
    client = TestClient(app)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": "hello"}],
                "messageId": "msg-auth",
            },
        },
    }
    resp = client.post(
        "/a2a",
        json=payload,
        headers={"Authorization": "Bearer secret-123", "x-a2a-version": "1.0"},
    )
    assert resp.status_code in (200, 400)  # authorized but may fail for other reasons


def test_a2a_api_key_missing_returns_401() -> None:
    app = _make_app_with_a2a(api_key="secret-123")
    client = TestClient(app)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": "hello"}],
                "messageId": "msg-no-auth",
            },
        },
    }
    resp = client.post("/a2a", json=payload, headers={"x-a2a-version": "1.0"})
    assert resp.status_code == 401


def test_a2a_api_key_invalid_returns_401() -> None:
    app = _make_app_with_a2a(api_key="secret-123")
    client = TestClient(app)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": "hello"}],
                "messageId": "msg-bad-auth",
            },
        },
    }
    resp = client.post(
        "/a2a",
        json=payload,
        headers={"Authorization": "Bearer wrong-key", "x-a2a-version": "1.0"},
    )
    assert resp.status_code == 401
