"""Tests for A2AAgentClient."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from a2a.types.a2a_pb2 import AgentCard, AgentInterface, AgentCapabilities, Task, TaskState


def _make_agent_card_dict(url: str = "http://agent.test") -> dict[str, Any]:
    """Build a minimal AgentCard JSON dict compatible with a2a-sdk v1.1."""
    return {
        "name": "remote-agent",
        "description": "A remote agent",
        "supportedInterfaces": [
            {
                "url": url,
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ],
        "version": "1.0.0",
        "capabilities": {"streaming": True},
    }


def _make_mock_http_client(
    get_response: dict[str, Any] | None = None,
    post_response: dict[str, Any] | None = None,
    status_code: int = 200,
) -> httpx.AsyncClient:
    client = MagicMock(spec=httpx.AsyncClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    if get_response is not None:
        client.get = AsyncMock(return_value=httpx.Response(status_code, json=get_response))
    if post_response is not None:
        client.post = AsyncMock(return_value=httpx.Response(status_code, json=post_response))
    return client


async def test_fetch_agent_card_returns_parsed_card() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient

    card_dict = _make_agent_card_dict("http://agent.test")
    mock_http = _make_mock_http_client(get_response=card_dict)

    client = A2AAgentClient(
        base_url="http://agent.test",
        _http_client=mock_http,
    )
    card = await client.fetch_agent_card()

    assert isinstance(card, AgentCard)
    assert card.name == "remote-agent"


async def test_send_message_returns_response() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient

    # JSON-RPC response format: completed task
    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "kind": "task",
            "id": "task-123",
            "contextId": "ctx-1",
            "status": {"state": "TASK_STATE_COMPLETED"},
        },
    }
    mock_http = _make_mock_http_client(
        get_response=_make_agent_card_dict(),
        post_response=rpc_response,
    )

    client = A2AAgentClient(base_url="http://agent.test", _http_client=mock_http)
    response = await client.send_message("hello")

    assert response is not None


async def test_send_message_sets_bearer_token_when_api_key() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient

    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "kind": "task",
            "id": "task-123",
            "contextId": "ctx-1",
            "status": {"state": "TASK_STATE_COMPLETED"},
        },
    }
    mock_http = _make_mock_http_client(
        get_response=_make_agent_card_dict(),
        post_response=rpc_response,
    )

    client = A2AAgentClient(
        base_url="http://agent.test",
        api_key="secret-key",
        _http_client=mock_http,
    )
    await client.send_message("hello with auth")

    # Verify Authorization header was set on the post call
    post_call = mock_http.post.call_args
    assert post_call is not None
    headers = post_call.kwargs.get("headers", {})
    assert headers.get("Authorization") == "Bearer secret-key"


async def test_send_message_raises_orchestration_error_on_http_error() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient
    from nexus.core.errors import OrchestrationError

    mock_http = _make_mock_http_client(
        get_response=_make_agent_card_dict(),
        post_response={"error": "oops"},
        status_code=500,
    )

    client = A2AAgentClient(base_url="http://agent.test", _http_client=mock_http)

    with pytest.raises(OrchestrationError) as exc_info:
        await client.send_message("hello")

    assert exc_info.value.code == "A2A_CLIENT_ERROR"


async def test_get_task_returns_task() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient

    task_rpc = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "id": "task-abc",
            "contextId": "ctx-1",
            "status": {"state": "TASK_STATE_COMPLETED"},
        },
    }
    mock_http = _make_mock_http_client(
        get_response=_make_agent_card_dict(),
        post_response=task_rpc,
    )

    client = A2AAgentClient(base_url="http://agent.test", _http_client=mock_http)
    task = await client.get_task("task-abc")

    assert isinstance(task, Task)


async def test_cancel_task_calls_tasks_cancel() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient

    cancel_rpc = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "id": "task-xyz",
            "contextId": "ctx-1",
            "status": {"state": "TASK_STATE_CANCELED"},
        },
    }
    mock_http = _make_mock_http_client(
        get_response=_make_agent_card_dict(),
        post_response=cancel_rpc,
    )

    client = A2AAgentClient(base_url="http://agent.test", _http_client=mock_http)
    await client.cancel_task("task-xyz")  # Should not raise


async def test_wait_for_completion_polls_until_terminal() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient

    working_rpc = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "id": "task-w",
            "contextId": "ctx-1",
            "status": {"state": "TASK_STATE_WORKING"},
        },
    }
    done_rpc = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "id": "task-w",
            "contextId": "ctx-1",
            "status": {"state": "TASK_STATE_COMPLETED"},
        },
    }

    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=httpx.Response(200, json=_make_agent_card_dict()))
    mock_http.post = AsyncMock(
        side_effect=[
            httpx.Response(200, json=working_rpc),
            httpx.Response(200, json=done_rpc),
        ]
    )

    client = A2AAgentClient(
        base_url="http://agent.test",
        _http_client=mock_http,
        poll_interval=0.0,
    )
    task = await client.wait_for_completion("task-w")

    assert task.status.state == TaskState.TASK_STATE_COMPLETED


async def test_wait_for_completion_times_out() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient
    from nexus.core.errors import OrchestrationError

    working_rpc = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "id": "task-slow",
            "contextId": "ctx-1",
            "status": {"state": "TASK_STATE_WORKING"},
        },
    }

    mock_http = _make_mock_http_client(
        get_response=_make_agent_card_dict(),
        post_response=working_rpc,
    )

    client = A2AAgentClient(
        base_url="http://agent.test",
        _http_client=mock_http,
        poll_interval=0.0,
    )

    with pytest.raises(OrchestrationError) as exc_info:
        await client.wait_for_completion("task-slow", timeout=0.001)

    assert "timeout" in str(exc_info.value).lower() or exc_info.value.code == "A2A_TIMEOUT"


async def test_missing_sdk_raises_tool_error() -> None:
    from nexus.core.errors import ToolError
    import nexus.orchestration.a2a.client as _mod

    orig = _mod._HAS_A2A
    try:
        _mod._HAS_A2A = False  # type: ignore[attr-defined]
        with pytest.raises(ToolError) as exc_info:
            _mod.A2AAgentClient(base_url="http://x.test")
        assert exc_info.value.code == "A2A_SDK_MISSING"
        assert "nexus-ai[a2a]" in exc_info.value.hint
    finally:
        _mod._HAS_A2A = orig  # type: ignore[attr-defined]
