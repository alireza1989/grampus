"""Tests for remote handoff path via A2AAgentClient."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_httpx import HTTPXMock

from nexus.core.errors import HandoffError
from nexus.core.types import AgentDefinition, AgentStatus, ExecutionResult, TokenUsage
from nexus.orchestration.handoff import HandoffContext, HandoffRequest


def _make_agent_def(name: str = "remote-target") -> AgentDefinition:
    return AgentDefinition(name=name, model="claude-3-5-haiku-20241022")


def _make_handoff_request(target: str = "remote-target") -> HandoffRequest:
    return HandoffRequest(
        source_agent_id="source-agent",
        source_session_id="session-1",
        target_agent_name=target,
        context=HandoffContext(task="do the remote work"),
    )


def _make_execution_result(output: str = "remote result") -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            model="claude-3-5-haiku-20241022",
        ),
        duration_seconds=0.5,
        steps_taken=2,
        status=AgentStatus.COMPLETED,
    )


async def test_remote_handoff_uses_a2a_client() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    registry = AgentRegistry()
    mock_client = MagicMock(spec=A2AAgentClient)
    mock_client.send_message = AsyncMock(
        return_value=MagicMock(
            task=MagicMock(
                status=MagicMock(
                    state=3,  # TASK_STATE_COMPLETED
                    message=MagicMock(
                        parts=[MagicMock(text="remote result", HasField=lambda _: True)]
                    ),
                )
            )
        )
    )

    registry.register_remote(
        name="remote-target",
        url="http://remote.test",
        api_key="key",
        _client=mock_client,
    )

    executor = HandoffExecutor(registry=registry)
    request = _make_handoff_request("remote-target")

    result = await executor.execute(request)

    mock_client.send_message.assert_called_once()
    assert result.status in ("completed", "failed")


async def test_remote_handoff_propagates_context_as_message() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    registry = AgentRegistry()
    mock_client = MagicMock(spec=A2AAgentClient)
    mock_client.send_message = AsyncMock(
        return_value=MagicMock(
            task=MagicMock(
                status=MagicMock(
                    state=3,
                    message=MagicMock(parts=[MagicMock(text="answer", HasField=lambda _: True)]),
                )
            )
        )
    )

    registry.register_remote(
        name="remote-ctx",
        url="http://remote.test",
        _client=mock_client,
    )

    executor = HandoffExecutor(registry=registry)
    request = HandoffRequest(
        source_agent_id="src",
        source_session_id="sess",
        target_agent_name="remote-ctx",
        context=HandoffContext(
            task="specific task text",
            context_summary="some context",
        ),
    )

    await executor.execute(request)

    call_kwargs = mock_client.send_message.call_args
    text_arg = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("text", "")
    assert "specific task text" in text_arg


async def test_remote_handoff_wraps_a2a_error_as_handoff_error() -> None:
    from nexus.core.errors import OrchestrationError
    from nexus.orchestration.a2a.client import A2AAgentClient
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    registry = AgentRegistry()
    mock_client = MagicMock(spec=A2AAgentClient)
    mock_client.send_message = AsyncMock(
        side_effect=OrchestrationError("remote failed", code="A2A_CLIENT_ERROR")
    )

    registry.register_remote(
        name="broken-remote",
        url="http://broken.test",
        _client=mock_client,
    )

    executor = HandoffExecutor(registry=registry)
    request = _make_handoff_request("broken-remote")

    with pytest.raises(HandoffError) as exc_info:
        await executor.execute(request)

    assert "broken-remote" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Dapr service invocation handoff tests
# ---------------------------------------------------------------------------

_JSONRPC_SUCCESS = {
    "jsonrpc": "2.0",
    "id": "test-id",
    "result": {
        "status": {
            "state": "completed",
            "message": {
                "parts": [{"text": "dapr result"}],
            },
        }
    },
}


async def test_dapr_handoff_posts_to_dapr_invoke_url(httpx_mock: HTTPXMock) -> None:
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    httpx_mock.add_response(
        url="http://localhost:3500/v1.0/invoke/nexus-worker/method/a2a",
        json=_JSONRPC_SUCCESS,
    )

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="dapr-target",
        dapr_app_id="nexus-worker",
        description="worker",
    )

    executor = HandoffExecutor(registry=registry)
    request = _make_handoff_request("dapr-target")

    result = await executor.execute(request)

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    assert "localhost:3500" in str(requests[0].url)
    assert "/v1.0/invoke/nexus-worker/method/a2a" in str(requests[0].url)
    assert result.status == "completed"


async def test_dapr_handoff_sends_jsonrpc_message_send(httpx_mock: HTTPXMock) -> None:
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    httpx_mock.add_response(
        url="http://localhost:3500/v1.0/invoke/nexus-worker/method/a2a",
        json=_JSONRPC_SUCCESS,
    )

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="dapr-target",
        dapr_app_id="nexus-worker",
        description="worker",
    )

    executor = HandoffExecutor(registry=registry)
    await executor.execute(_make_handoff_request("dapr-target"))

    req = httpx_mock.get_requests()[0]
    body = json.loads(req.content)
    assert body["method"] == "message/send"
    assert body["jsonrpc"] == "2.0"


async def test_dapr_handoff_extracts_output_from_response(httpx_mock: HTTPXMock) -> None:
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    httpx_mock.add_response(
        url="http://localhost:3500/v1.0/invoke/nexus-worker/method/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "x",
            "result": {
                "status": {
                    "state": "completed",
                    "message": {"parts": [{"text": "hello from dapr"}]},
                }
            },
        },
    )

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="dapr-target",
        dapr_app_id="nexus-worker",
        description="worker",
    )

    executor = HandoffExecutor(registry=registry)
    result = await executor.execute(_make_handoff_request("dapr-target"))

    assert result.output == "hello from dapr"


async def test_dapr_handoff_custom_port(httpx_mock: HTTPXMock) -> None:
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    httpx_mock.add_response(
        url="http://localhost:3501/v1.0/invoke/nexus-worker/method/a2a",
        json=_JSONRPC_SUCCESS,
    )

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="dapr-target",
        dapr_app_id="nexus-worker",
        description="worker",
    )

    executor = HandoffExecutor(registry=registry, dapr_http_port=3501)
    await executor.execute(_make_handoff_request("dapr-target"))

    req = httpx_mock.get_requests()[0]
    assert "3501" in str(req.url)


async def test_dapr_handoff_http_error_raises_handoff_error(httpx_mock: HTTPXMock) -> None:
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    httpx_mock.add_response(
        url="http://localhost:3500/v1.0/invoke/nexus-worker/method/a2a",
        status_code=503,
        text="service unavailable",
    )

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="dapr-target",
        dapr_app_id="nexus-worker",
        description="worker",
    )

    executor = HandoffExecutor(registry=registry)

    with pytest.raises(HandoffError) as exc_info:
        await executor.execute(_make_handoff_request("dapr-target"))

    assert "dapr-target" in str(exc_info.value) or "nexus-worker" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Three-way dispatch tests
# ---------------------------------------------------------------------------


async def test_three_way_dispatch_local_uses_runner() -> None:
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    registry = AgentRegistry()
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=_make_execution_result("local"))
    registry.register_local(
        name="local-agent",
        runner=mock_runner,
        description="local",
        agent_def=_make_agent_def("local-agent"),
    )

    executor = HandoffExecutor(registry=registry)
    result = await executor.execute(_make_handoff_request("local-agent"))

    mock_runner.run.assert_called_once()
    assert result.status == "completed"


async def test_three_way_dispatch_dapr_uses_dapr_path(httpx_mock: HTTPXMock) -> None:
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    httpx_mock.add_response(
        url="http://localhost:3500/v1.0/invoke/dapr-svc/method/a2a",
        json=_JSONRPC_SUCCESS,
    )

    registry = AgentRegistry()
    registry.register_dapr_service(name="dapr-agent", dapr_app_id="dapr-svc", description="d")

    executor = HandoffExecutor(registry=registry)
    result = await executor.execute(_make_handoff_request("dapr-agent"))

    assert result.status == "completed"
    assert len(httpx_mock.get_requests()) == 1


async def test_three_way_dispatch_remote_uses_a2a_client() -> None:
    from nexus.orchestration.a2a.client import A2AAgentClient
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    registry = AgentRegistry()
    mock_client = MagicMock(spec=A2AAgentClient)
    mock_client.send_message = AsyncMock(return_value=None)
    registry.register_remote(name="remote-agent", url="http://remote.test", _client=mock_client)

    executor = HandoffExecutor(registry=registry)
    result = await executor.execute(_make_handoff_request("remote-agent"))

    mock_client.send_message.assert_called_once()
    assert result.status == "completed"


async def test_three_way_dispatch_no_path_raises_handoff_error() -> None:
    from nexus.orchestration.a2a.registry import AgentEntry, AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    registry = AgentRegistry()
    # Manually insert an entry with no execution path
    from a2a.types.a2a_pb2 import AgentCard

    card = AgentCard()
    card.name = "orphan"
    registry._agents["orphan"] = AgentEntry(name="orphan", card=card)

    executor = HandoffExecutor(registry=registry)

    with pytest.raises(HandoffError) as exc_info:
        await executor.execute(_make_handoff_request("orphan"))

    assert exc_info.value.code in ("AGENT_NOT_CONFIGURED", "HANDOFF_EXECUTION_FAILED")


async def test_local_handoff_unchanged() -> None:
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.handoff import HandoffExecutor

    registry = AgentRegistry()
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=_make_execution_result("local done"))

    agent_def = _make_agent_def("local-target")
    registry.register_local(
        name="local-target",
        runner=mock_runner,
        description="Local agent",
        agent_def=agent_def,
    )

    executor = HandoffExecutor(registry=registry)
    request = HandoffRequest(
        source_agent_id="src",
        source_session_id="sess",
        target_agent_name="local-target",
        context=HandoffContext(task="local task"),
    )

    result = await executor.execute(request)

    mock_runner.run.assert_called_once()
    assert result.status == "completed"
    assert result.output == "local done"
