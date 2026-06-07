"""Tests for remote handoff path via A2AAgentClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.core.errors import HandoffError
from nexus.core.types import AgentDefinition, AgentStatus, ExecutionResult, TokenUsage
from nexus.orchestration.handoff import HandoffContext, HandoffRequest, HandoffPolicy


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
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.a2a.client import A2AAgentClient
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
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.a2a.client import A2AAgentClient
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
    from nexus.orchestration.a2a.registry import AgentRegistry
    from nexus.orchestration.a2a.client import A2AAgentClient
    from nexus.orchestration.handoff import HandoffExecutor
    from nexus.core.errors import OrchestrationError

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
