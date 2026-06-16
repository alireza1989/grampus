"""Tests for AgentRunner + VersionRouter integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grampus.core.types import AgentDefinition, AgentStatus, ExecutionResult, TokenUsage
from grampus.orchestration.runner import AgentRunner


def _make_def(prompt: str = "Original prompt.") -> AgentDefinition:
    return AgentDefinition(
        name="runner-test-agent",
        model="claude-sonnet-4-6",
        system_prompt=prompt,
        tools=[],
    )


def _make_execution_result(prompt: str = "done") -> ExecutionResult:
    return ExecutionResult(
        output=prompt,
        messages=[],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
            model="claude-sonnet-4-6",
        ),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


def _make_mock_model_client(response_content: str = "Final answer.") -> MagicMock:
    mock = MagicMock()
    response = MagicMock()
    response.content = response_content
    response.tool_calls = []
    response.stop_reason = "end_turn"
    response.model = "claude-sonnet-4-6"
    response.token_usage = TokenUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cost_usd=0.001,
        model="claude-sonnet-4-6",
    )
    mock.complete = AsyncMock(return_value=response)
    return mock


def _make_mock_tool_executor() -> MagicMock:
    mock = MagicMock()
    mock.execute = AsyncMock()
    return mock


def _make_mock_event_log() -> MagicMock:
    mock = MagicMock()
    event = MagicMock()
    event.event_id = "test-event"
    mock.append = AsyncMock(return_value=event)
    return mock


class TestRunnerWithoutVersionRouter:
    @pytest.mark.asyncio
    async def test_runner_without_version_router_no_regression(self) -> None:
        model_client = _make_mock_model_client()
        tool_executor = _make_mock_tool_executor()

        with patch("grampus.observability.events.EventLog.open") as mock_open:
            mock_open.return_value = _make_mock_event_log()
            runner = AgentRunner(model_client, tool_executor)
            defn = _make_def()
            result = await runner.run(defn, "Hello", session_id="test-session")

        assert result.output is not None
        assert result.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_runner_version_router_defaults_to_none(self) -> None:
        model_client = _make_mock_model_client()
        tool_executor = _make_mock_tool_executor()
        runner = AgentRunner(model_client, tool_executor)
        assert runner._version_router is None


class TestRunnerWithVersionRouter:
    @pytest.mark.asyncio
    async def test_runner_calls_version_router_resolve(self) -> None:
        model_client = _make_mock_model_client()
        tool_executor = _make_mock_tool_executor()

        version_router = MagicMock()
        resolved_def = _make_def("Resolved prompt.")
        version_router.resolve = AsyncMock(return_value=resolved_def)

        with patch("grampus.observability.events.EventLog.open") as mock_open:
            mock_open.return_value = _make_mock_event_log()
            runner = AgentRunner(model_client, tool_executor, version_router=version_router)
            original_def = _make_def("Original prompt.")
            await runner.run(original_def, "Hello", session_id="vs-session", user_id="user-1")

        version_router.resolve.assert_called_once()
        call_args = version_router.resolve.call_args
        assert call_args[0][0] == "runner-test-agent"

    @pytest.mark.asyncio
    async def test_runner_falls_back_on_router_error(self) -> None:
        model_client = _make_mock_model_client()
        tool_executor = _make_mock_tool_executor()

        version_router = MagicMock()
        version_router.resolve = AsyncMock(side_effect=RuntimeError("Router exploded"))

        with patch("grampus.observability.events.EventLog.open") as mock_open:
            mock_open.return_value = _make_mock_event_log()
            runner = AgentRunner(model_client, tool_executor, version_router=version_router)
            original_def = _make_def("Fallback prompt.")
            # Should NOT raise even though version_router raised
            result = await runner.run(original_def, "Hello", session_id="fallback-session")

        assert result.output is not None

    @pytest.mark.asyncio
    async def test_runner_falls_back_when_router_returns_none(self) -> None:
        model_client = _make_mock_model_client()
        tool_executor = _make_mock_tool_executor()

        version_router = MagicMock()
        version_router.resolve = AsyncMock(return_value=None)

        with patch("grampus.observability.events.EventLog.open") as mock_open:
            mock_open.return_value = _make_mock_event_log()
            runner = AgentRunner(model_client, tool_executor, version_router=version_router)
            original_def = _make_def("Original fallback.")
            result = await runner.run(original_def, "Hello", session_id="none-session")

        assert result.output is not None
