"""Tests for NexusMetrics recording in AgentRunner."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nexus.core.models.base import ModelResponse
from nexus.core.types import (
    AgentDefinition,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from nexus.observability.metrics import NexusMetrics
from nexus.orchestration.runner import AgentRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_def(name: str = "test-agent") -> AgentDefinition:
    return AgentDefinition(name=name, model="test-model")


def _token_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="test-model"
    )


def _model_response(
    content: str | None = "Hello!",
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls or [],
        token_usage=_token_usage(),
        model="test-model",
        stop_reason="end_turn",
    )


def _tool_call(name: str = "my_tool", call_id: str = "tc-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={})


def _tool_result(call_id: str = "tc-1") -> ToolResult:
    return ToolResult(tool_call_id=call_id, output="tool output", duration_ms=5)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_client() -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=_model_response())
    return client


@pytest.fixture
def tool_executor() -> AsyncMock:
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_tool_result())
    return executor


@pytest.fixture
def metrics() -> NexusMetrics:
    return NexusMetrics(agent_id="test-agent")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunnerMetrics:
    async def test_run_records_llm_call(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        metrics: NexusMetrics,
    ) -> None:
        runner = AgentRunner(model_client, tool_executor, nexus_metrics=metrics)
        await runner.run(_agent_def(), "Hello", session_id="s1")
        assert metrics.snapshot().llm_call_count == 1

    async def test_run_records_token_counts(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        metrics: NexusMetrics,
    ) -> None:
        runner = AgentRunner(model_client, tool_executor, nexus_metrics=metrics)
        await runner.run(_agent_def(), "Hello", session_id="s1")
        assert metrics.snapshot().total_tokens > 0

    async def test_run_records_tool_call(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        metrics: NexusMetrics,
    ) -> None:
        tc = _tool_call()
        model_client.complete.side_effect = [
            _model_response(content=None, tool_calls=[tc]),
            _model_response(content="Done"),
        ]
        runner = AgentRunner(model_client, tool_executor, nexus_metrics=metrics)
        await runner.run(_agent_def(), "Use a tool", session_id="s1")
        assert metrics.snapshot().total_tool_calls == 1

    async def test_run_records_error(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        metrics: NexusMetrics,
    ) -> None:
        model_client.complete.side_effect = RuntimeError("boom")
        runner = AgentRunner(model_client, tool_executor, nexus_metrics=metrics)
        with pytest.raises(RuntimeError):
            await runner.run(_agent_def(), "Fail please", session_id="s1")
        assert metrics.snapshot().total_errors == 1

    async def test_no_metrics_no_crash(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
    ) -> None:
        runner = AgentRunner(model_client, tool_executor, nexus_metrics=None)
        result = await runner.run(_agent_def(), "Hello", session_id="s1")
        assert result.output is not None

    async def test_multiple_llm_calls_accumulate(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        metrics: NexusMetrics,
    ) -> None:
        tc = _tool_call()
        model_client.complete.side_effect = [
            _model_response(content=None, tool_calls=[tc]),
            _model_response(content="Done"),
        ]
        runner = AgentRunner(model_client, tool_executor, nexus_metrics=metrics)
        await runner.run(_agent_def(), "Two LLM calls", session_id="s1")
        assert metrics.snapshot().llm_call_count == 2

    async def test_tool_success_recorded(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        metrics: NexusMetrics,
    ) -> None:
        tc = _tool_call()
        model_client.complete.side_effect = [
            _model_response(content=None, tool_calls=[tc]),
            _model_response(content="Done"),
        ]
        runner = AgentRunner(model_client, tool_executor, nexus_metrics=metrics)
        await runner.run(_agent_def(), "Use a tool", session_id="s1")
        snap = metrics.snapshot()
        assert snap.total_tool_calls == 1

    async def test_cost_recorded(
        self,
        model_client: AsyncMock,
        tool_executor: AsyncMock,
        metrics: NexusMetrics,
    ) -> None:
        runner = AgentRunner(model_client, tool_executor, nexus_metrics=metrics)
        await runner.run(_agent_def(), "Hello", session_id="s1")
        assert metrics.snapshot().total_cost_usd > 0.0
