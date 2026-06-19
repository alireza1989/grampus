"""Shared fixtures for E2E agent tests."""

from __future__ import annotations

import pytest

from grampus.core.models.base import ModelResponse
from grampus.core.types import (
    AgentDefinition,
    TokenUsage,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


class _FakeLLM:
    """Returns responses from a list in sequence; repeats last entry when exhausted."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def complete(self, messages: list, **kwargs: object) -> ModelResponse:
        if not self._responses:
            return _default_response()
        resp = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return resp


class _FakeToolExecutor:
    """Returns fixed string outputs keyed by tool name."""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self._results = results or {}

    async def execute(self, tc: object) -> ToolResult:
        name = getattr(tc, "name", "")
        tc_id = getattr(tc, "id", "unknown")
        output = self._results.get(name, "done")
        return ToolResult(tool_call_id=tc_id, output=output, error=None, duration_ms=1)


def _default_response(content: str = "Done.") -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="fake"
        ),
        model="fake",
        stop_reason="end_turn",
    )


@pytest.fixture
def agent_def() -> AgentDefinition:
    return AgentDefinition(
        name="test-agent",
        model="fake",
        system_prompt="You are a test agent.",
        cost_budget_usd=1.0,
        tools=["calculator"],
    )


@pytest.fixture
def calculator_tool_def() -> ToolDefinition:
    return ToolDefinition(
        name="calculator",
        description="Evaluate math expressions",
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="Math expression",
                required=True,
            )
        ],
    )


@pytest.fixture
def fake_llm() -> _FakeLLM:
    return _FakeLLM(responses=[])


@pytest.fixture
def fake_executor() -> _FakeToolExecutor:
    return _FakeToolExecutor()
