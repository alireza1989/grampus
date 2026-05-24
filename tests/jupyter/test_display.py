"""Tests for nexus.jupyter.display — HTML/rich rendering helpers."""

from __future__ import annotations

from unittest.mock import patch

from nexus.core.types import AgentStatus, ExecutionResult, Message, Role, TokenUsage
from nexus.jupyter.display import (
    render_messages,
    render_result,
    render_tool_call,
    render_tool_result,
)


def _make_result(
    output: str = "Hello world",
    duration: float = 1.5,
    total_tokens: int = 100,
    cost_usd: float = 0.0025,
    tool_calls_made: int = 2,
    steps_taken: int = 3,
) -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[],
        tool_calls_made=tool_calls_made,
        token_usage=TokenUsage(
            input_tokens=80,
            output_tokens=20,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            model="test-model",
        ),
        duration_seconds=duration,
        steps_taken=steps_taken,
        status=AgentStatus.COMPLETED,
    )


class TestRenderResult:
    def test_render_result_returns_string_without_ipython(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_result(_make_result())
        assert isinstance(result, str)

    def test_render_result_contains_output_text(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_result(_make_result(output="My agent output"))
        assert "My agent output" in result

    def test_render_result_contains_token_count(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_result(_make_result(total_tokens=1234))
        assert "1,234" in result

    def test_render_result_contains_cost(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_result(_make_result(cost_usd=0.0025))
        assert "0.0025" in result

    def test_render_result_escapes_html_in_output(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_result(_make_result(output="<script>alert('xss')</script>"))
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_render_result_contains_agent_name(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_result(_make_result(), agent_name="MyAgent")
        assert "MyAgent" in result


class TestRenderToolCall:
    def test_render_tool_call_contains_tool_name(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_tool_call("search_web", {"query": "test"})
        assert "search_web" in result

    def test_render_tool_call_contains_args_preview(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_tool_call("search_web", {"query": "nexus"})
        assert "nexus" in result

    def test_render_tool_call_truncates_long_args(self) -> None:
        long_args = {"key": "x" * 200}
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_tool_call("some_tool", long_args)
        # args_preview is str(arguments)[:80], so full 200-char value is not in the output
        assert "x" * 81 not in result


class TestRenderToolResult:
    def test_render_tool_result_contains_tool_name(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_tool_result("search_web", "some result")
        assert "search_web" in result

    def test_render_tool_result_contains_output_preview(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_tool_result("search_web", "found relevant data")
        assert "found relevant data" in result


class TestRenderMessages:
    def test_render_messages_empty_list(self) -> None:
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_messages([])
        assert isinstance(result, str)

    def test_render_messages_contains_role(self) -> None:
        messages = [Message(role=Role.USER, content="Hello")]
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_messages(messages)
        assert "user" in result

    def test_render_messages_contains_content(self) -> None:
        messages = [Message(role=Role.ASSISTANT, content="World")]
        with patch("nexus.jupyter.display._ipython_available", return_value=False):
            result = render_messages(messages)
        assert "World" in result
