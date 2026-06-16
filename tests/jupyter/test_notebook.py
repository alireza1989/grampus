"""Tests for grampus.jupyter.notebook — GrampusNotebook façade."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from grampus.core.types import (
    AgentDefinition,
    AgentStatus,
    ExecutionResult,
    StreamChunk,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from grampus.jupyter._compat import ensure_async_compatible, run_async
from grampus.jupyter.notebook import GrampusNotebook, StreamSummary


def _make_agent_def(name: str = "TestAgent") -> AgentDefinition:
    return AgentDefinition(name=name, model="test-model")


def _make_result(output: str = "Done!") -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[],
        tool_calls_made=1,
        token_usage=TokenUsage(
            input_tokens=50,
            output_tokens=10,
            total_tokens=60,
            cost_usd=0.001,
            model="test-model",
        ),
        duration_seconds=0.5,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


def _make_mock_runner(result: ExecutionResult | None = None) -> MagicMock:
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result or _make_result())
    return runner


async def _stream_events(
    tokens: list[str] | None = None,
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Yield a representative sequence of StreamEvents for testing."""
    yield StreamEvent(event_type=StreamEventType.AGENT_START, message="start")
    for token in tokens or ["Hello", " world"]:
        yield StreamEvent(
            event_type=StreamEventType.TOKEN,
            chunk=StreamChunk(delta=token),
        )
    for tc_name, tc_args in tool_calls or []:
        tc = ToolCall(id="tc-1", name=tc_name, arguments=tc_args)
        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_START, tool_call=tc)
        tr = ToolResult(tool_call_id="tc-1", output="tool output", error=None)
        yield StreamEvent(
            event_type=StreamEventType.TOOL_CALL_END,
            tool_call=tc,
            tool_result=tr,
        )
    yield StreamEvent(
        event_type=StreamEventType.AGENT_END,
        chunk=StreamChunk(
            is_final=True,
            token_usage=TokenUsage(
                input_tokens=50,
                output_tokens=20,
                total_tokens=70,
                cost_usd=0.002,
                model="test-model",
            ),
        ),
    )


class TestGrampusNotebook:
    async def test_notebook_run_returns_execution_result(self) -> None:
        expected = _make_result()
        runner = _make_mock_runner(expected)
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=False)
        result = await nb.run("do something")
        assert result is expected

    async def test_notebook_run_calls_runner_run(self) -> None:
        runner = _make_mock_runner()
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=False)
        await nb.run("do something")
        runner.run.assert_called_once()

    async def test_notebook_run_uses_instance_session_id(self) -> None:
        runner = _make_mock_runner()
        nb = GrampusNotebook(runner, _make_agent_def(), session_id="my-session", auto_display=False)
        await nb.run("task")
        _, kwargs = runner.run.call_args
        assert kwargs["session_id"] == "my-session"

    async def test_notebook_run_uses_override_session_id(self) -> None:
        runner = _make_mock_runner()
        nb = GrampusNotebook(runner, _make_agent_def(), session_id="default", auto_display=False)
        await nb.run("task", session_id="override")
        _, kwargs = runner.run.call_args
        assert kwargs["session_id"] == "override"

    async def test_notebook_run_auto_display_calls_display(self) -> None:
        runner = _make_mock_runner()
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=True)
        with patch.object(nb, "display") as mock_display:
            await nb.run("task")
        mock_display.assert_called_once()

    async def test_notebook_run_auto_display_false_skips_display(self) -> None:
        runner = _make_mock_runner()
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=False)
        with patch.object(nb, "display") as mock_display:
            await nb.run("task")
        mock_display.assert_not_called()

    def test_notebook_run_sync_returns_result(self) -> None:
        expected = _make_result()
        runner = _make_mock_runner(expected)
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=False)
        result = nb.run_sync("do something")
        assert result.output == expected.output

    async def test_notebook_stream_returns_stream_summary(self) -> None:
        runner = MagicMock()
        runner.stream = MagicMock(return_value=_stream_events())
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=False)
        with patch("grampus.jupyter.notebook.render_stream_token"):
            summary = await nb.stream("stream task")
        assert isinstance(summary, StreamSummary)

    async def test_notebook_stream_summary_has_output(self) -> None:
        runner = MagicMock()
        runner.stream = MagicMock(return_value=_stream_events(tokens=["Hello", " world"]))
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=False)
        with patch("grampus.jupyter.notebook.render_stream_token"):
            summary = await nb.stream("stream task")
        assert summary.output == "Hello world"

    async def test_notebook_stream_summary_tool_calls_counted(self) -> None:
        runner = MagicMock()
        runner.stream = MagicMock(
            return_value=_stream_events(tool_calls=[("search", {"q": "test"})])
        )
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=False)
        with patch("grampus.jupyter.notebook.render_stream_token"):
            summary = await nb.stream("stream task")
        assert summary.tool_calls_made == 1

    async def test_notebook_stream_auto_display_shows_tokens(self) -> None:
        runner = MagicMock()
        runner.stream = MagicMock(return_value=_stream_events(tokens=["Hi"]))
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=True)
        with patch("grampus.jupyter.notebook.render_stream_token") as mock_rst:
            await nb.stream("task")
        mock_rst.assert_called_with("Hi")

    async def test_notebook_stream_tool_events_displayed(self) -> None:
        runner = MagicMock()
        runner.stream = MagicMock(return_value=_stream_events(tool_calls=[("my_tool", {"x": 1})]))
        nb = GrampusNotebook(runner, _make_agent_def(), auto_display=True)
        with (
            patch("grampus.jupyter.notebook._display") as mock_display,
            patch("grampus.jupyter.notebook.render_stream_token"),
        ):
            await nb.stream("task")
        # TOOL_CALL_START and TOOL_CALL_END each call _display once
        assert mock_display.call_count >= 2

    def test_stream_summary_dataclass_fields(self) -> None:
        usage = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3, cost_usd=0.0, model="m")
        s = StreamSummary(output="hi", tool_calls_made=5, token_usage=usage)
        assert s.output == "hi"
        assert s.tool_calls_made == 5
        assert s.token_usage is usage


class TestCompat:
    def test_run_async_in_non_jupyter_context(self) -> None:
        """run_async uses asyncio.run when there is no running event loop."""

        async def simple() -> str:
            return "result"

        result = run_async(simple())
        assert result == "result"

    def test_ensure_async_compatible_no_ops_without_nest_asyncio(self) -> None:
        """ensure_async_compatible is a no-op when nest_asyncio is not installed."""
        with patch.dict("sys.modules", {"nest_asyncio": None}):
            ensure_async_compatible()  # must not raise
