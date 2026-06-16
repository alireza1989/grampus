"""Tests for grampus run --stream flag."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from grampus.cli.main import cli
from grampus.core.types import (
    AgentStatus,
    ExecutionResult,
    StreamChunk,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model="test"
    )


def _mock_execution_result() -> ExecutionResult:
    return ExecutionResult(
        output="Hello from agent",
        messages=[],
        tool_calls_made=0,
        token_usage=_token_usage(),
        duration_seconds=0.5,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


def _write_grampus_yaml(path: Path) -> None:
    (path / "grampus.yaml").write_text(
        "agent:\n  name: test-agent\n  model: claude-sonnet-4-6\n"
        "  system_prompt: You are helpful.\n  max_iterations: 10\n"
        "  memory_enabled: false\n  cost_budget_usd: 1.0\n"
    )


def _write_agent_py(path: Path) -> Path:
    agent_file = path / "agent.py"
    agent_file.write_text(
        "from unittest.mock import MagicMock\n"
        "from grampus.core.types import AgentDefinition\n\n"
        "def create_runner():\n"
        "    return MagicMock()\n\n"
        "def create_agent_def():\n"
        "    return AgentDefinition(name='test-agent', model='claude-sonnet-4-6')\n"
    )
    return agent_file


def _make_stream_events(tokens: list[str]) -> list[StreamEvent]:
    events: list[StreamEvent] = [
        StreamEvent(event_type=StreamEventType.AGENT_START, message="test-agent")
    ]
    events.append(StreamEvent(event_type=StreamEventType.ITERATION_START, iteration=1))
    for token in tokens:
        events.append(
            StreamEvent(
                event_type=StreamEventType.TOKEN,
                chunk=StreamChunk(delta=token, model="test"),
            )
        )
    events.append(
        StreamEvent(
            event_type=StreamEventType.AGENT_END,
            chunk=StreamChunk(
                delta="",
                is_final=True,
                token_usage=_token_usage(),
                model="test",
            ),
        )
    )
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunStreamFlag:
    def test_cli_stream_flag_prints_tokens(self, tmp_path: Path) -> None:
        _write_grampus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path)

        tokens = ["Hello", " ", "world", "!"]
        stream_events = _make_stream_events(tokens)

        async def _mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
            for event in stream_events:
                yield event

        mock_runner = type("MockRunner", (), {"stream": _mock_stream})()

        with patch("grampus.cli.commands.run._build_runner", return_value=mock_runner):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(agent_file),
                    "--config",
                    str(tmp_path / "grampus.yaml"),
                    "--input",
                    "Hi",
                    "--stream",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Hello" in result.output
        assert "world" in result.output

    def test_cli_no_stream_flag_unchanged(self, tmp_path: Path) -> None:
        _write_grampus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path)
        from unittest.mock import AsyncMock

        mock_result = _mock_execution_result()
        mock_runner = type("MockRunner", (), {"run": AsyncMock(return_value=mock_result)})()

        with patch("grampus.cli.commands.run._build_runner", return_value=mock_runner):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(agent_file),
                    "--config",
                    str(tmp_path / "grampus.yaml"),
                    "--input",
                    "Hello agent",
                    "--no-stream",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Hello from agent" in result.output

    def test_cli_stream_prints_tool_call_events(self, tmp_path: Path) -> None:
        _write_grampus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path)

        tc = ToolCall(id="tc-1", name="web_search", arguments={"q": "test"})
        tr = ToolResult(tool_call_id="tc-1", output="3 results", duration_ms=10)
        stream_events = [
            StreamEvent(event_type=StreamEventType.AGENT_START, message="test-agent"),
            StreamEvent(event_type=StreamEventType.ITERATION_START, iteration=1),
            StreamEvent(event_type=StreamEventType.TOOL_CALL_START, tool_call=tc),
            StreamEvent(event_type=StreamEventType.TOOL_CALL_END, tool_call=tc, tool_result=tr),
            StreamEvent(event_type=StreamEventType.ITERATION_START, iteration=2),
            StreamEvent(
                event_type=StreamEventType.TOKEN,
                chunk=StreamChunk(delta="Done!", model="test"),
            ),
            StreamEvent(
                event_type=StreamEventType.AGENT_END,
                chunk=StreamChunk(is_final=True, token_usage=_token_usage(), model="test"),
            ),
        ]

        async def _mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
            for event in stream_events:
                yield event

        mock_runner = type("MockRunner", (), {"stream": _mock_stream})()

        with patch("grampus.cli.commands.run._build_runner", return_value=mock_runner):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(agent_file),
                    "--config",
                    str(tmp_path / "grampus.yaml"),
                    "--input",
                    "Search something",
                    "--stream",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "web_search" in result.output
        assert "Done!" in result.output

    def test_cli_stream_prints_token_summary(self, tmp_path: Path) -> None:
        _write_grampus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path)

        stream_events = _make_stream_events(["Hi!"])

        async def _mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
            for event in stream_events:
                yield event

        mock_runner = type("MockRunner", (), {"stream": _mock_stream})()

        with patch("grampus.cli.commands.run._build_runner", return_value=mock_runner):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(agent_file),
                    "--config",
                    str(tmp_path / "grampus.yaml"),
                    "--input",
                    "Hi",
                    "--stream",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "tokens" in result.output.lower() or "30" in result.output

    def test_cli_stream_flag_help_shown(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert "--stream" in result.output or "stream" in result.output.lower()
