"""Tests for nexus run command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from nexus.cli.main import cli
from nexus.core.types import AgentStatus, ExecutionResult, TokenUsage


def _mock_execution_result() -> ExecutionResult:
    return ExecutionResult(
        output="Hello from agent",
        messages=[],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            model="test-model",
        ),
        duration_seconds=0.5,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


def _write_nexus_yaml(path: Path) -> None:
    (path / "nexus.yaml").write_text(
        "agent:\n  name: test-agent\n  model: claude-sonnet-4-6\n"
        "  system_prompt: You are helpful.\n  max_iterations: 10\n"
        "  memory_enabled: false\n  cost_budget_usd: 1.0\n"
    )


def _write_agent_py(path: Path, *, has_runner: bool = True, has_agent_def: bool = True) -> Path:
    agent_file = path / "agent.py"
    lines = []
    if has_runner:
        lines.append("from unittest.mock import AsyncMock, MagicMock")
        lines.append("from nexus.orchestration.runner import AgentRunner, RunnerConfig")
        lines.append("")
        lines.append("def create_runner():")
        lines.append("    return MagicMock()")
    if has_agent_def:
        lines.append("")
        lines.append("from nexus.core.types import AgentDefinition")
        lines.append("def create_agent_def():")
        lines.append("    return AgentDefinition(name='test-agent', model='claude-sonnet-4-6')")
    agent_file.write_text("\n".join(lines))
    return agent_file


class TestRunCommand:
    """Tests for nexus run command."""

    def test_run_with_input_flag_calls_runner(self, tmp_path: Path) -> None:
        _write_nexus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path)
        mock_result = _mock_execution_result()
        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_result)
        with patch("nexus.cli.commands.run._build_runner", return_value=mock_runner):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(agent_file),
                    "--config",
                    str(tmp_path / "nexus.yaml"),
                    "--input",
                    "Hello agent",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_runner.run.assert_called_once()

    def test_run_prints_agent_output(self, tmp_path: Path) -> None:
        _write_nexus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path)
        mock_result = _mock_execution_result()
        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_result)
        with patch("nexus.cli.commands.run._build_runner", return_value=mock_runner):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(agent_file),
                    "--config",
                    str(tmp_path / "nexus.yaml"),
                    "--input",
                    "Hello agent",
                ],
            )
        assert "Hello from agent" in result.output

    def test_run_prints_token_usage(self, tmp_path: Path) -> None:
        _write_nexus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path)
        mock_result = _mock_execution_result()
        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_result)
        with patch("nexus.cli.commands.run._build_runner", return_value=mock_runner):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(agent_file),
                    "--config",
                    str(tmp_path / "nexus.yaml"),
                    "--input",
                    "Hello",
                ],
            )
        assert "30" in result.output or "tokens" in result.output.lower()

    def test_run_missing_agent_file_exits_nonzero(self, tmp_path: Path) -> None:
        _write_nexus_yaml(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(tmp_path / "nonexistent.py"),
                "--config",
                str(tmp_path / "nexus.yaml"),
                "--input",
                "hi",
            ],
        )
        assert result.exit_code != 0

    def test_run_missing_create_runner_exits_with_helpful_error(self, tmp_path: Path) -> None:
        _write_nexus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path, has_runner=False)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(agent_file),
                "--config",
                str(tmp_path / "nexus.yaml"),
                "--input",
                "hi",
            ],
        )
        assert result.exit_code != 0
        assert "create_runner" in result.output

    def test_run_missing_create_agent_def_uses_yaml_fallback(self, tmp_path: Path) -> None:
        _write_nexus_yaml(tmp_path)
        agent_file = _write_agent_py(tmp_path, has_agent_def=False)
        mock_result = _mock_execution_result()
        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_result)
        with patch("nexus.cli.commands.run._build_runner", return_value=mock_runner):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(agent_file),
                    "--config",
                    str(tmp_path / "nexus.yaml"),
                    "--input",
                    "Hello",
                ],
            )
        assert result.exit_code == 0, result.output
