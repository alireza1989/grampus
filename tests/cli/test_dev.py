"""Tests for grampus dev command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from grampus.cli.main import cli


def _write_grampus_yaml(path: Path) -> Path:
    cfg = path / "grampus.yaml"
    cfg.write_text(
        "agent:\n  name: dev-agent\n  model: claude-sonnet-4-6\n"
        "  system_prompt: null\n  max_iterations: 5\n"
        "  memory_enabled: false\n  cost_budget_usd: 0.5\n"
    )
    return cfg


class TestDevCommand:
    """Tests for grampus dev command."""

    def test_dev_prints_startup_banner(self, tmp_path: Path) -> None:
        cfg = _write_grampus_yaml(tmp_path)
        runner = CliRunner()
        with patch("grampus.cli.commands.dev._run_dev_loop") as mock_loop:
            mock_loop.return_value = None
            result = runner.invoke(cli, ["dev", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        output_lower = result.output.lower()
        assert "grampus" in output_lower or "dev" in output_lower

    def test_dev_validates_config_on_start(self, tmp_path: Path) -> None:
        cfg = _write_grampus_yaml(tmp_path)
        runner = CliRunner()
        with patch("grampus.cli.commands.dev._run_dev_loop") as mock_loop:
            mock_loop.return_value = None
            result = runner.invoke(cli, ["dev", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        # Config should have been loaded and summarized
        assert "dev-agent" in result.output or "claude" in result.output.lower()

    def test_dev_missing_config_exits_with_error(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["dev", "--config", str(tmp_path / "missing.yaml")])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_dev_prints_config_summary(self, tmp_path: Path) -> None:
        cfg = _write_grampus_yaml(tmp_path)
        runner = CliRunner()
        with patch("grampus.cli.commands.dev._run_dev_loop") as mock_loop:
            mock_loop.return_value = None
            result = runner.invoke(cli, ["dev", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        # Should show some config summary
        output = result.output
        assert "8000" in output or "port" in output.lower() or "claude" in output.lower()
