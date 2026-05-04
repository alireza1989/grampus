"""Tests for nexus cost command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from click.testing import CliRunner

from nexus.cli.main import cli


def _write_cost_log(path: Path, events: list[dict]) -> Path:  # type: ignore[type-arg]
    log_dir = path / ".nexus"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cost_log.jsonl"
    with log_file.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return log_file


def _make_events(
    count: int = 5,
    *,
    agent_id: str = "research-agent",
    session_id: str = "sess-abc",
) -> list[dict]:  # type: ignore[type-arg]
    return [
        {
            "agent_id": agent_id,
            "session_id": session_id,
            "step_name": f"step_{i}",
            "model_id": "claude-sonnet-4-6",
            "input_tokens": 100 * (i + 1),
            "output_tokens": 50 * (i + 1),
            "cost_usd": 0.001 * (i + 1),
            "timestamp": datetime(2024, 1, 15, 10, i, 0, tzinfo=UTC).isoformat(),
        }
        for i in range(count)
    ]


class TestCostCommand:
    """Tests for nexus cost command."""

    def test_cost_prints_table_from_log_file(self, tmp_path: Path) -> None:
        events = _make_events()
        log_file = _write_cost_log(tmp_path, events)
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--log-file", str(log_file)])
        assert result.exit_code == 0, result.output
        assert "step_0" in result.output or "claude" in result.output.lower()

    def test_cost_filters_by_agent(self, tmp_path: Path) -> None:
        events = _make_events(agent_id="agent-a") + _make_events(agent_id="agent-b")
        log_file = _write_cost_log(tmp_path, events)
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--log-file", str(log_file), "--agent", "agent-a"])
        assert result.exit_code == 0, result.output
        assert "agent-a" in result.output
        assert "agent-b" not in result.output

    def test_cost_filters_by_session(self, tmp_path: Path) -> None:
        events = _make_events(session_id="sess-x") + _make_events(session_id="sess-y")
        log_file = _write_cost_log(tmp_path, events)
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--log-file", str(log_file), "--session", "sess-x"])
        assert result.exit_code == 0, result.output
        assert "sess-x" in result.output
        assert "sess-y" not in result.output

    def test_cost_last_n_limits_events(self, tmp_path: Path) -> None:
        events = _make_events(count=10)
        log_file = _write_cost_log(tmp_path, events)
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--log-file", str(log_file), "--last", "3"])
        assert result.exit_code == 0, result.output
        # Only the last 3 events should appear: step_7, step_8, step_9
        assert "step_7" in result.output or "step_9" in result.output
        assert "step_0" not in result.output

    def test_cost_missing_log_file_prints_friendly_message(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["cost", "--log-file", str(tmp_path / ".nexus" / "cost_log.jsonl")],
        )
        assert result.exit_code == 0
        assert "No cost data" in result.output or "not found" in result.output.lower()

    def test_cost_shows_total_row(self, tmp_path: Path) -> None:
        events = _make_events(count=3)
        log_file = _write_cost_log(tmp_path, events)
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--log-file", str(log_file)])
        assert result.exit_code == 0, result.output
        assert "TOTAL" in result.output or "total" in result.output.lower()
