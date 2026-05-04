"""Tests for nexus eval command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from nexus.cli.main import cli
from nexus.evaluation.suite import CaseResult, SuiteResult


def _make_suite_result(*, pass_rate: float = 0.8, passed: int = 4, total: int = 5) -> SuiteResult:
    failed = total - passed
    case_results = [
        CaseResult(
            case_id=f"case-{i}",
            case_name=f"Case {i}",
            passed=(i < passed),
            assertion_results=[],
            duration_seconds=0.1,
        )
        for i in range(total)
    ]
    return SuiteResult(
        suite_name="test-suite",
        total_cases=total,
        passed=passed,
        failed=failed,
        errors=0,
        pass_rate=pass_rate,
        avg_duration_seconds=0.1,
        case_results=case_results,
        run_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        total_cost_usd=0.0042,
    )


def _write_suite_file(path: Path) -> Path:
    suite_file = path / "suite.py"
    suite_file.write_text("def create_suite(): pass\n")
    return suite_file


class TestEvalCommand:
    """Tests for nexus eval command."""

    def test_eval_runs_suite_and_prints_report(self, tmp_path: Path) -> None:
        suite_file = _write_suite_file(tmp_path)
        mock_result = _make_suite_result()
        with (
            patch("nexus.cli.commands.eval._load_suite"),
            patch("nexus.cli.commands.eval._run_suite", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            runner = CliRunner()
            result = runner.invoke(cli, ["eval", str(suite_file)])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()

    def test_eval_json_format_outputs_valid_json(self, tmp_path: Path) -> None:
        suite_file = _write_suite_file(tmp_path)
        mock_result = _make_suite_result()
        with (
            patch("nexus.cli.commands.eval._load_suite"),
            patch("nexus.cli.commands.eval._run_suite", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            runner = CliRunner()
            result = runner.invoke(cli, ["eval", str(suite_file), "--format", "json"])
        assert result.exit_code == 0, result.output
        # JSON output appears before/after the stderr summary line; find the JSON block
        output = result.output
        json_start = output.find("{")
        assert json_start != -1, f"No JSON in output: {output!r}"
        parsed = json.loads(output[json_start:].strip())
        assert "suite_name" in parsed or "suite_result" in parsed

    def test_eval_junit_format_outputs_xml(self, tmp_path: Path) -> None:
        suite_file = _write_suite_file(tmp_path)
        mock_result = _make_suite_result()
        with (
            patch("nexus.cli.commands.eval._load_suite"),
            patch("nexus.cli.commands.eval._run_suite", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            runner = CliRunner()
            result = runner.invoke(cli, ["eval", str(suite_file), "--format", "junit"])
        assert result.exit_code == 0, result.output
        assert "<?xml" in result.output
        assert "testsuites" in result.output

    def test_eval_output_file_written(self, tmp_path: Path) -> None:
        suite_file = _write_suite_file(tmp_path)
        output_file = tmp_path / "report.txt"
        mock_result = _make_suite_result()
        with (
            patch("nexus.cli.commands.eval._load_suite"),
            patch("nexus.cli.commands.eval._run_suite", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            runner = CliRunner()
            result = runner.invoke(cli, ["eval", str(suite_file), "--output", str(output_file)])
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        assert len(output_file.read_text()) > 0

    def test_eval_fail_under_exits_1_when_below_threshold(self, tmp_path: Path) -> None:
        suite_file = _write_suite_file(tmp_path)
        mock_result = _make_suite_result(pass_rate=0.8)
        with (
            patch("nexus.cli.commands.eval._load_suite"),
            patch("nexus.cli.commands.eval._run_suite", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            runner = CliRunner()
            result = runner.invoke(cli, ["eval", str(suite_file), "--fail-under", "0.9"])
        assert result.exit_code == 1

    def test_eval_fail_under_exits_0_when_above_threshold(self, tmp_path: Path) -> None:
        suite_file = _write_suite_file(tmp_path)
        mock_result = _make_suite_result(pass_rate=0.95, passed=5, total=5)
        with (
            patch("nexus.cli.commands.eval._load_suite"),
            patch("nexus.cli.commands.eval._run_suite", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            runner = CliRunner()
            result = runner.invoke(cli, ["eval", str(suite_file), "--fail-under", "0.9"])
        assert result.exit_code == 0

    def test_eval_prints_summary_to_stderr(self, tmp_path: Path) -> None:
        suite_file = _write_suite_file(tmp_path)
        mock_result = _make_suite_result()
        with (
            patch("nexus.cli.commands.eval._load_suite"),
            patch("nexus.cli.commands.eval._run_suite", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            runner = CliRunner()
            result = runner.invoke(cli, ["eval", str(suite_file)])
        assert result.exit_code == 0
        # CliRunner merges stderr into output by default
        combined = result.output
        assert "passed" in combined.lower() or "%" in combined

    def test_eval_missing_suite_file_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", str(tmp_path / "nonexistent.py")])
        assert result.exit_code != 0

    def test_eval_missing_create_suite_exits_with_helpful_error(self, tmp_path: Path) -> None:
        suite_file = tmp_path / "suite.py"
        suite_file.write_text("# no create_suite function\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", str(suite_file)])
        assert result.exit_code != 0
        assert "create_suite" in result.output
