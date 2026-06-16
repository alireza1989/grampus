"""Tests for grampus version CLI subcommands."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from grampus.cli.main import cli
from grampus.core.types import AgentDefinition
from grampus.versioning.types import (
    ABTestConfig,
    AgentVersion,
    DeploymentRecord,
    VersionDiff,
    VersionStatus,
    compute_version_id,
)


def _make_def(prompt: str = "You are helpful.") -> AgentDefinition:
    return AgentDefinition(name="cli-agent", model="m", system_prompt=prompt, tools=[])


def _make_version(defn: AgentDefinition, tag: str = "v1.0") -> AgentVersion:
    return AgentVersion(
        version_id=compute_version_id(defn),
        agent_id="cli-agent",
        version_tag=tag,
        definition=defn,
        status=VersionStatus.PRODUCTION,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


class TestVersionListCommand:
    def test_list_output_contains_headers(self) -> None:
        runner = CliRunner()
        defn = _make_def()
        versions = [_make_version(defn, "v1.0")]

        mock_mgr = MagicMock()
        mock_mgr.list_versions = AsyncMock(return_value=versions)

        with patch("grampus.cli.commands.version._build_version_manager", return_value=mock_mgr):
            result = runner.invoke(cli, ["version", "list", "cli-agent"])

        assert result.exit_code == 0
        assert "v1.0" in result.output
        assert "production" in result.output.lower() or "PRODUCTION" in result.output

    def test_list_shows_version_id(self) -> None:
        runner = CliRunner()
        defn = _make_def()
        v = _make_version(defn)
        versions = [v]

        mock_mgr = MagicMock()
        mock_mgr.list_versions = AsyncMock(return_value=versions)

        with patch("grampus.cli.commands.version._build_version_manager", return_value=mock_mgr):
            result = runner.invoke(cli, ["version", "list", "cli-agent"])

        assert result.exit_code == 0
        assert v.version_id[:12] in result.output


class TestVersionDeployCommand:
    def test_deploy_prints_success(self) -> None:
        runner = CliRunner()
        defn = _make_def()
        v = _make_version(defn)
        record = DeploymentRecord(
            agent_id="cli-agent",
            version_id=v.version_id,
            deployed_at=datetime.now(UTC),
        )

        mock_mgr = MagicMock()
        mock_mgr.deploy = AsyncMock(return_value=record)

        with patch("grampus.cli.commands.version._build_version_manager", return_value=mock_mgr):
            result = runner.invoke(cli, ["version", "deploy", v.version_id, "--agent", "cli-agent"])

        assert result.exit_code == 0
        assert "Deployed" in result.output
        assert v.version_id[:12] in result.output or v.version_id in result.output


class TestVersionRollbackCommand:
    def test_rollback_prints_success(self) -> None:
        runner = CliRunner()
        defn = _make_def()
        v = _make_version(defn)
        record = DeploymentRecord(
            agent_id="cli-agent",
            version_id=v.version_id,
            deployed_at=datetime.now(UTC),
        )

        mock_mgr = MagicMock()
        mock_mgr.rollback = AsyncMock(return_value=record)

        with patch("grampus.cli.commands.version._build_version_manager", return_value=mock_mgr):
            result = runner.invoke(cli, ["version", "rollback", "cli-agent"])

        assert result.exit_code == 0
        assert "Rolled back" in result.output
        assert v.version_id[:12] in result.output or v.version_id in result.output

    def test_rollback_versioning_error_exits_one(self) -> None:
        from grampus.core.errors import VersioningError

        runner = CliRunner()
        mock_mgr = MagicMock()
        mock_mgr.rollback = AsyncMock(
            side_effect=VersioningError("No prior", code="NO_PRIOR_VERSION")
        )

        with patch("grampus.cli.commands.version._build_version_manager", return_value=mock_mgr):
            result = runner.invoke(cli, ["version", "rollback", "cli-agent"])

        assert result.exit_code == 1
        assert "No prior" in result.output


class TestVersionDiffCommand:
    def test_diff_prints_changes(self) -> None:
        runner = CliRunner()
        diff = VersionDiff(
            version_id_a="aaa",
            version_id_b="bbb",
            system_prompt_diff="--- a/aaa\n+++ b/bbb\n-Old\n+New\n",
            tools_added=["calc"],
            tools_removed=[],
            config_changes={"temperature": (0.0, 0.7)},
            has_changes=True,
        )

        mock_mgr = MagicMock()
        mock_mgr.diff = AsyncMock(return_value=diff)

        with patch("grampus.cli.commands.version._build_version_manager", return_value=mock_mgr):
            result = runner.invoke(cli, ["version", "diff", "aaa", "bbb", "--agent", "cli-agent"])

        assert result.exit_code == 0
        assert "calc" in result.output
        assert "temperature" in result.output

    def test_diff_no_changes(self) -> None:
        runner = CliRunner()
        diff = VersionDiff(
            version_id_a="aaa",
            version_id_b="aaa",
            system_prompt_diff="",
            tools_added=[],
            tools_removed=[],
            config_changes={},
            has_changes=False,
        )

        mock_mgr = MagicMock()
        mock_mgr.diff = AsyncMock(return_value=diff)

        with patch("grampus.cli.commands.version._build_version_manager", return_value=mock_mgr):
            result = runner.invoke(cli, ["version", "diff", "aaa", "aaa", "--agent", "cli-agent"])

        assert result.exit_code == 0
        assert "No changes" in result.output or "no changes" in result.output.lower()


class TestVersionABStartCommand:
    def test_ab_start_prints_experiment_id(self) -> None:
        runner = CliRunner()
        cfg = ABTestConfig(
            experiment_id="exp-123",
            agent_id="cli-agent",
            control_version_id="ctrl",
            treatment_version_id="trt",
            traffic_split=0.1,
            created_at=datetime.now(UTC),
        )

        mock_ab_mgr = MagicMock()
        mock_ab_mgr.start_test = AsyncMock(return_value=cfg)

        with patch("grampus.cli.commands.version._build_ab_manager", return_value=mock_ab_mgr):
            result = runner.invoke(
                cli,
                [
                    "version",
                    "ab-start",
                    "cli-agent",
                    "--control",
                    "ctrl",
                    "--treatment",
                    "trt",
                    "--split",
                    "0.1",
                ],
            )

        assert result.exit_code == 0
        assert "exp-123" in result.output


class TestVersionABStatusCommand:
    def test_ab_status_prints_metrics(self) -> None:
        from grampus.versioning.metrics import VersionMetrics
        from grampus.versioning.types import ABTestResult

        runner = CliRunner()
        ab_result = ABTestResult(
            experiment_id="exp-123",
            control_metrics=VersionMetrics(
                version_id="ctrl", total_runs=100, eval_pass_count=70, eval_total=100
            ),
            treatment_metrics=VersionMetrics(
                version_id="trt", total_runs=100, eval_pass_count=80, eval_total=100
            ),
            p_value=0.03,
            significant=True,
            winner_version_id="trt",
            recommendation="promote treatment",
        )

        mock_ab_mgr = MagicMock()
        mock_ab_mgr.evaluate = AsyncMock(return_value=ab_result)

        with patch("grampus.cli.commands.version._build_ab_manager", return_value=mock_ab_mgr):
            result = runner.invoke(cli, ["version", "ab-status", "exp-123"])

        assert result.exit_code == 0
        assert "ctrl" in result.output
        assert "trt" in result.output
        assert "0.03" in result.output or "p-value" in result.output.lower()
        assert "promote" in result.output.lower()
