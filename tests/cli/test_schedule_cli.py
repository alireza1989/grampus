"""Tests for nexus schedule CLI commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from nexus.cli.main import cli
from nexus.core.errors import DaprJobsError


def _invoke(*args: str) -> object:
    runner = CliRunner()
    return runner.invoke(cli, list(args))


def _make_cfg() -> object:
    """Minimal config stub with dapr sub-config."""
    from types import SimpleNamespace

    dapr = SimpleNamespace(host="localhost", port=3500)
    return SimpleNamespace(dapr=dapr)


def test_schedule_help() -> None:
    result = _invoke("schedule", "--help")
    assert result.exit_code == 0
    assert "create" in result.output
    assert "delete" in result.output
    assert "list" in result.output
    assert "trigger" in result.output


def test_schedule_create_dry_run() -> None:
    result = _invoke(
        "schedule",
        "create",
        "--name",
        "x",
        "--cron",
        "@daily",
        "--input",
        "go",
        "--dry-run",
        "--config",
        "/dev/null",
    )
    assert result.exit_code == 0, result.output
    assert "x" in result.output
    assert "@daily" in result.output


def test_schedule_create_dry_run_no_dapr_call() -> None:
    # DaprJobsClient is imported lazily inside _create_async from nexus.dapr.jobs
    with patch("nexus.dapr.jobs.DaprJobsClient") as mock_cls:
        result = _invoke(
            "schedule",
            "create",
            "--name",
            "nodapr",
            "--cron",
            "@hourly",
            "--input",
            "test",
            "--dry-run",
            "--config",
            "/dev/null",
        )
    assert result.exit_code == 0
    mock_cls.assert_not_called()


def test_schedule_create_success() -> None:
    mock_client = AsyncMock()
    mock_client.schedule = AsyncMock(return_value=None)

    with (
        patch(
            "nexus.cli.commands.schedule.load_config",
            return_value=_make_cfg(),
        ),
        patch("nexus.dapr.jobs.DaprJobsClient", return_value=mock_client),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["schedule", "create", "--name", "daily", "--cron", "@daily", "--input", "go"],
        )
    assert result.exit_code == 0, result.output
    assert "daily" in result.output
    assert "created" in result.output
    mock_client.schedule.assert_awaited_once()


def test_schedule_create_dapr_error() -> None:
    mock_client = AsyncMock()
    mock_client.schedule = AsyncMock(
        side_effect=DaprJobsError("sidecar unreachable", code="JOBS_SCHEDULE_FAILED")
    )

    with (
        patch(
            "nexus.cli.commands.schedule.load_config",
            return_value=_make_cfg(),
        ),
        patch("nexus.dapr.jobs.DaprJobsClient", return_value=mock_client),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["schedule", "create", "--name", "bad", "--cron", "@daily", "--input", "go"],
        )
    assert result.exit_code == 1
    assert "sidecar unreachable" in result.output or "Error" in result.output
    assert "Dapr sidecar" in result.output


def test_schedule_list_prints_note() -> None:
    with patch(
        "nexus.cli.commands.schedule.load_config",
        return_value=_make_cfg(),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule", "list"])
    assert result.exit_code == 0
    assert "does not support listing" in result.output


def test_schedule_delete_found() -> None:
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock(return_value=True)

    with (
        patch(
            "nexus.cli.commands.schedule.load_config",
            return_value=_make_cfg(),
        ),
        patch("nexus.dapr.jobs.DaprJobsClient", return_value=mock_client),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule", "delete", "my-job"])
    assert result.exit_code == 0
    assert "deleted" in result.output


def test_schedule_delete_not_found() -> None:
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock(return_value=False)

    with (
        patch(
            "nexus.cli.commands.schedule.load_config",
            return_value=_make_cfg(),
        ),
        patch("nexus.dapr.jobs.DaprJobsClient", return_value=mock_client),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule", "delete", "missing-job"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_schedule_delete_dapr_error() -> None:
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock(
        side_effect=DaprJobsError("connection refused", code="JOBS_DELETE_FAILED")
    )

    with (
        patch(
            "nexus.cli.commands.schedule.load_config",
            return_value=_make_cfg(),
        ),
        patch("nexus.dapr.jobs.DaprJobsClient", return_value=mock_client),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule", "delete", "err-job"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_schedule_create_missing_config() -> None:
    result = _invoke(
        "schedule",
        "create",
        "--name",
        "x",
        "--cron",
        "@daily",
        "--input",
        "go",
        "--config",
        "/nonexistent/nexus.yaml",
    )
    assert result.exit_code == 1
