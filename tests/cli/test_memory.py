"""Tests for grampus memory command."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from grampus.cli.main import cli
from grampus.memory.types import EpisodicRecord, SemanticFact


def _make_episodic_records() -> list[EpisodicRecord]:
    return [
        EpisodicRecord(
            id="rec-001",
            agent_id="test-agent",
            session_id="sess-a",
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            content="The user asked about Python async programming.",
        ),
        EpisodicRecord(
            id="rec-002",
            agent_id="test-agent",
            session_id="sess-b",
            timestamp=datetime(2024, 1, 16, 11, 0, 0, tzinfo=UTC),
            content="Discussion about Dapr state management patterns.",
        ),
    ]


def _make_semantic_facts() -> list[SemanticFact]:
    return [
        SemanticFact(
            id="fact-001",
            subject="Python",
            predicate="is",
            object_value="async-capable",
        ),
    ]


def _make_mock_episodic(records: list[EpisodicRecord]) -> AsyncMock:
    mock = AsyncMock()
    mock.list_all = AsyncMock(return_value=records)
    mock.delete = AsyncMock()
    return mock


def _make_mock_semantic(facts: list[SemanticFact]) -> AsyncMock:
    mock = AsyncMock()
    mock.list_all = AsyncMock(return_value=facts)
    mock.delete = AsyncMock()
    return mock


class TestMemoryInspect:
    """Tests for grampus memory inspect."""

    def test_inspect_prints_table_of_records(self, tmp_path: Path) -> None:
        records = _make_episodic_records()
        mock_ep = _make_mock_episodic(records)
        mock_sem = _make_mock_semantic([])
        with (
            patch("grampus.cli.commands.memory._make_episodic", return_value=mock_ep),
            patch("grampus.cli.commands.memory._make_semantic", return_value=mock_sem),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["memory", "inspect", "test-agent"])
        assert result.exit_code == 0, result.output
        assert "rec-001" in result.output
        assert "rec-002" in result.output

    def test_inspect_filters_by_session(self, tmp_path: Path) -> None:
        records = _make_episodic_records()
        mock_ep = _make_mock_episodic(records)
        mock_sem = _make_mock_semantic([])
        with (
            patch("grampus.cli.commands.memory._make_episodic", return_value=mock_ep),
            patch("grampus.cli.commands.memory._make_semantic", return_value=mock_sem),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["memory", "inspect", "test-agent", "--session", "sess-a"])
        assert result.exit_code == 0, result.output
        assert "rec-001" in result.output
        assert "rec-002" not in result.output

    def test_inspect_filters_by_type(self) -> None:
        records = _make_episodic_records()
        facts = _make_semantic_facts()
        mock_ep = _make_mock_episodic(records)
        mock_sem = _make_mock_semantic(facts)
        with (
            patch("grampus.cli.commands.memory._make_episodic", return_value=mock_ep),
            patch("grampus.cli.commands.memory._make_semantic", return_value=mock_sem),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["memory", "inspect", "test-agent", "--type", "semantic"])
        assert result.exit_code == 0, result.output
        assert "fact-001" in result.output


class TestMemoryClear:
    """Tests for grampus memory clear."""

    def test_clear_prompts_confirmation(self) -> None:
        records = _make_episodic_records()
        mock_ep = _make_mock_episodic(records)
        mock_sem = _make_mock_semantic([])
        with (
            patch("grampus.cli.commands.memory._make_episodic", return_value=mock_ep),
            patch("grampus.cli.commands.memory._make_semantic", return_value=mock_sem),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["memory", "clear", "test-agent"], input="n\n")
        assert result.exit_code == 0
        assert "Continue" in result.output or "continue" in result.output.lower()
        mock_ep.delete.assert_not_called()

    def test_clear_yes_flag_skips_prompt(self) -> None:
        records = _make_episodic_records()
        mock_ep = _make_mock_episodic(records)
        mock_sem = _make_mock_semantic([])
        with (
            patch("grampus.cli.commands.memory._make_episodic", return_value=mock_ep),
            patch("grampus.cli.commands.memory._make_semantic", return_value=mock_sem),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["memory", "clear", "test-agent", "--yes"])
        assert result.exit_code == 0, result.output
        assert mock_ep.delete.call_count == len(records)

    def test_clear_prints_deleted_count(self) -> None:
        records = _make_episodic_records()
        mock_ep = _make_mock_episodic(records)
        mock_sem = _make_mock_semantic([])
        with (
            patch("grampus.cli.commands.memory._make_episodic", return_value=mock_ep),
            patch("grampus.cli.commands.memory._make_semantic", return_value=mock_sem),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["memory", "clear", "test-agent", "--yes"])
        assert "2" in result.output or "deleted" in result.output.lower()

    def test_clear_aborts_on_no(self) -> None:
        records = _make_episodic_records()
        mock_ep = _make_mock_episodic(records)
        mock_sem = _make_mock_semantic([])
        with (
            patch("grampus.cli.commands.memory._make_episodic", return_value=mock_ep),
            patch("grampus.cli.commands.memory._make_semantic", return_value=mock_sem),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["memory", "clear", "test-agent"], input="n\n")
        assert result.exit_code == 0
        mock_ep.delete.assert_not_called()


class TestMemoryStats:
    """Tests for grampus memory stats."""

    def test_stats_prints_counts_and_timestamps(self) -> None:
        records = _make_episodic_records()
        facts = _make_semantic_facts()
        mock_ep = _make_mock_episodic(records)
        mock_sem = _make_mock_semantic(facts)
        with (
            patch("grampus.cli.commands.memory._make_episodic", return_value=mock_ep),
            patch("grampus.cli.commands.memory._make_semantic", return_value=mock_sem),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["memory", "stats", "test-agent"])
        assert result.exit_code == 0, result.output
        assert "2" in result.output  # episodic count
        assert "1" in result.output  # semantic count
