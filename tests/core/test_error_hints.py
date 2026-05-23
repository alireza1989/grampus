"""Tests for the hint field added to NexusError and its subclasses."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from nexus.core.errors import (
    BudgetExceededError,
    ConfigError,
    NexusError,
    OrchestrationError,
    ToolNotFoundError,
)
from nexus.tools.registry import ToolRegistry


class TestHintOnBase:
    def test_nexus_error_default_hint_is_empty_string(self) -> None:
        err = NexusError("msg", code="X")
        assert err.hint == ""

    def test_nexus_error_hint_stored(self) -> None:
        err = NexusError("msg", code="X", hint="do this")
        assert err.hint == "do this"

    def test_subclass_accepts_hint(self) -> None:
        err = ConfigError("msg", code="X", hint="fix it")
        assert err.hint == "fix it"

    def test_hint_does_not_affect_str(self) -> None:
        err = ConfigError("msg", code="X", hint="h")
        assert str(err) == "msg"

    def test_hint_defaults_do_not_break_existing_construction(self) -> None:
        err = NexusError("msg", code="X", details={"k": "v"})
        assert err.hint == ""
        assert err.details == {"k": "v"}


class TestHintOnRaiseSites:
    def test_budget_exceeded_error_has_hint(self) -> None:
        err = BudgetExceededError(
            "over budget",
            code="BUDGET_EXCEEDED",
            hint="Raise cost_budget_usd in AgentDefinition or break the task into smaller sub-tasks.",
        )
        assert err.hint != ""

    def test_tool_not_found_has_hint(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.get_or_raise("nonexistent_tool")
        assert exc_info.value.hint != ""

    def test_max_iterations_error_has_hint(self) -> None:
        err = OrchestrationError(
            "Max iterations exceeded",
            code="MAX_ITERATIONS_EXCEEDED",
            hint="Increase max_iterations in AgentDefinition or simplify the task to require fewer tool calls.",
        )
        assert err.hint != ""


class TestCliHintOutput:
    def test_cli_prints_hint_when_present(self) -> None:
        runner = CliRunner()
        exc = ConfigError("file not found", code="CONFIG_NOT_FOUND", hint="Run 'nexus init' first.")

        with runner.isolated_filesystem():
            result = runner.invoke(
                _make_echo_command(exc),
            )

        assert "Error:" in result.output
        assert "Hint:" in result.output
        assert "nexus init" in result.output

    def test_cli_no_hint_line_when_hint_empty(self) -> None:
        runner = CliRunner()
        exc = ConfigError("file not found", code="CONFIG_NOT_FOUND")

        with runner.isolated_filesystem():
            result = runner.invoke(
                _make_echo_command(exc),
            )

        assert "Error:" in result.output
        assert "Hint:" not in result.output


def _make_echo_command(exc: Exception):  # type: ignore[return]
    """Return a tiny Click command that calls _print_nexus_error with exc."""
    import click

    from nexus.cli.commands.run import _print_nexus_error

    @click.command()
    def _cmd() -> None:
        _print_nexus_error(exc)

    return _cmd
