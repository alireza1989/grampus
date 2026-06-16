"""Tests for the playground Renderer (no-color mode)."""

from __future__ import annotations

from grampus.cli.playground.renderer import Renderer
from grampus.cli.playground.session import PlaygroundSession, PlaygroundTurn
from grampus.core.types import TokenUsage


def _r() -> Renderer:
    return Renderer(use_color=False)


def _usage(
    input_tokens: int = 1234,
    output_tokens: int = 456,
    cost_usd: float = 0.0042,
) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost_usd,
        model="test-model",
    )


def _session_with_turns(n: int = 2, cost: float = 0.0042, tokens: int = 1690) -> PlaygroundSession:
    usage = _usage(cost_usd=cost)
    turn = PlaygroundTurn(
        user_input="Q", assistant_output="A", model="test-model", token_usage=usage
    )
    return PlaygroundSession(model="test-model", turns=[turn] * n)


class TestModelHeader:
    def test_model_header_contains_model_name(self) -> None:
        r = _r()
        header = r.model_header("claude-sonnet-4-6")
        assert "claude-sonnet-4-6" in header

    def test_model_header_has_box_chars(self) -> None:
        r = _r()
        header = r.model_header("gpt-4o")
        assert "╭" in header
        assert "╮" in header


class TestModelFooter:
    def test_model_footer_contains_token_counts(self) -> None:
        r = _r()
        footer = r.model_footer(_usage(), 2.1)
        assert "1,234" in footer
        assert "456" in footer

    def test_model_footer_contains_cost(self) -> None:
        r = _r()
        footer = r.model_footer(_usage(), 2.1)
        assert "$0.0042" in footer

    def test_model_footer_contains_duration(self) -> None:
        r = _r()
        footer = r.model_footer(_usage(), 3.7)
        assert "3.7" in footer

    def test_model_footer_none_usage_graceful(self) -> None:
        r = _r()
        footer = r.model_footer(None, 1.5)
        assert "1.5" in footer
        assert "╰" in footer

    def test_model_footer_has_box_chars(self) -> None:
        r = _r()
        footer = r.model_footer(None, 0.0)
        assert "╰" in footer
        assert "╯" in footer


class TestSeparator:
    def test_separator_contains_label(self) -> None:
        r = _r()
        sep = r.separator("Step 1")
        assert "Step 1" in sep

    def test_separator_no_label(self) -> None:
        r = _r()
        sep = r.separator()
        assert "─" in sep

    def test_separator_contains_dashes(self) -> None:
        r = _r()
        sep = r.separator("foo")
        assert "─" in sep


class TestCostSummary:
    def test_cost_summary_correct_turn_count(self) -> None:
        r = _r()
        session = _session_with_turns(3)
        summary = r.cost_summary(session)
        assert "3" in summary

    def test_cost_summary_correct_token_total(self) -> None:
        r = _r()
        # Two turns, each 1690 tokens → 3380 total
        session = _session_with_turns(2)
        summary = r.cost_summary(session)
        assert "3,380" in summary

    def test_cost_summary_shows_cost(self) -> None:
        r = _r()
        session = _session_with_turns(1, cost=0.0100)
        summary = r.cost_summary(session)
        assert "$" in summary


class TestFormatUsd:
    def test_format_usd_four_decimals(self) -> None:
        r = _r()
        assert r.format_usd(0.0042) == "$0.0042"

    def test_format_usd_rounds_to_four_places(self) -> None:
        r = _r()
        assert r.format_usd(1.23456) == "$1.2346"

    def test_format_usd_zero(self) -> None:
        r = _r()
        assert r.format_usd(0.0) == "$0.0000"


class TestComparisonHeader:
    def test_comparison_header_contains_models(self) -> None:
        r = _r()
        header = r.comparison_header(["claude-sonnet-4-6", "gpt-4o"])
        assert "claude-sonnet-4-6" in header
        assert "gpt-4o" in header

    def test_comparison_header_uses_vs_separator(self) -> None:
        r = _r()
        header = r.comparison_header(["a", "b", "c"])
        assert "vs" in header


class TestHelpText:
    def test_help_text_contains_all_commands(self) -> None:
        r = _r()
        help_ = r.help_text()
        for cmd in [
            "/model",
            "/system",
            "/compare",
            "/cost",
            "/reset",
            "/save",
            "/load",
            "/sessions",
            "/export",
            "/version",
            "/help",
            "/exit",
        ]:
            assert cmd in help_, f"Missing '{cmd}' in help text"


class TestColoredOutputs:
    def test_info_no_ansi_when_no_color(self) -> None:
        r = _r()
        assert "\033[" not in r.info("test message")

    def test_error_no_ansi_when_no_color(self) -> None:
        r = _r()
        assert "\033[" not in r.error("boom")

    def test_success_no_ansi_when_no_color(self) -> None:
        r = _r()
        assert "\033[" not in r.success("great")

    def test_info_contains_text(self) -> None:
        r = _r()
        assert "some info" in r.info("some info")

    def test_error_contains_text(self) -> None:
        r = _r()
        assert "oops" in r.error("oops")

    def test_success_contains_text(self) -> None:
        r = _r()
        assert "great" in r.success("great")

    def test_color_ansi_present_when_enabled(self) -> None:
        r = Renderer(use_color=True)
        assert "\033[" in r.error("test")
        assert "\033[" in r.success("test")
        assert "\033[" in r.info("test")
