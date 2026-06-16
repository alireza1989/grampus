"""Terminal output formatter using ANSI codes and Unicode box-drawing characters."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from grampus.core.types import TokenUsage

if TYPE_CHECKING:
    from grampus.cli.playground.session import PlaygroundSession

_WIDTH = 70

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"

_COMMANDS = [
    ("/model <name>", "Switch active model"),
    ("/models", "List available model names"),
    ("/system <text>", "Set system prompt (or /system file:<path>)"),
    ("/compare <model2> [model3...]", "Run last message against multiple models"),
    ("/cost", "Show session token and cost summary"),
    ("/reset", "Clear conversation history and turns"),
    ("/save [name]", "Save session to ~/.grampus/playground/"),
    ("/load <name>", "Load a saved session"),
    ("/sessions", "List saved sessions with metadata"),
    ("/export [path]", "Export last turn as EvalCase JSON"),
    ("/version save <name>", "Save system prompt as named version"),
    ("/version diff <v1> <v2>", "Show unified diff between two versions"),
    ("/help", "Show this help message"),
    ("/exit", "Exit the playground (also /quit)"),
]


class Renderer:
    """Formats terminal output for the playground REPL.

    Args:
        use_color: Force color on/off. None auto-detects via sys.stdout.isatty().
    """

    def __init__(self, use_color: bool | None = None) -> None:
        if use_color is None:
            self._color = sys.stdout.isatty()
        else:
            self._color = use_color

    def _c(self, code: str, text: str) -> str:
        if self._color:
            return f"{code}{text}{_RESET}"
        return text

    def model_header(self, model: str) -> str:
        """╭─── model-name ────────╮"""
        label = f" {model} "
        dashes = "─" * max(0, _WIDTH - len(label) - 2)
        mid = f"─── {label}{dashes}"
        mid = mid[: _WIDTH - 2]
        line = f"╭{mid}╮"
        return self._c(_CYAN, line)

    def model_footer(self, usage: TokenUsage | None, duration: float) -> str:
        """╰─ ↑input ↓output tokens · $cost · Xs ─╯"""
        if usage:
            in_fmt = f"{usage.input_tokens:,}"
            out_fmt = f"{usage.output_tokens:,}"
            cost_fmt = self.format_usd(usage.cost_usd)
            inner = f" ↑{in_fmt} ↓{out_fmt} tokens · {cost_fmt} · {duration:.1f}s "
        else:
            inner = f" {duration:.1f}s "
        pad = "─" * max(0, _WIDTH - len(inner) - 3)
        line = f"╰─{inner}{pad}╯"
        return self._c(_CYAN, line)

    def separator(self, label: str = "") -> str:
        """── label ──────────────────────────────"""
        if label:
            inner = f" {label} "
            dashes = "─" * max(0, _WIDTH - len(inner) - 2)
            return self._c(_DIM, f"──{inner}{dashes}")
        return self._c(_DIM, "─" * _WIDTH)

    def cost_summary(self, session: PlaygroundSession) -> str:
        """Session: N turns · T tokens · $X total"""
        n = len(session.turns)
        tokens = session.total_tokens()
        cost = session.total_cost_usd()
        return (
            f"Session: {n} turn{'s' if n != 1 else ''} · "
            f"{tokens:,} tokens · {self.format_usd(cost)} total"
        )

    def comparison_header(self, models: list[str]) -> str:
        """Comparing: model-a vs model-b vs model-c"""
        joined = " vs ".join(models)
        return self._c(_BOLD, f"Comparing: {joined}")

    def format_usd(self, usd: float) -> str:
        """Format a dollar amount to 4 decimal places."""
        return f"${usd:.4f}"

    def info(self, msg: str) -> str:
        """Dim/gray informational line."""
        return self._c(_DIM, msg)

    def error(self, msg: str) -> str:
        """Red error line."""
        return self._c(_RED, msg)

    def success(self, msg: str) -> str:
        """Green success line."""
        return self._c(_GREEN, msg)

    def help_text(self) -> str:
        """Full /help output listing all REPL commands."""
        col_w = max(len(cmd) for cmd, _ in _COMMANDS) + 2
        lines = [self.separator("Nexus Playground Commands"), ""]
        for cmd, desc in _COMMANDS:
            lines.append(f"  {cmd:<{col_w}}  {desc}")
        lines.append("")
        return "\n".join(lines)
