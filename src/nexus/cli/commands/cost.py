"""nexus cost — display cost summary from the local cost log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from nexus.core.logging import get_logger

_log = get_logger(__name__)

_DEFAULT_LOG = ".nexus/cost_log.jsonl"
_COL_STEP = 14
_COL_MODEL = 14
_COL_TOKENS = 10
_COL_COST = 10


@click.command("cost")
@click.option("--agent", "agent_filter", default=None, help="Filter by agent ID.")
@click.option("--session", "session_filter", default=None, help="Filter by session ID.")
@click.option(
    "--last",
    "last_n",
    type=int,
    default=20,
    show_default=True,
    help="Show the last N events.",
)
@click.option(
    "--log-file",
    default=_DEFAULT_LOG,
    show_default=True,
    help="Path to the JSONL cost log file.",
)
def cost(
    agent_filter: str | None,
    session_filter: str | None,
    last_n: int,
    log_file: str,
) -> None:
    """Show cost summary from the cost log."""
    log_path = Path(log_file)
    if not log_path.exists():
        click.echo("No cost data found. Run an agent first.")
        return

    events = _load_events(log_path)
    events = _filter_events(events, agent_filter=agent_filter, session_filter=session_filter)
    events = events[-last_n:]

    if not events:
        click.echo("No matching cost events.")
        return

    _print_cost_table(events, agent_filter=agent_filter, session_filter=session_filter)


def _load_events(log_path: Path) -> list[dict[str, Any]]:
    """Read and parse all JSONL cost events from *log_path*."""
    events: list[dict[str, Any]] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    _log.warning("cost_log_parse_error", line=line)
    return events


def _filter_events(
    events: list[dict[str, Any]],
    *,
    agent_filter: str | None,
    session_filter: str | None,
) -> list[dict[str, Any]]:
    """Apply agent and session filters to the event list."""
    if agent_filter:
        events = [e for e in events if e.get("agent_id") == agent_filter]
    if session_filter:
        events = [e for e in events if e.get("session_id") == session_filter]
    return events


def _print_cost_table(
    events: list[dict[str, Any]],
    *,
    agent_filter: str | None,
    session_filter: str | None,
) -> None:
    """Render and print the cost summary table."""
    header = _build_header(events, agent_filter=agent_filter, session_filter=session_filter)
    click.echo(header)
    click.echo(_h_line())
    click.echo(_col_header())
    click.echo(_h_line())

    total_tokens = 0
    total_cost = 0.0

    for ev in events:
        tokens = ev.get("input_tokens", 0) + ev.get("output_tokens", 0)
        c = ev.get("cost_usd", 0.0)
        total_tokens += tokens
        total_cost += c
        click.echo(_data_row(ev.get("step_name", ""), ev.get("model_id", ""), tokens, c))

    click.echo(_h_line())
    click.echo(_total_row(total_tokens, total_cost))
    click.echo(_h_line())


def _build_header(
    events: list[dict[str, Any]],
    *,
    agent_filter: str | None,
    session_filter: str | None,
) -> str:
    """Build the table header line."""
    agent_str = agent_filter or (events[0].get("agent_id", "all") if events else "all")
    session_str = session_filter or "all"
    return f"Agent: {agent_str}   Session: {session_str}   Events: {len(events)}"


def _h_line() -> str:
    total_width = _COL_STEP + _COL_MODEL + _COL_TOKENS + _COL_COST + 13
    return "-" * total_width


def _col_header() -> str:
    return (
        f"{'Step':<{_COL_STEP}} | {'Model':<{_COL_MODEL}} | "
        f"{'Tokens':>{_COL_TOKENS}} | {'Cost':>{_COL_COST}}"
    )


def _data_row(step: str, model: str, tokens: int, c: float) -> str:
    return (
        f"{step:<{_COL_STEP}} | {model:<{_COL_MODEL}} | "
        f"{tokens:>{_COL_TOKENS},} | {'$' + f'{c:.4f}':>{_COL_COST}}"
    )


def _total_row(tokens: int, c: float) -> str:
    label = "TOTAL"
    return (
        f"{label:<{_COL_STEP}} | {'':^{_COL_MODEL}} | "
        f"{tokens:>{_COL_TOKENS},} | {'$' + f'{c:.4f}':>{_COL_COST}}"
    )
