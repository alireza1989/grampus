"""grampus replay — step through a past agent session from its event log."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import click

from grampus.cli.commands._utils import load_config
from grampus.core.errors import ConfigError
from grampus.observability.events import AgentEvent, EventType


@click.command("replay")
@click.argument("session_id")
@click.option("--config", "config_path", default="grampus.yaml", show_default=True)
@click.option(
    "--agent", "agent_id", default=None, help="Agent name (defaults to config default_agent)"
)
@click.option(
    "--from-step", default=0, show_default=True, type=int, help="Replay from sequence number"
)
@click.option("--json", "output_json", is_flag=True, default=False, help="Output raw JSON events")
def replay(
    session_id: str,
    config_path: str,
    agent_id: str | None,
    from_step: int,
    output_json: bool,
) -> None:
    """Replay a past agent session step-by-step from its event log.

    SESSION_ID is the session identifier printed by `grampus run`.
    Requires a state store configured in grampus.yaml.
    """
    asyncio.run(_replay_async(session_id, config_path, agent_id, from_step, output_json))


async def _replay_async(
    session_id: str,
    config_path: str,
    agent_id: str | None,
    from_step: int,
    output_json: bool,
) -> None:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    resolved_agent_id = (
        agent_id or getattr(getattr(cfg, "agent", None), "name", None) or "grampus-agent"
    )

    try:
        from grampus.dapr.client import DaprGateway
        from grampus.dapr.state import DaprStateStore
        from grampus.observability.events import EventLog
    except ImportError as exc:
        click.echo(f"Error: missing dependency: {exc}", err=True)
        sys.exit(1)

    dapr_cfg = getattr(cfg, "dapr", None)
    http_port = getattr(dapr_cfg, "http_port", 3500) if dapr_cfg else 3500
    store_name = getattr(dapr_cfg, "state_store", "statestore") if dapr_cfg else "statestore"

    try:
        gateway = DaprGateway(port=http_port)
        state_store: Any = DaprStateStore(
            gateway=gateway, store_name=store_name, namespace="grampus"
        )
        event_log = await EventLog.open(
            agent_id=resolved_agent_id,
            session_id=session_id,
            state_store=state_store,
        )
    except Exception as exc:
        click.echo(f"Error connecting to state store: {exc}", err=True)
        sys.exit(1)

    events = await event_log.replay_since(from_step)

    if not events:
        click.echo(f"No events found for session '{session_id}' (agent='{resolved_agent_id}').")
        click.echo("Tip: run with --agent to specify the agent name, or check the session ID.")
        sys.exit(1)

    if output_json:
        import json

        click.echo(json.dumps([e.model_dump(mode="json") for e in events], indent=2))
        return

    _render_events(events, resolved_agent_id, session_id)


def _render_events(events: list[AgentEvent], agent_id: str, session_id: str) -> None:
    width = 55
    bar = "═" * width

    click.echo(f"\n  {click.style(bar, bold=True)}")
    header = f"  Replay: session '{session_id}'   agent '{agent_id}'   {len(events)} events"
    click.echo(click.style(header, bold=True))
    click.echo(f"  {click.style(bar, bold=True)}\n")

    total_cost = 0.0
    total_steps = 0

    for event in events:
        seq = event.sequence_number
        ts = event.timestamp.strftime("%H:%M:%S")
        p = event.payload

        if event.event_type == EventType.AGENT_STARTED:
            inp = str(p.get("input", ""))[:80]
            model = p.get("model", "")
            click.echo(f"  [{seq}] {ts}  {click.style('AGENT STARTED', bold=True)}")
            click.echo(f'      Input: "{inp}"')
            click.echo(f"      Model: {model}")

        elif event.event_type == EventType.LLM_CALLED:
            step = p.get("step", "")
            in_tok = p.get("input_tokens", 0)
            out_tok = p.get("output_tokens", 0)
            click.echo(
                f"  [{seq}] {ts}  {click.style('LLM CALL', bold=True)}"
                f"          step={step}   in={in_tok}  out={out_tok}  tokens"
            )
            total_steps = max(total_steps, int(step) if step else 0)

        elif event.event_type == EventType.TOOL_CALLED:
            tool = p.get("tool", "")
            args = str(p.get("args", ""))[:80]
            click.echo(f"  [{seq}] {ts}  {click.style('TOOL CALLED', bold=True)}       {tool}")
            click.echo(f"      Args: {args}")

        elif event.event_type == EventType.TOOL_RESULT:
            tool = p.get("tool", "")
            ok = p.get("ok", True)
            output = str(p.get("output", ""))[:80]
            status_icon = "✓" if ok else "✗"
            click.echo(
                f"  [{seq}] {ts}  {click.style('TOOL RESULT', bold=True)}       {tool}   {status_icon}"
            )
            click.echo(f"      Output: {output}")

        elif event.event_type == EventType.HUMAN_INPUT_REQUESTED:
            question = str(p.get("question", ""))[:80]
            click.echo(
                f"  [{seq}] {ts}  {click.style('HUMAN INPUT', bold=True)}       ⏸ PAUSED — human input required"
            )
            if question:
                click.echo(f"      Question: {question}")

        elif event.event_type == EventType.AGENT_COMPLETED:
            output = str(p.get("output", ""))[:120]
            steps = p.get("steps", 0)
            cost = float(p.get("cost_usd", 0.0))
            total_cost += cost
            click.echo(f"  [{seq}] {ts}  {click.style('AGENT COMPLETED', bold=True)}")
            click.echo(f'      Output: "{output}"')
            click.echo(f"      Steps: {steps}   Cost: ${cost:.4f}")

        elif event.event_type == EventType.AGENT_FAILED:
            error = str(p.get("error", "unknown error"))
            click.echo(f"  [{seq}] {ts}  {click.style('AGENT FAILED', bold=True, fg='red')}")
            click.echo(f"      {click.style(error, fg='red')}")

        elif event.event_type == EventType.MEMORY_READ:
            click.echo(
                f"  [{seq}] {ts}  {click.style('MEMORY READ', bold=True)}"
                f"       query={str(p.get('query', ''))[:60]}"
            )

        elif event.event_type == EventType.MEMORY_WRITTEN:
            click.echo(
                f"  [{seq}] {ts}  {click.style('MEMORY WRITTEN', bold=True)}"
                f"    key={str(p.get('key', ''))[:60]}"
            )

        elif event.event_type == EventType.SAFETY_VIOLATION:
            reason = str(p.get("reason", ""))[:80]
            click.echo(
                f"  [{seq}] {ts}  {click.style('SAFETY VIOLATION', bold=True, fg='red')}  {reason}"
            )

        else:
            click.echo(f"  [{seq}] {ts}  {event.event_type}  {p}")

        click.echo()

    click.echo(f"  {click.style(bar, bold=True)}")
    click.echo(
        click.style(
            f"  Total: {total_steps} steps   ${total_cost:.4f}",
            bold=True,
        )
    )
    click.echo(f"  {click.style(bar, bold=True)}\n")
