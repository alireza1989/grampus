"""Click commands for the interactive prompt playground."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from grampus.cli.playground.model_factory import make_client
from grampus.cli.playground.renderer import Renderer
from grampus.cli.playground.repl import CompareResult, run_compare, run_repl
from grampus.cli.playground.session import _SESSIONS_DIR, PlaygroundSession
from grampus.core.config import GrampusConfig
from grampus.core.types import Message, Role


def _load_config() -> GrampusConfig:
    return GrampusConfig()


@click.group()
def playground() -> None:
    """Interactive prompt playground for testing and comparing LLM responses."""


@playground.command("start")
@click.option("--model", "-m", default="claude-haiku-4-5", help="Starting model")
@click.option("--system", "-s", default="", help="System prompt text")
@click.option("--system-file", type=click.Path(exists=True), help="Load system prompt from file")
@click.option("--load", "load_session", default=None, help="Load a saved session by name")
def start(model: str, system: str, system_file: str | None, load_session: str | None) -> None:
    """Start interactive REPL playground."""
    config = _load_config()
    renderer = Renderer()

    if system_file:
        system = Path(system_file).read_text()

    if load_session:
        try:
            session = PlaygroundSession.load(load_session)
            system = session.system_prompt
            model = session.model
            click.echo(renderer.success(f"Loaded session '{load_session}'"))
        except FileNotFoundError as exc:
            click.echo(renderer.error(str(exc)), err=True)
            sys.exit(1)

    asyncio.run(run_repl(config, model=model, system_prompt=system))


@playground.command("run")
@click.argument("message")
@click.option("--model", "-m", default="claude-haiku-4-5", help="Model to use")
@click.option("--system", "-s", default="", help="System prompt text")
@click.option("--stream/--no-stream", default=True, help="Stream the response")
def run(message: str, model: str, system: str, stream: bool) -> None:
    """Single-shot prompt — print response and cost."""
    config = _load_config()
    renderer = Renderer()
    client = make_client(model, config)

    messages: list[Message] = []
    if system:
        messages.append(Message(role=Role.SYSTEM, content=system))
    messages.append(Message(role=Role.USER, content=message))

    async def _run() -> None:
        import time

        start = time.monotonic()
        if stream:
            click.echo(renderer.model_header(model))
            usage = None
            async for chunk in client.stream(messages=messages, model=model):
                if chunk.is_final:
                    usage = chunk.token_usage
                else:
                    sys.stdout.write(chunk.delta)
                    sys.stdout.flush()
            elapsed = time.monotonic() - start
            sys.stdout.write("\n")
            click.echo(renderer.model_footer(usage, elapsed))
        else:
            response = await client.complete(messages=messages, model=model)
            elapsed = time.monotonic() - start
            click.echo(renderer.model_header(model))
            click.echo(response.content or "")
            click.echo(renderer.model_footer(response.token_usage, elapsed))

    asyncio.run(_run())


@playground.command("compare")
@click.argument("message")
@click.option("--models", "-m", required=True, help="Comma-separated model names")
@click.option("--system", "-s", default="", help="System prompt text")
def compare(message: str, models: str, system: str) -> None:
    """Run same prompt against multiple models side-by-side."""
    config = _load_config()
    renderer = Renderer()
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    results: list[CompareResult] = asyncio.run(
        run_compare(user_message=message, models=model_list, system_prompt=system, config=config)
    )

    click.echo(renderer.comparison_header(model_list))
    for r in results:
        click.echo(renderer.model_header(r.model))
        if r.error:
            click.echo(renderer.error(r.error))
        else:
            click.echo(r.output)
        click.echo(renderer.model_footer(r.token_usage, r.duration_seconds))

    click.echo(renderer.separator("cost comparison"))
    click.echo(f"  {'Model':<30}  {'Tokens':>8}  {'Cost':>10}  {'Time':>8}")
    click.echo(f"  {'─' * 30}  {'─' * 8}  {'─' * 10}  {'─' * 8}")
    for r in results:
        tokens = r.token_usage.total_tokens if r.token_usage else 0
        cost = renderer.format_usd(r.token_usage.cost_usd) if r.token_usage else "N/A"
        click.echo(f"  {r.model:<30}  {tokens:>8,}  {cost:>10}  {r.duration_seconds:>7.1f}s")


@playground.command("sessions")
def sessions() -> None:
    """List saved playground sessions."""
    renderer = Renderer()
    if not _SESSIONS_DIR.exists() or not list(_SESSIONS_DIR.glob("*.json")):
        click.echo(renderer.info("No saved sessions found"))
        return
    for p in sorted(_SESSIONS_DIR.glob("*.json")):
        try:
            data = PlaygroundSession.model_validate_json(p.read_text())
            n = len(data.turns)
            cost = renderer.format_usd(data.total_cost_usd())
            label = data.name or data.session_id[:8]
            click.echo(
                f"  {label:<20}  {n} turn{'s' if n != 1 else ''}  {cost}  "
                f"{data.created_at.strftime('%Y-%m-%d')}"
            )
        except Exception:
            click.echo(f"  {p.stem} (unreadable)")


@playground.command("show")
@click.argument("name")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["transcript", "json"]),
    default="transcript",
    help="Output format",
)
def show(name: str, fmt: str) -> None:
    """Show a saved session as transcript or raw JSON."""
    renderer = Renderer()
    try:
        session = PlaygroundSession.load(name)
    except FileNotFoundError as exc:
        click.echo(renderer.error(str(exc)), err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(session.model_dump_json(indent=2))
        return

    # Transcript
    if session.system_prompt:
        click.echo(renderer.separator("system"))
        click.echo(session.system_prompt)
    for i, turn in enumerate(session.turns, 1):
        click.echo(renderer.separator(f"turn {i}"))
        click.echo(f"User: {turn.user_input}")
        click.echo(renderer.model_header(turn.model))
        click.echo(turn.assistant_output)
        click.echo(renderer.model_footer(turn.token_usage, turn.duration_seconds))
    click.echo(renderer.cost_summary(session))
