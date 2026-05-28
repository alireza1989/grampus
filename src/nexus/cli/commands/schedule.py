"""nexus schedule — manage scheduled agent runs via Dapr Jobs."""

from __future__ import annotations

import asyncio
import sys

import click

from nexus.cli.commands._utils import load_config, print_table
from nexus.core.errors import ConfigError, DaprJobsError


@click.group("schedule")
def schedule() -> None:
    """Manage scheduled agent runs (requires Dapr sidecar)."""


@schedule.command("create")
@click.option("--name", required=True, help="Unique job name.")
@click.option(
    "--cron",
    required=True,
    help='Cron string or interval, e.g. "0 9 * * *" or "@every 1h".',
)
@click.option(
    "--input",
    "input_text",
    required=True,
    help="Input text sent to the agent on each trigger.",
)
@click.option("--session-prefix", default="sched", show_default=True)
@click.option("--config", "config_path", default="nexus.yaml", show_default=True)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print config without calling Dapr.",
)
def schedule_create(
    name: str,
    cron: str,
    input_text: str,
    session_prefix: str,
    config_path: str,
    dry_run: bool,
) -> None:
    """Register a new scheduled job with Dapr."""
    asyncio.run(_create_async(name, cron, input_text, session_prefix, config_path, dry_run))


async def _create_async(
    name: str,
    cron: str,
    input_text: str,
    session_prefix: str,
    config_path: str,
    dry_run: bool,
) -> None:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}")
        hint = getattr(exc, "hint", "")
        if hint:
            click.echo(f"  Hint: {hint}")
        sys.exit(1)

    from nexus.dapr.schedule_store import ScheduleConfig

    config = ScheduleConfig(
        name=name,
        cron=cron,
        input_text=input_text,
        session_prefix=session_prefix,
    )

    if dry_run:
        print_table(
            ["Field", "Value"],
            [
                ["name", config.name],
                ["cron", config.cron],
                ["input", config.input_text],
                ["session_prefix", config.session_prefix],
                ["enabled", str(config.enabled)],
            ],
            title="Schedule configuration (dry-run):",
        )
        return

    from nexus.dapr.jobs import DaprJobsClient

    client = DaprJobsClient(host=cfg.dapr.host, port=cfg.dapr.port)
    try:
        await client.schedule(
            name,
            cron=cron,
            data={"input": input_text, "session_prefix": session_prefix},
        )
    except DaprJobsError as exc:
        click.echo(f"Error: {exc}")
        click.echo("  Hint: Ensure Dapr sidecar is running: dapr run --app-id nexus ...")
        sys.exit(1)

    click.echo(f"Scheduled job '{name}' created. Cron: {cron}")


@schedule.command("list")
@click.option("--config", "config_path", default="nexus.yaml", show_default=True)
def schedule_list(config_path: str) -> None:
    """List registered scheduled jobs."""
    asyncio.run(_list_async(config_path))


async def _list_async(config_path: str) -> None:
    try:
        load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}")
        hint = getattr(exc, "hint", "")
        if hint:
            click.echo(f"  Hint: {hint}")
        sys.exit(1)

    click.echo(
        "Note: Dapr Jobs API does not support listing. "
        "Use 'nexus schedule get <name>' or check your Dapr dashboard."
    )


@schedule.command("delete")
@click.argument("name")
@click.option("--config", "config_path", default="nexus.yaml", show_default=True)
def schedule_delete(name: str, config_path: str) -> None:
    """Delete a scheduled job."""
    asyncio.run(_delete_async(name, config_path))


async def _delete_async(name: str, config_path: str) -> None:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}")
        hint = getattr(exc, "hint", "")
        if hint:
            click.echo(f"  Hint: {hint}")
        sys.exit(1)

    from nexus.dapr.jobs import DaprJobsClient

    client = DaprJobsClient(host=cfg.dapr.host, port=cfg.dapr.port)
    try:
        deleted = await client.delete(name)
    except DaprJobsError as exc:
        click.echo(f"Error: {exc}")
        sys.exit(1)

    if deleted:
        click.echo(f"Job '{name}' deleted.")
    else:
        click.echo(f"Job '{name}' not found.")
        sys.exit(1)


@schedule.command("trigger")
@click.argument("name")
@click.option("--server", default="http://localhost:8000", show_default=True)
def schedule_trigger(name: str, server: str) -> None:
    """Manually trigger a scheduled job (calls running nexus serve instance)."""
    asyncio.run(_trigger_async(name, server))


async def _trigger_async(name: str, server: str) -> None:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{server}/job/{name}", json={})
        if resp.status_code == 200:
            click.echo(f"Job '{name}' triggered. Response: {resp.json()}")
        else:
            click.echo(f"Error: HTTP {resp.status_code} — {resp.text}")
            sys.exit(1)
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to {server}. Is 'nexus serve' running?")
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}")
        sys.exit(1)
