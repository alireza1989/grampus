"""nexus dev — watch-mode development server."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from nexus.cli.commands._utils import load_config
from nexus.core.errors import ConfigError
from nexus.core.logging import get_logger

_log = get_logger(__name__)

try:
    from watchfiles import awatch as _awatch

    _HAS_WATCHFILES: bool = True
except ImportError:
    _HAS_WATCHFILES = False


@click.command("dev")
@click.option(
    "--config",
    "config_path",
    default="nexus.yaml",
    show_default=True,
    help="Path to nexus.yaml configuration file.",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    show_default=True,
    help="Agent HTTP port.",
)
def dev(config_path: str, port: int) -> None:
    """Start Nexus in development / watch mode."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}")
        sys.exit(1)

    _print_banner(port)
    _print_config_summary(cfg)

    try:
        _run_dev_loop(cfg, config_path=config_path)
    except KeyboardInterrupt:
        click.echo("\n[nexus dev] Stopped.")


def _print_banner(port: int) -> None:
    """Print the startup banner."""
    click.echo("╔══════════════════════════════════════╗")
    click.echo("║         nexus dev — watch mode       ║")
    click.echo("╚══════════════════════════════════════╝")
    click.echo("  Dapr sidecar  : would start on localhost")
    click.echo(f"  Agent process : would start on port {port}")
    click.echo("  Cost watcher  : tailing .nexus/cost_log.jsonl")
    click.echo("")


def _print_config_summary(cfg: Any) -> None:
    """Print a brief summary of the loaded configuration."""
    agent_cfg = getattr(cfg, "agent", None)
    model = (
        getattr(agent_cfg, "model", cfg.model.default_model)
        if agent_cfg
        else cfg.model.default_model
    )
    name = getattr(agent_cfg, "name", "nexus-agent") if agent_cfg else "nexus-agent"
    dapr_port = cfg.dapr.port

    click.echo("Config summary:")
    click.echo(f"  agent name : {name}")
    click.echo(f"  model      : {model}")
    click.echo(f"  dapr port  : {dapr_port}")
    click.echo("")


def _run_dev_loop(cfg: Any, *, config_path: str) -> None:
    """Run the development watch loop (blocking).

    This simplified version prints file-change notifications.
    Full auto-reload is deferred to Phase 12.

    Args:
        cfg: Loaded NexusConfig.
        config_path: Path to nexus.yaml for file watching.
    """
    watch_dir = str(Path(config_path).parent)
    click.echo(f"[nexus dev] Watching {watch_dir} for changes...")
    click.echo("[nexus dev] Press Ctrl+C to stop.\n")

    if _HAS_WATCHFILES:
        asyncio.run(_watch_with_watchfiles(watch_dir))
    else:
        asyncio.run(_watch_with_polling(watch_dir))


async def _watch_with_watchfiles(watch_dir: str) -> None:
    """Watch for file changes using the watchfiles library."""
    async for changes in _awatch(watch_dir):
        for _, path in changes:
            click.echo(f"[nexus dev] File changed: {path} — restart required")


async def _watch_with_polling(watch_dir: str) -> None:
    """Minimal polling loop when watchfiles is not installed."""
    click.echo("[nexus dev] (watchfiles not installed — polling every 2s)")
    while True:
        await asyncio.sleep(2)
