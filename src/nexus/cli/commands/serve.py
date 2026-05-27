"""nexus serve — serve a Nexus agent as a REST API."""

from __future__ import annotations

import sys
from typing import Any

import click

from nexus.cli.commands._utils import load_config, load_module, require_function
from nexus.core.errors import ConfigError
from nexus.core.types import AgentDefinition


def _print_nexus_error(exc: Exception) -> None:
    click.echo(f"Error: {exc}", err=False)
    hint = getattr(exc, "hint", "")
    if hint:
        click.echo(f"  Hint: {hint}", err=False)


def _resolve_agent_def(module: Any, cfg: Any) -> AgentDefinition:
    factory = getattr(module, "create_agent_def", None)
    if factory is not None:
        result: AgentDefinition = factory()
        return result
    agent_cfg = getattr(cfg, "agent", None)
    if agent_cfg is not None:
        return AgentDefinition(
            name=getattr(agent_cfg, "name", "nexus-agent"),
            model=getattr(agent_cfg, "model", cfg.model.default_model),
            system_prompt=getattr(agent_cfg, "system_prompt", None),
            max_iterations=getattr(agent_cfg, "max_iterations", 10),
            memory_enabled=getattr(agent_cfg, "memory_enabled", True),
            cost_budget_usd=getattr(agent_cfg, "cost_budget_usd", None),
        )
    return AgentDefinition(name="nexus-agent", model=cfg.model.default_model)


@click.command("serve")
@click.argument("agent_file")
@click.option("--config", "config_path", default="nexus.yaml", show_default=True)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, default=False, help="Auto-reload on file changes.")
@click.option("--workers", default=1, show_default=True, type=int)
def serve(
    agent_file: str,
    config_path: str,
    host: str,
    port: int,
    reload: bool,
    workers: int,
) -> None:
    """Serve a Nexus agent as a REST API."""
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "Error: uvicorn is not installed. Install server deps: pip install nexus-ai[server]"
        )
        sys.exit(1)

    try:
        cfg = load_config(config_path)
        module = load_module(agent_file)
    except ConfigError as exc:
        _print_nexus_error(exc)
        sys.exit(1)

    try:
        factory = require_function(module, "create_runner")
        runner = factory()
    except ConfigError as exc:
        _print_nexus_error(exc)
        sys.exit(1)

    agent_def = _resolve_agent_def(module, cfg)

    from nexus.server.app import create_app

    app = create_app(runner, agent_def)

    click.echo(f"Nexus agent '{agent_def.name}' serving at http://{host}:{port}")
    click.echo(f"  Human-in-the-loop UI: http://{host}:{port}/ui")
    click.echo(f"  Execution trace viewer: http://{host}:{port}/trace")
    uvicorn.run(app, host=host, port=port, reload=reload, workers=workers)
