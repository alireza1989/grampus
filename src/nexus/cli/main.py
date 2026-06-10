"""Nexus CLI entry point."""

from __future__ import annotations

import click

from nexus.cli.commands.alerts import alerts
from nexus.cli.commands.cost import cost
from nexus.cli.commands.dev import dev
from nexus.cli.commands.eval import eval_cmd
from nexus.cli.commands.hub import hub
from nexus.cli.commands.init import init
from nexus.cli.commands.memory import memory
from nexus.cli.commands.playground import playground
from nexus.cli.commands.redteam import redteam_command
from nexus.cli.commands.replay import replay
from nexus.cli.commands.run import run
from nexus.cli.commands.schedule import schedule
from nexus.cli.commands.serve import serve
from nexus.cli.commands.state import state
from nexus.cli.commands.version import version_group


@click.group()
@click.version_option()
def cli() -> None:
    """Nexus — production-grade agentic AI framework."""


cli.add_command(alerts)
cli.add_command(hub)
cli.add_command(init)
cli.add_command(run)
cli.add_command(eval_cmd, name="eval")
cli.add_command(memory)
cli.add_command(cost)
cli.add_command(dev)
cli.add_command(serve)
cli.add_command(replay)
cli.add_command(schedule)
cli.add_command(state)
cli.add_command(playground)
cli.add_command(redteam_command, name="redteam")
cli.add_command(version_group, name="version")
