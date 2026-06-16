"""Nexus CLI entry point."""

from __future__ import annotations

import click

from grampus.cli.commands.alerts import alerts
from grampus.cli.commands.cost import cost
from grampus.cli.commands.dev import dev
from grampus.cli.commands.eval import eval_cmd
from grampus.cli.commands.hub import hub
from grampus.cli.commands.init import init
from grampus.cli.commands.memory import memory
from grampus.cli.commands.playground import playground
from grampus.cli.commands.redteam import redteam_command
from grampus.cli.commands.replay import replay
from grampus.cli.commands.run import run
from grampus.cli.commands.schedule import schedule
from grampus.cli.commands.serve import serve
from grampus.cli.commands.state import state
from grampus.cli.commands.version import version_group


@click.group()
@click.version_option()
def cli() -> None:
    """Grampus — production-grade agentic AI framework."""


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
