"""Nexus CLI entry point."""

from __future__ import annotations

import click

from nexus.cli.commands.cost import cost
from nexus.cli.commands.dev import dev
from nexus.cli.commands.eval import eval_cmd
from nexus.cli.commands.init import init
from nexus.cli.commands.memory import memory
from nexus.cli.commands.replay import replay
from nexus.cli.commands.run import run
from nexus.cli.commands.schedule import schedule
from nexus.cli.commands.serve import serve


@click.group()
@click.version_option()
def cli() -> None:
    """Nexus — production-grade agentic AI framework."""


cli.add_command(init)
cli.add_command(run)
cli.add_command(eval_cmd, name="eval")
cli.add_command(memory)
cli.add_command(cost)
cli.add_command(dev)
cli.add_command(serve)
cli.add_command(replay)
cli.add_command(schedule)
