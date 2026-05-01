"""Nexus CLI entry point.

Full command implementations are added in Phase 11. This stub registers the
root Click group so the `nexus` script entry point resolves correctly.
"""

import click


@click.group()
def cli() -> None:
    """Nexus — Production-grade agentic AI framework."""
