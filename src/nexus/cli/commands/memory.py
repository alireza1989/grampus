"""nexus memory — inspect and manage agent memory."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from nexus.cli.commands._utils import print_table
from nexus.core.logging import get_logger
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.semantic import SemanticMemory

_log = get_logger(__name__)

_TRUNCATE_LEN = 80


@click.group("memory")
def memory() -> None:
    """Inspect and manage agent memory."""


# ---------------------------------------------------------------------------
# nexus memory inspect
# ---------------------------------------------------------------------------


@memory.command("inspect")
@click.argument("agent_id")
@click.option("--session", default=None, help="Filter to a single session ID.")
@click.option(
    "--type",
    "memory_type",
    type=click.Choice(["episodic", "semantic", "all"]),
    default="all",
    show_default=True,
    help="Memory type to display.",
)
def inspect(agent_id: str, session: str | None, memory_type: str) -> None:
    """List memory records for AGENT_ID."""
    asyncio.run(_inspect_async(agent_id, session=session, memory_type=memory_type))


async def _inspect_async(agent_id: str, *, session: str | None, memory_type: str) -> None:
    """Async implementation of the inspect command."""
    ep_mem = _make_episodic(agent_id)
    sem_mem = _make_semantic(agent_id)

    if memory_type in ("episodic", "all"):
        await _show_episodic(ep_mem, session_filter=session)
    if memory_type in ("semantic", "all"):
        await _show_semantic(sem_mem)


async def _show_episodic(ep_mem: Any, *, session_filter: str | None) -> None:
    """Fetch and print episodic records as a table."""
    records = await ep_mem.list_all()
    if session_filter:
        records = [r for r in records if r.session_id == session_filter]
    rows = [
        [
            r.id,
            r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "episodic",
            r.content[:_TRUNCATE_LEN],
        ]
        for r in records
    ]
    if rows:
        print_table(["ID", "Timestamp", "Type", "Content"], rows, title="Episodic Records")
    else:
        click.echo("No episodic records found.")


async def _show_semantic(sem_mem: Any) -> None:
    """Fetch and print semantic facts as a table."""
    facts = await sem_mem.list_all()
    rows = [
        [
            f.id,
            "—",
            "semantic",
            f"{f.subject} {f.predicate} {f.object_value}"[:_TRUNCATE_LEN],
        ]
        for f in facts
    ]
    if rows:
        print_table(["ID", "Timestamp", "Type", "Content"], rows, title="Semantic Facts")
    else:
        click.echo("No semantic facts found.")


# ---------------------------------------------------------------------------
# nexus memory clear
# ---------------------------------------------------------------------------


@memory.command("clear")
@click.argument("agent_id")
@click.option("--session", default=None, help="Limit deletion to this session ID.")
@click.option(
    "--type",
    "memory_type",
    type=click.Choice(["episodic", "semantic", "all"]),
    default="all",
    show_default=True,
    help="Memory type to clear.",
)
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def clear(agent_id: str, session: str | None, memory_type: str, yes: bool) -> None:
    """Delete memory records for AGENT_ID."""
    if not yes:
        confirmed = click.confirm(
            f"This will delete all {memory_type} memory for agent '{agent_id}'. Continue?",
            default=False,
        )
        if not confirmed:
            return

    asyncio.run(_clear_async(agent_id, session=session, memory_type=memory_type))


async def _clear_async(agent_id: str, *, session: str | None, memory_type: str) -> None:
    """Async implementation of the clear command."""
    ep_mem = _make_episodic(agent_id)
    sem_mem = _make_semantic(agent_id)
    deleted = 0

    if memory_type in ("episodic", "all"):
        deleted += await _delete_episodic(ep_mem, session_filter=session)
    if memory_type in ("semantic", "all"):
        deleted += await _delete_semantic(sem_mem)

    click.echo(f"Deleted {deleted} record(s).")


async def _delete_episodic(ep_mem: Any, *, session_filter: str | None) -> int:
    """Delete episodic records matching the filter, return deleted count."""
    records = await ep_mem.list_all()
    if session_filter:
        records = [r for r in records if r.session_id == session_filter]
    for rec in records:
        await ep_mem.delete(rec.id)
    return len(records)


async def _delete_semantic(sem_mem: Any) -> int:
    """Delete all semantic facts, return deleted count."""
    facts = await sem_mem.list_all()
    for fact in facts:
        await sem_mem.delete(fact.id)
    return len(facts)


# ---------------------------------------------------------------------------
# nexus memory stats
# ---------------------------------------------------------------------------


@memory.command("stats")
@click.argument("agent_id")
def stats(agent_id: str) -> None:
    """Show memory statistics for AGENT_ID."""
    asyncio.run(_stats_async(agent_id))


async def _stats_async(agent_id: str) -> None:
    """Async implementation of the stats command."""
    ep_mem = _make_episodic(agent_id)
    sem_mem = _make_semantic(agent_id)

    records = await ep_mem.list_all()
    facts = await sem_mem.list_all()

    click.echo(f"Agent: {agent_id}")
    click.echo(f"  Episodic records : {len(records)}")
    click.echo(f"  Semantic facts   : {len(facts)}")

    if records:
        oldest = min(r.timestamp for r in records)
        newest = max(r.timestamp for r in records)
        click.echo(f"  Oldest record    : {oldest.strftime('%Y-%m-%d %H:%M UTC')}")
        click.echo(f"  Newest record    : {newest.strftime('%Y-%m-%d %H:%M UTC')}")


# ---------------------------------------------------------------------------
# Internal constructors (patched in tests)
# ---------------------------------------------------------------------------


def _make_episodic(agent_id: str) -> EpisodicMemory:
    """Construct an EpisodicMemory connected to Dapr state.

    Args:
        agent_id: Agent scope for the memory store.

    Returns:
        EpisodicMemory backed by Dapr state store.
    """
    from nexus.core.config import NexusConfig
    from nexus.dapr.client import DaprGateway
    from nexus.dapr.state import DaprStateStore

    cfg = NexusConfig()
    gw = DaprGateway(host=cfg.dapr.host, port=cfg.dapr.grpc_port)
    store = DaprStateStore(gw, cfg.dapr.state_store_name, agent_id)
    return EpisodicMemory(store, None, agent_id=agent_id)


def _make_semantic(agent_id: str) -> SemanticMemory:
    """Construct a SemanticMemory connected to Dapr state.

    Args:
        agent_id: Agent scope for the memory store.

    Returns:
        SemanticMemory backed by Dapr state store.
    """
    from nexus.core.config import NexusConfig
    from nexus.dapr.client import DaprGateway
    from nexus.dapr.state import DaprStateStore

    cfg = NexusConfig()
    gw = DaprGateway(host=cfg.dapr.host, port=cfg.dapr.grpc_port)
    store = DaprStateStore(gw, cfg.dapr.state_store_name, agent_id)
    return SemanticMemory(store, agent_id=agent_id)
