"""grampus state — export and restore agent state snapshots."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from grampus.cli.commands._utils import load_config
from grampus.core.errors import ConfigError, SnapshotError
from grampus.orchestration.snapshot import SnapshotManager, StateSnapshot


@click.group("state")
def state() -> None:
    """Export, import, and inspect agent state snapshots."""


# ---------------------------------------------------------------------------
# grampus state export
# ---------------------------------------------------------------------------


@state.command("export")
@click.argument("agent_id")
@click.argument("session_id")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path (default: <agent_id>_<session_id>_snapshot.json).",
)
@click.option("--description", "-d", default="", help="Human-readable description.")
@click.option("--tag", multiple=True, help="Repeatable tag (e.g. --tag production).")
@click.option("--environment", default="", help="Source environment label.")
@click.option("--config", default="grampus.yaml", help="Path to grampus.yaml.")
def export_cmd(
    agent_id: str,
    session_id: str,
    output: str | None,
    description: str,
    tag: tuple[str, ...],
    environment: str,
    config: str,
) -> None:
    """Export a live agent session to a snapshot file."""
    asyncio.run(
        _export_async(
            agent_id=agent_id,
            session_id=session_id,
            output=output,
            description=description,
            tags=list(tag),
            environment=environment,
            config_path=config,
        )
    )


async def _export_async(
    *,
    agent_id: str,
    session_id: str,
    output: str | None,
    description: str,
    tags: list[str],
    environment: str,
    config_path: str,
) -> None:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        if exc.hint:
            click.echo(f"Hint: {exc.hint}", err=True)
        sys.exit(1)

    try:
        from grampus.dapr.client import DaprGateway
        from grampus.dapr.state import DaprStateStore

        dapr_cfg = cfg.dapr
        gateway = DaprGateway(
            host=dapr_cfg.host,
            port=dapr_cfg.grpc_port,
        )
        store = DaprStateStore(
            gateway=gateway,
            store_name=dapr_cfg.state_store_name,
            namespace="grampus",
        )
        mgr = SnapshotManager(state_store=store)
        snap = await mgr.export_session(
            agent_id,
            session_id,
            description=description,
            tags=tags,
            source_environment=environment,
        )
    except SnapshotError as exc:
        click.echo(f"Error: {exc}", err=True)
        if exc.hint:
            click.echo(f"Hint: {exc.hint}", err=True)
        sys.exit(1)

    out_path = Path(output) if output else Path(f"{agent_id}_{session_id}_snapshot.json")
    SnapshotManager.to_file(snap, out_path)
    click.echo(
        f"Snapshot saved: {out_path} "
        f"(session: {session_id}, steps: {snap.state.current_step}, status: {snap.state.status})"
    )


# ---------------------------------------------------------------------------
# grampus state import
# ---------------------------------------------------------------------------


@state.command("import")
@click.argument("snapshot_file", type=click.Path(exists=False))
@click.option("--session-id", "-s", default=None, help="Override the session ID.")
@click.option("--config", default="grampus.yaml", help="Path to grampus.yaml.")
@click.option("--dry-run", is_flag=True, help="Validate and show without writing.")
def import_cmd(
    snapshot_file: str,
    session_id: str | None,
    config: str,
    dry_run: bool,
) -> None:
    """Restore an agent session from a snapshot file."""
    try:
        snap = SnapshotManager.from_file(Path(snapshot_file))
    except SnapshotError as exc:
        click.echo(f"Error: {exc}", err=True)
        if exc.hint:
            click.echo(f"Hint: {exc.hint}", err=True)
        sys.exit(1)

    if dry_run:
        _print_snapshot_summary(snap, session_id_override=session_id)
        return

    asyncio.run(
        _import_async(
            snap=snap,
            session_id_override=session_id,
            config_path=config,
        )
    )


async def _import_async(
    *,
    snap: StateSnapshot,
    session_id_override: str | None,
    config_path: str,
) -> None:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        if exc.hint:
            click.echo(f"Hint: {exc.hint}", err=True)
        sys.exit(1)

    try:
        from grampus.dapr.client import DaprGateway
        from grampus.dapr.state import DaprStateStore

        dapr_cfg = cfg.dapr
        gateway = DaprGateway(
            host=dapr_cfg.host,
            port=dapr_cfg.grpc_port,
        )
        store = DaprStateStore(
            gateway=gateway,
            store_name=dapr_cfg.state_store_name,
            namespace="grampus",
        )
        mgr = SnapshotManager(state_store=store)
        await mgr.restore_snapshot(snap, session_id_override=session_id_override)
    except SnapshotError as exc:
        click.echo(f"Error: {exc}", err=True)
        if exc.hint:
            click.echo(f"Hint: {exc.hint}", err=True)
        sys.exit(1)

    target = session_id_override or snap.session_id
    click.echo(f"Restored snapshot for agent='{snap.agent_id}' → session='{target}'")


# ---------------------------------------------------------------------------
# grampus state show
# ---------------------------------------------------------------------------


@state.command("show")
@click.argument("snapshot_file", type=click.Path(exists=False))
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def show_cmd(snapshot_file: str, fmt: str) -> None:
    """Display a snapshot file without connecting to Dapr."""
    try:
        snap = SnapshotManager.from_file(Path(snapshot_file))
    except SnapshotError as exc:
        click.echo(f"Error: {exc}", err=True)
        if exc.hint:
            click.echo(f"Hint: {exc.hint}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(snap.model_dump_json(indent=2))
        return

    _print_snapshot_summary(snap)
    _print_recent_messages(snap)


def _print_snapshot_summary(
    snap: StateSnapshot,
    *,
    session_id_override: str | None = None,
) -> None:
    target_session = session_id_override or snap.session_id
    msg_count = len(snap.state.messages)
    tags_str = ", ".join(snap.tags) if snap.tags else "(none)"
    created_str = snap.created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        ("Snapshot ID:", snap.snapshot_id),
        ("Schema version:", snap.schema_version),
        ("Grampus version:", snap.grampus_version),
        ("Created at:", created_str),
        ("Agent ID:", snap.agent_id),
        ("Session ID:", target_session),
        ("Status:", str(snap.state.status)),
        ("Steps:", str(snap.state.current_step)),
        ("Messages:", str(msg_count)),
        ("Description:", snap.description or "(none)"),
        ("Tags:", tags_str),
    ]
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        click.echo(f"{label:<{label_width}} {value}")


def _print_recent_messages(snap: StateSnapshot, *, n: int = 3, max_chars: int = 120) -> None:
    messages = snap.state.messages[-n:] if snap.state.messages else []
    if not messages:
        return
    click.echo("")
    click.echo(f"Last {len(messages)} message(s):")
    for msg in messages:
        content = str(msg.content) if msg.content else ""
        truncated = content[:max_chars] + ("…" if len(content) > max_chars else "")
        click.echo(f"  [{msg.role}] {truncated}")
