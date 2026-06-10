"""nexus version — agent version lifecycle, A/B testing, rollback."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from nexus.cli.commands._utils import print_table
from nexus.core.errors import VersioningError
from nexus.core.logging import get_logger

_log = get_logger(__name__)

_STATE_STORE_URL_OPT = click.option(
    "--state-store-url",
    default="http://localhost:3500",
    envvar="DAPR_HTTP_URL",
    show_default=True,
    help="Dapr sidecar HTTP URL.",
)


def _build_version_store(state_store_url: str) -> Any:
    """Construct a VersionStore backed by Dapr. Overridable in tests."""
    from nexus.dapr.client import DaprGateway
    from nexus.dapr.state import DaprStateStore
    from nexus.versioning.store import VersionStore

    host, _, port_str = state_store_url.replace("http://", "").partition(":")
    port = int(port_str) if port_str else 3500
    gw = DaprGateway(host=host, port=port)
    state = DaprStateStore(gw, "statestore", "versioning")
    return VersionStore(state)


def _build_version_manager(state_store_url: str, agent_id: str) -> Any:
    """Construct a VersionManager. Overridable in tests via patch."""
    from nexus.versioning.manager import VersionManager

    store = _build_version_store(state_store_url)
    return VersionManager(store, agent_id=agent_id)


def _build_ab_manager(state_store_url: str) -> Any:
    """Construct an ABTestManager. Overridable in tests via patch."""
    from nexus.versioning.ab_testing import ABTestManager
    from nexus.versioning.metrics import QualityTracker

    store = _build_version_store(state_store_url)
    tracker = QualityTracker(store._state)
    return ABTestManager(store, tracker)


@click.group("version")
def version_group() -> None:
    """Manage agent version lifecycle, A/B tests, and rollback."""


@version_group.command("list")
@click.argument("agent_id")
@_STATE_STORE_URL_OPT
def version_list(agent_id: str, state_store_url: str) -> None:
    """List all versions for AGENT_ID, newest first."""
    mgr = _build_version_manager(state_store_url, agent_id)
    try:
        versions = asyncio.run(mgr.list_versions())
    except VersioningError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    if not versions:
        click.echo(f"No versions found for agent '{agent_id}'.")
        return

    rows = []
    for v in versions:
        rows.append(
            [
                v.version_id[:12],
                v.version_tag,
                str(v.status),
                v.created_at.strftime("%Y-%m-%d %H:%M"),
                v.description[:40] if v.description else "",
            ]
        )
    print_table(
        ["VERSION ID", "TAG", "STATUS", "CREATED", "DESCRIPTION"],
        rows,
        title=f"Versions for {agent_id}",
    )


@version_group.command("tag")
@click.argument("version_id")
@click.option("--tag", required=True, help="New version tag label.")
@click.option("--agent", "agent_id", required=True, help="Agent ID owning the version.")
@_STATE_STORE_URL_OPT
def version_tag(version_id: str, tag: str, agent_id: str, state_store_url: str) -> None:
    """Update the human-readable tag label for a version."""
    store = _build_version_store(state_store_url)

    async def _run() -> None:
        version = await store.get_version(version_id)
        if version is None:
            click.echo(f"Error: version '{version_id}' not found.", err=True)
            raise SystemExit(1)
        updated = version.model_copy(update={"version_tag": tag})
        await store.save_version(updated)
        click.echo(f"Tagged {version_id[:12]} as '{tag}'")

    asyncio.run(_run())


@version_group.command("deploy")
@click.argument("version_id")
@click.option("--agent", "agent_id", required=True, help="Agent ID to deploy to.")
@click.option("--by", default="cli", show_default=True, help="Who is deploying.")
@_STATE_STORE_URL_OPT
def version_deploy(version_id: str, agent_id: str, by: str, state_store_url: str) -> None:
    """Deploy VERSION_ID as the active version for an agent."""
    mgr = _build_version_manager(state_store_url, agent_id)
    try:
        record = asyncio.run(mgr.deploy(version_id, deployed_by=by))
    except VersioningError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo(f"Deployed {record.version_id[:12]} to {agent_id}")


@version_group.command("rollback")
@click.argument("agent_id")
@_STATE_STORE_URL_OPT
def version_rollback(agent_id: str, state_store_url: str) -> None:
    """Revert AGENT_ID to its previous deployed version."""
    mgr = _build_version_manager(state_store_url, agent_id)
    try:
        record = asyncio.run(mgr.rollback())
    except VersioningError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo(f"Rolled back {agent_id} to {record.version_id[:12]}")


@version_group.command("diff")
@click.argument("version_id_a")
@click.argument("version_id_b")
@click.option("--agent", "agent_id", required=True, help="Agent ID owning the versions.")
@_STATE_STORE_URL_OPT
def version_diff(version_id_a: str, version_id_b: str, agent_id: str, state_store_url: str) -> None:
    """Show the diff between two versions of an agent."""
    mgr = _build_version_manager(state_store_url, agent_id)
    try:
        diff = asyncio.run(mgr.diff(version_id_a, version_id_b))
    except VersioningError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    if not diff.has_changes:
        click.echo("No changes between the two versions.")
        return

    if diff.system_prompt_diff:
        click.echo("--- System prompt diff ---")
        click.echo(diff.system_prompt_diff)

    if diff.tools_added:
        click.echo(f"Tools added: {', '.join(diff.tools_added)}")
    if diff.tools_removed:
        click.echo(f"Tools removed: {', '.join(diff.tools_removed)}")

    for field, (old_val, new_val) in diff.config_changes.items():
        click.echo(f"{field}: {old_val} -> {new_val}")


@version_group.command("ab-start")
@click.argument("agent_id")
@click.option("--control", required=True, help="Control version ID.")
@click.option("--treatment", required=True, help="Treatment version ID.")
@click.option("--split", default=0.1, show_default=True, help="Fraction going to treatment.")
@click.option(
    "--metric",
    type=click.Choice(
        ["eval_pass_rate", "avg_cost_usd", "avg_latency_seconds", "error_rate"],
        case_sensitive=False,
    ),
    default="eval_pass_rate",
    show_default=True,
    help="Success metric for auto-promotion.",
)
@click.option("--min-samples", default=100, show_default=True, help="Min runs before promotion.")
@_STATE_STORE_URL_OPT
def version_ab_start(
    agent_id: str,
    control: str,
    treatment: str,
    split: float,
    metric: str,
    min_samples: int,
    state_store_url: str,
) -> None:
    """Start an A/B experiment for AGENT_ID."""
    from nexus.versioning.types import SuccessMetric

    ab_mgr = _build_ab_manager(state_store_url)
    try:
        cfg = asyncio.run(
            ab_mgr.start_test(
                agent_id,
                control_version_id=control,
                treatment_version_id=treatment,
                traffic_split=split,
                success_metric=SuccessMetric(metric),
                min_samples=min_samples,
            )
        )
    except VersioningError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo(f"Started experiment {cfg.experiment_id}")


@version_group.command("ab-status")
@click.argument("experiment_id")
@_STATE_STORE_URL_OPT
def version_ab_status(experiment_id: str, state_store_url: str) -> None:
    """Show current metrics for an A/B experiment."""
    ab_mgr = _build_ab_manager(state_store_url)
    try:
        result = asyncio.run(ab_mgr.evaluate(experiment_id))
    except VersioningError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    ctrl = result.control_metrics
    trt = result.treatment_metrics

    def _fmt_rate(r: float | None) -> str:
        return f"{r * 100:.1f}%" if r is not None else "n/a"

    click.echo(
        f"Control   {ctrl.version_id[:12]}: "
        f"pass_rate={_fmt_rate(ctrl.eval_pass_rate)}, runs={ctrl.total_runs}"
    )
    click.echo(
        f"Treatment {trt.version_id[:12]}: "
        f"pass_rate={_fmt_rate(trt.eval_pass_rate)}, runs={trt.total_runs}"
    )

    if result.p_value is not None:
        sig = "significant" if result.significant else "not significant"
        click.echo(f"p-value: {result.p_value:.3f}  [{sig}]")
    else:
        sig_label = "significant (>10% diff)" if result.significant else "not significant"
        click.echo(f"p-value: n/a  [{sig_label}]")

    click.echo(f"Recommendation: {result.recommendation}")


@version_group.command("ab-stop")
@click.argument("experiment_id")
@_STATE_STORE_URL_OPT
def version_ab_stop(experiment_id: str, state_store_url: str) -> None:
    """Stop an active A/B experiment."""
    ab_mgr = _build_ab_manager(state_store_url)
    try:
        asyncio.run(ab_mgr.stop_test(experiment_id))
    except VersioningError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo(f"Stopped experiment {experiment_id}")
