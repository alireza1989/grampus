"""nexus alerts — manage cost alert rules and notification channels."""

from __future__ import annotations

import asyncio

import click

from nexus.cli.commands._utils import print_table
from nexus.observability.alerts import (
    AlertEvent,
    AlertRule,
    AlertSeverity,
    ThresholdType,
)
from nexus.observability.notification import LogChannel, NotificationDispatcher


@click.group("alerts")
def alerts() -> None:
    """Manage cost alert rules and test notification channels."""


@alerts.command("list")
@click.option("--server", default="http://localhost:8000", show_default=True)
def alerts_list(server: str) -> None:
    """List all configured alert rules."""
    asyncio.run(_list_async(server))


async def _list_async(server: str) -> None:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{server}/alerts/rules")
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        return

    rules = data.get("rules", [])
    if not rules:
        click.echo("No alert rules configured.")
        return

    rows = [
        [
            r["rule_id"][:8],
            r["name"],
            r["threshold_type"],
            f"${r['threshold_usd']:.2f}",
            r["severity"],
            "yes" if r["enabled"] else "no",
        ]
        for r in rules
    ]
    print_table(
        ["ID", "Name", "Type", "Threshold", "Severity", "Enabled"],
        rows,
        title=f"Alert Rules ({data.get('count', len(rules))} total)",
    )


@alerts.command("add")
@click.option("--name", required=True, help="Rule name.")
@click.option(
    "--threshold-usd",
    "threshold_usd",
    type=float,
    required=True,
    help="USD threshold value.",
)
@click.option(
    "--threshold-type",
    "threshold_type",
    type=click.Choice([t.value for t in ThresholdType]),
    default=ThresholdType.PER_SESSION_USD.value,
    show_default=True,
)
@click.option(
    "--severity",
    type=click.Choice([s.value for s in AlertSeverity]),
    default=AlertSeverity.WARNING.value,
    show_default=True,
)
@click.option(
    "--agent-id", "agent_id", default=None, help="Restrict to specific agent (omit for all)."
)
@click.option(
    "--cooldown",
    "cooldown_seconds",
    type=int,
    default=3600,
    show_default=True,
    help="Cooldown in seconds between re-fires of the same rule.",
)
@click.option("--server", default="http://localhost:8000", show_default=True)
def alerts_add(
    name: str,
    threshold_usd: float,
    threshold_type: str,
    severity: str,
    agent_id: str | None,
    cooldown_seconds: int,
    server: str,
) -> None:
    """Add a new alert rule."""
    asyncio.run(
        _add_async(
            name, threshold_usd, threshold_type, severity, agent_id, cooldown_seconds, server
        )
    )


async def _add_async(
    name: str,
    threshold_usd: float,
    threshold_type: str,
    severity: str,
    agent_id: str | None,
    cooldown_seconds: int,
    server: str,
) -> None:
    import httpx

    payload: dict[str, object] = {
        "name": name,
        "threshold_usd": threshold_usd,
        "threshold_type": threshold_type,
        "severity": severity,
        "cooldown_seconds": cooldown_seconds,
    }
    if agent_id:
        payload["agent_id"] = agent_id

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{server}/alerts/rules", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        return

    click.echo(f"Created alert rule: {data['rule_id']} ({data['name']})")


@alerts.command("remove")
@click.argument("rule_id")
@click.option("--server", default="http://localhost:8000", show_default=True)
def alerts_remove(rule_id: str, server: str) -> None:
    """Remove an alert rule by ID."""
    asyncio.run(_remove_async(rule_id, server))


async def _remove_async(rule_id: str, server: str) -> None:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{server}/alerts/rules/{rule_id}")
            if resp.status_code == 404:
                click.echo(f"Rule '{rule_id}' not found.", err=True)
                return
            resp.raise_for_status()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        return

    click.echo(f"Removed alert rule: {rule_id}")


@alerts.command("enable")
@click.argument("rule_id")
@click.option("--server", default="http://localhost:8000", show_default=True)
def alerts_enable(rule_id: str, server: str) -> None:
    """Enable an alert rule."""
    asyncio.run(_toggle_async(rule_id, enabled=True, server=server))


@alerts.command("disable")
@click.argument("rule_id")
@click.option("--server", default="http://localhost:8000", show_default=True)
def alerts_disable(rule_id: str, server: str) -> None:
    """Disable an alert rule."""
    asyncio.run(_toggle_async(rule_id, enabled=False, server=server))


async def _toggle_async(rule_id: str, *, enabled: bool, server: str) -> None:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.patch(
                f"{server}/alerts/rules/{rule_id}",
                json={"enabled": enabled},
            )
            if resp.status_code == 404:
                click.echo(f"Rule '{rule_id}' not found.", err=True)
                return
            resp.raise_for_status()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        return

    state = "enabled" if enabled else "disabled"
    click.echo(f"Alert rule {rule_id} {state}.")


@alerts.command("test")
@click.argument("rule_id")
@click.option("--server", default="http://localhost:8000", show_default=True)
def alerts_test(rule_id: str, server: str) -> None:
    """Fire a synthetic AlertEvent through the dispatcher to verify channels."""
    asyncio.run(_test_async(rule_id, server))


async def _test_async(rule_id: str, server: str) -> None:
    import httpx

    # Fetch the rule first
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{server}/alerts/rules/{rule_id}")
            if resp.status_code == 404:
                click.echo(f"Rule '{rule_id}' not found.", err=True)
                return
            resp.raise_for_status()
            rule_data = resp.json()
    except Exception as exc:
        click.echo(f"Error fetching rule: {exc}", err=True)
        return

    rule = AlertRule(**rule_data)
    disp = NotificationDispatcher(channels=[LogChannel()])

    synthetic_event = AlertEvent(
        rule_id=rule.rule_id,
        rule_name=rule.name,
        agent_id=rule.agent_id or "test-agent",
        session_id="test-session",
        severity=rule.severity,
        threshold_type=rule.threshold_type,
        threshold_usd=rule.threshold_usd,
        actual_usd=rule.threshold_usd * 1.5,
        message=f"[TEST] Rule '{rule.name}' synthetic fire — actual ${rule.threshold_usd * 1.5:.2f}",
    )
    await disp.dispatch(synthetic_event)
    click.echo(f"Synthetic alert fired for rule: {rule.name}")
    click.echo(f"  Message: {synthetic_event.message}")
