"""nexus redteam — run an adversarial red-team campaign against an agent."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

import click

from nexus.cli.commands._utils import load_module
from nexus.core.logging import get_logger

_log = get_logger(__name__)


@click.command("redteam")
@click.argument("agent_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--categories",
    "-c",
    multiple=True,
    help=(
        "Attack categories to run (default: all). "
        "Choices: prompt_injection, jailbreak, reasoning_hijack, "
        "memory_poison, tool_misuse, excessive_agency"
    ),
)
@click.option("--count", "-n", default=5, show_default=True, help="Payloads per strategy")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Report format",
)
@click.option(
    "--stop-on-critical",
    is_flag=True,
    default=False,
    help="Stop campaign after first CRITICAL finding",
)
@click.option(
    "--model",
    default=None,
    help="Model ID for LLM-based judge and attacker (e.g. claude-sonnet-4-6)",
)
def redteam_command(
    agent_file: Path,
    categories: tuple[str, ...],
    count: int,
    output: str,
    stop_on_critical: bool,
    model: str | None,
) -> None:
    """Run an adversarial red-team campaign against AGENT_FILE.

    AGENT_FILE must expose a `get_agent_config()` function returning a
    RedTeamTargetConfig, and a `run_conversation(messages)` async function
    accepting list[tuple[str, str]] and returning str.

    \b
    Example:
        nexus redteam my_agent.py --categories prompt_injection jailbreak --count 10
    """
    asyncio.run(_run(agent_file, categories, count, output, stop_on_critical, model))


async def _run(
    agent_file: Path,
    categories: tuple[str, ...],
    count: int,
    output: str,
    stop_on_critical: bool,
    model_id: str | None,
) -> None:
    from nexus.evaluation.red_team.attacker import AttackerAgent  # noqa: PLC0415
    from nexus.evaluation.red_team.judge import RedTeamJudge  # noqa: PLC0415
    from nexus.evaluation.red_team.report import RedTeamReport  # noqa: PLC0415
    from nexus.evaluation.red_team.runner import RedTeamRunner  # noqa: PLC0415
    from nexus.evaluation.red_team.types import (  # noqa: PLC0415
        AttackCategory,
        RedTeamCampaignConfig,
    )

    mod = load_module(str(agent_file))

    if not hasattr(mod, "get_agent_config") or not hasattr(mod, "run_conversation"):
        click.echo(
            "Error: agent file must expose get_agent_config() and run_conversation(messages).",
            err=True,
        )
        sys.exit(1)

    target_config = mod.get_agent_config()
    target_fn = mod.run_conversation

    model_client: Any = None
    if model_id:
        try:
            from nexus.cli.playground.model_factory import make_client  # noqa: PLC0415
            from nexus.core.config import NexusConfig  # noqa: PLC0415

            model_client = make_client(model_id, NexusConfig())
        except Exception:
            click.echo(
                f"Warning: could not load model {model_id!r}, using rule-based judge only.",
                err=True,
            )

    enabled_categories = (
        [AttackCategory(c) for c in categories] if categories else list(AttackCategory)
    )

    campaign_id = str(uuid.uuid4())[:8]
    config = RedTeamCampaignConfig(
        campaign_id=campaign_id,
        target=target_config,
        enabled_categories=enabled_categories,
        payloads_per_strategy=count,
        stop_on_critical=stop_on_critical,
    )

    attacker = AttackerAgent(model_client=model_client)
    judge = RedTeamJudge(model_client=model_client)
    runner = RedTeamRunner(attacker=attacker, judge=judge, target_fn=target_fn)

    click.echo(f"Starting red-team campaign {campaign_id} against {target_config.agent_name}...")
    results = await runner.run(config)

    report = RedTeamReport()
    summary = report.build(config, results)

    if output == "json":
        click.echo(report.to_json(summary))
    else:
        click.echo(report.to_text(summary))

    critical_count = summary.severity_counts.get("critical", 0)
    high_count = summary.severity_counts.get("high", 0)
    if critical_count + high_count > 0:
        sys.exit(1)
