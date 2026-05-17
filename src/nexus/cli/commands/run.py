"""nexus run — execute an agent from a Python file."""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any

import click

from nexus.cli.commands._utils import load_config, load_module, require_function
from nexus.core.errors import ConfigError
from nexus.core.logging import get_logger
from nexus.core.types import AgentDefinition, ExecutionResult, StreamEventType

_log = get_logger(__name__)


@click.command("run")
@click.argument("agent_file")
@click.option(
    "--config",
    "config_path",
    default="nexus.yaml",
    show_default=True,
    help="Path to nexus.yaml configuration file.",
)
@click.option("--session-id", default=None, help="Session identifier (auto-generated if omitted).")
@click.option(
    "--input",
    "input_text",
    default=None,
    help="Input text for a single-shot run. Omit to enter interactive REPL.",
)
@click.option(
    "--stream/--no-stream",
    default=False,
    show_default=True,
    help="Stream output token-by-token instead of waiting for full response.",
)
def run(
    agent_file: str,
    config_path: str,
    session_id: str | None,
    input_text: str | None,
    stream: bool,
) -> None:
    """Run a Nexus agent defined in AGENT_FILE."""
    try:
        cfg = load_config(config_path)
        module = load_module(agent_file)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=False)
        sys.exit(1)

    try:
        runner = _build_runner(module, cfg)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=False)
        sys.exit(1)

    agent_def = _resolve_agent_def(module, cfg)
    sid = session_id or f"nexus-{uuid.uuid4().hex[:8]}"

    if input_text is not None:
        if stream:
            _run_once_streaming(runner, agent_def, input_text, sid)
        else:
            _run_once(runner, agent_def, input_text, sid)
    else:
        _run_repl(runner, agent_def, sid)


def _build_runner(module: Any, cfg: Any) -> Any:
    """Load create_runner() from the module and call it.

    Args:
        module: The dynamically loaded agent module.
        cfg: Loaded NexusConfig (reserved for future dependency injection).

    Returns:
        AgentRunner instance returned by create_runner().

    Raises:
        ConfigError: When create_runner is not defined in the module.
    """
    factory = require_function(module, "create_runner")
    return factory()


def _resolve_agent_def(module: Any, cfg: Any) -> AgentDefinition:
    """Resolve the AgentDefinition from the module or fall back to nexus.yaml.

    Args:
        module: Dynamically loaded agent module.
        cfg: Loaded NexusConfig used as fallback.

    Returns:
        AgentDefinition instance.
    """
    factory = getattr(module, "create_agent_def", None)
    if factory is not None:
        result: AgentDefinition = factory()
        return result
    return _agent_def_from_config(cfg)


def _agent_def_from_config(cfg: Any) -> AgentDefinition:
    """Build a minimal AgentDefinition from NexusConfig.

    Args:
        cfg: NexusConfig with agent sub-settings stored in a nexus.yaml file.

    Returns:
        AgentDefinition populated from config defaults.
    """
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


def _run_once(runner: Any, agent_def: AgentDefinition, input_text: str, session_id: str) -> None:
    """Execute one agent turn and print the result.

    Args:
        runner: AgentRunner to invoke.
        agent_def: Agent definition.
        input_text: User input for this turn.
        session_id: Session identifier.
    """
    result: ExecutionResult = asyncio.run(runner.run(agent_def, input_text, session_id=session_id))
    _print_result(result)


def _run_repl(runner: Any, agent_def: AgentDefinition, session_id: str) -> None:
    """Enter an interactive REPL loop.

    Args:
        runner: AgentRunner to invoke for each turn.
        agent_def: Agent definition.
        session_id: Session identifier shared across turns.
    """
    click.echo("Nexus interactive session. Press Ctrl+C to exit.\n")
    try:
        while True:
            user_input = click.prompt("You")
            result: ExecutionResult = asyncio.run(
                runner.run(agent_def, user_input, session_id=session_id)
            )
            _print_result(result)
    except (KeyboardInterrupt, EOFError):
        click.echo("\nGoodbye.")


def _print_result(result: ExecutionResult) -> None:
    """Print agent output and token usage to stdout.

    Args:
        result: ExecutionResult from the agent run.
    """
    click.echo(f"\nAgent: {result.output or '(no output)'}\n")
    usage = result.token_usage
    click.echo(
        f"  tokens: {usage.total_tokens:,}  "
        f"(in={usage.input_tokens:,} out={usage.output_tokens:,})  "
        f"cost: ${usage.cost_usd:.4f}"
    )


def _run_once_streaming(
    runner: Any, agent_def: AgentDefinition, input_text: str, session_id: str
) -> None:
    """Execute one agent turn in streaming mode, printing events as they arrive.

    Args:
        runner: AgentRunner to invoke.
        agent_def: Agent definition.
        input_text: User input for this turn.
        session_id: Session identifier.
    """
    asyncio.run(_stream_async(runner, agent_def, input_text, session_id))


async def _stream_async(
    runner: Any, agent_def: AgentDefinition, input_text: str, session_id: str
) -> None:
    use_color = sys.stdout.isatty()

    async for event in runner.stream(agent_def, input_text, session_id=session_id):
        if event.event_type == StreamEventType.TOKEN:
            if event.chunk and event.chunk.delta:
                click.echo(event.chunk.delta, nl=False)
                sys.stdout.flush()

        elif event.event_type == StreamEventType.TOOL_CALL_START:
            if event.tool_call:
                args_summary = str(event.tool_call.arguments)[:60]
                label = f"\n[→ {event.tool_call.name}({args_summary})]\n"
                click.echo(click.style(label, dim=True) if use_color else label, nl=False)

        elif event.event_type == StreamEventType.TOOL_CALL_END:
            if event.tool_call and event.tool_result:
                result_preview = str(event.tool_result.output or "")[:60]
                label = f"[← {event.tool_call.name}: {result_preview}]\n"
                click.echo(click.style(label, dim=True) if use_color else label, nl=False)

        elif event.event_type == StreamEventType.AGENT_END:
            click.echo()
            if event.chunk and event.chunk.token_usage:
                usage = event.chunk.token_usage
                click.echo(
                    f"\n  tokens: {usage.total_tokens:,}  "
                    f"(in={usage.input_tokens:,} out={usage.output_tokens:,})  "
                    f"cost: ${usage.cost_usd:.4f}"
                )

        elif event.event_type == StreamEventType.ERROR:
            click.echo(f"\nError: {event.message}", err=True)
            sys.exit(1)
