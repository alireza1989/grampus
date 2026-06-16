"""Shared utilities for Nexus CLI commands."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import click

from grampus.core.errors import ConfigError
from grampus.core.logging import get_logger

_log = get_logger(__name__)


def load_config(path: str) -> Any:
    """Load grampus.yaml and return GrampusConfig.

    Args:
        path: Path to the grampus.yaml configuration file.

    Returns:
        GrampusConfig instance loaded from the given file.

    Raises:
        ConfigError: If the file does not exist or cannot be parsed.
    """
    from grampus.core.config import GrampusConfig

    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"Config file not found: {path}",
            code="CONFIG_NOT_FOUND",
            hint="Run 'grampus init' to create a grampus.yaml, or pass --config <path>.",
        )
    try:
        return GrampusConfig(_config_file=str(p))
    except Exception as exc:
        raise ConfigError(
            f"Failed to load config from {path}: {exc}",
            code="CONFIG_PARSE_ERROR",
            hint="Run 'grampus init' to create a grampus.yaml, or pass --config <path>.",
        ) from exc


def load_module(path: str) -> ModuleType:
    """Dynamically import a Python file by path.

    Args:
        path: Filesystem path to the Python source file.

    Returns:
        Loaded module object.

    Raises:
        ConfigError: If the file does not exist or cannot be imported.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"File not found: {path}",
            code="FILE_NOT_FOUND",
            hint="Check the file path and ensure it is a valid Python module.",
        )
    spec = importlib.util.spec_from_file_location("_grampus_user_module", p)
    if spec is None or spec.loader is None:
        raise ConfigError(
            f"Cannot load module from: {path}",
            code="MODULE_LOAD_ERROR",
            hint="Check the file path and ensure it is a valid Python module.",
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(
            f"Error executing module {path}: {exc}",
            code="MODULE_EXEC_ERROR",
            hint="Check the file path and ensure it is a valid Python module.",
        ) from exc
    return module


def require_function(module: ModuleType, name: str) -> Any:
    """Get a named function from a module.

    Args:
        module: The already-loaded module.
        name: Name of the function to retrieve.

    Returns:
        The function object.

    Raises:
        ConfigError: If the function is not found in the module.
    """
    fn = getattr(module, name, None)
    if fn is None:
        if name == "create_runner":
            hint = "Define 'def create_runner() -> AgentRunner:' in your agent file."
        elif name == "create_agent_def":
            hint = "Define 'def create_agent_def() -> AgentDefinition:' in your agent file or configure [agent] in grampus.yaml."
        else:
            hint = f"Add a `def {name}():` to your file."
        raise ConfigError(
            f"Function '{name}' not found in '{module.__file__}'. "
            f"Add a `def {name}():` to your file.",
            code="FUNCTION_NOT_FOUND",
            hint=hint,
        )
    return fn


def print_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    title: str = "",
) -> None:
    """Print a plain-text ASCII table to stdout.

    Args:
        headers: Column header names.
        rows: Data rows — each inner list must have the same length as headers.
        title: Optional title printed above the table.
    """
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    if title:
        click.echo(title)
    click.echo(sep)
    click.echo(header_row)
    click.echo(sep)
    for row in rows:
        cells = [
            str(row[i]).ljust(col_widths[i]) if i < len(col_widths) else str(row[i])
            for i in range(len(headers))
        ]
        click.echo("| " + " | ".join(cells) + " |")
    click.echo(sep)


def confirm(prompt: str, *, default: bool = False) -> bool:
    """Prompt the user for y/n confirmation.

    Args:
        prompt: Text to display before the [y/N] indicator.
        default: Default answer when the user presses Enter.

    Returns:
        True if the user confirmed, False otherwise.
    """
    return click.confirm(prompt, default=default)
