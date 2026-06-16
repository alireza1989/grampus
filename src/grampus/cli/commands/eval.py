"""grampus eval — run an evaluation suite against an agent."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import click

from grampus.cli.commands._utils import load_module, require_function
from grampus.core.errors import ConfigError
from grampus.core.logging import get_logger
from grampus.evaluation.reporter import EvalReport, EvalReporter, ReportFormat
from grampus.evaluation.suite import SuiteResult

_log = get_logger(__name__)

_FORMAT_MAP: dict[str, ReportFormat] = {
    "text": ReportFormat.TEXT,
    "json": ReportFormat.JSON,
    "junit": ReportFormat.JUNIT_XML,
}


@click.command("eval")
@click.argument("suite_file")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json", "junit"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Write report to FILE instead of stdout.",
)
@click.option(
    "--fail-under",
    type=float,
    default=None,
    help="Exit 1 if pass rate is below this threshold (0.0–1.0).",
)
def eval_cmd(
    suite_file: str,
    fmt: str,
    output_path: str | None,
    fail_under: float | None,
) -> None:
    """Run an evaluation suite defined in SUITE_FILE."""
    try:
        suite = _load_suite(suite_file)
    except ConfigError as exc:
        click.echo(f"Error: {exc}")
        sys.exit(1)

    start = time.monotonic()
    suite_result = asyncio.run(_run_suite(suite))
    elapsed = time.monotonic() - start

    _print_stderr_summary(suite_result, elapsed)

    report = EvalReport(suite_result=suite_result)
    reporter = EvalReporter()
    rendered = reporter.render(report, fmt=_FORMAT_MAP.get(fmt, ReportFormat.TEXT))

    if output_path:
        Path(output_path).write_text(rendered)
    else:
        click.echo(rendered)

    if fail_under is not None and suite_result.pass_rate < fail_under:
        sys.exit(1)


def _load_suite(suite_file: str) -> Any:
    """Load the EvalSuite from a user-provided file.

    Args:
        suite_file: Path to the Python file containing create_suite().

    Returns:
        The EvalSuite returned by create_suite().

    Raises:
        ConfigError: If the file or function is missing.
    """
    module = load_module(suite_file)
    factory = require_function(module, "create_suite")
    return factory()


async def _run_suite(suite: Any) -> SuiteResult:
    """Await the suite's run() coroutine.

    Args:
        suite: EvalSuite instance.

    Returns:
        SuiteResult with pass/fail counts.
    """
    return await suite.run()  # type: ignore[no-any-return]


def _print_stderr_summary(result: SuiteResult, elapsed: float) -> None:
    """Print a one-line summary to stderr regardless of --output.

    Args:
        result: Completed SuiteResult.
        elapsed: Wall-clock time in seconds.
    """
    pct = result.pass_rate * 100
    click.echo(
        f"Eval complete: {result.passed}/{result.total_cases} passed "
        f"({pct:.1f}%) in {elapsed:.2f}s  [${result.total_cost_usd:.4f}]",
        err=True,
    )
