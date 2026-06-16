"""EvalReporter — renders and publishes evaluation reports."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

from pydantic import BaseModel

from grampus.core.logging import get_logger
from grampus.evaluation.baseline import RegressionReport
from grampus.evaluation.suite import CaseResult, SuiteResult

logger = get_logger(__name__)

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum as StrEnum  # type: ignore[assignment]


class ReportFormat(StrEnum):
    """Output formats for evaluation reports."""

    TEXT = "text"
    JSON = "json"
    JUNIT_XML = "junit_xml"


class EvalReport(BaseModel):
    """Container for a full evaluation report.

    Attributes:
        suite_result: The completed SuiteResult.
        regression_report: Optional regression comparison.
        prompt_version: Optional prompt version label.
        format: Preferred default format.
    """

    suite_result: SuiteResult
    regression_report: RegressionReport | None = None
    prompt_version: str | None = None
    format: ReportFormat = ReportFormat.TEXT


class EvalReporter:
    """Renders and outputs evaluation reports.

    Args:
        pubsub: Optional DaprPubSub for publishing results.
        report_topic: Pub/sub topic for full report JSON.
        run_store: Optional EvalRunStore to persist run records.
        pubsub_topic: Pub/sub topic for the lightweight eval.suite.completed event.
    """

    def __init__(
        self,
        *,
        pubsub: Any | None = None,
        report_topic: str = "grampus.eval.results",
        run_store: Any | None = None,
        pubsub_topic: str = "eval.suite.completed",
    ) -> None:
        self._pubsub = pubsub
        self._topic = report_topic
        self._run_store = run_store
        self._pubsub_topic = pubsub_topic

    def render(self, report: EvalReport, *, fmt: ReportFormat = ReportFormat.TEXT) -> str:
        """Render report as a string in the requested format.

        Args:
            report: The EvalReport to render.
            fmt: Output format.

        Returns:
            Rendered string.
        """
        if fmt == ReportFormat.JSON:
            return _render_json(report)
        if fmt == ReportFormat.JUNIT_XML:
            return _render_junit_xml(report)
        return _render_text(report)

    def print(self, report: EvalReport, *, fmt: ReportFormat = ReportFormat.TEXT) -> None:
        """Render and print to stdout.

        Args:
            report: The EvalReport to print.
            fmt: Output format.
        """
        print(self.render(report, fmt=fmt))  # noqa: T201 — intentional stdout

    async def publish(self, report: EvalReport) -> None:
        """Publish report JSON to pub/sub topic; save to run_store; emit completed event.

        Failures from the store or pub/sub never propagate to the caller.

        Args:
            report: The EvalReport to publish.
        """
        if self._pubsub is not None:
            try:
                payload = json.dumps(json.loads(_render_json(report))).encode()
                await self._pubsub.publish(topic=self._topic, data=payload)
                logger.info("eval_report_published", topic=self._topic)
            except Exception:  # noqa: BLE001
                logger.warning("eval_report_publish_failed", topic=self._topic)

        if self._run_store is not None:
            try:
                record = self._run_store.from_suite_result(report.suite_result)
                self._run_store.append(record)
                logger.debug("eval_run_saved", run_id=record.run_id)
            except Exception:  # noqa: BLE001
                logger.warning("eval_run_save_failed")

        if self._pubsub is not None and self._pubsub_topic:
            try:
                sr = report.suite_result
                completed: dict[str, Any] = {
                    "suite_name": sr.suite_name,
                    "pass_rate": sr.pass_rate,
                    "total_cases": sr.total_cases,
                    "passed": sr.passed,
                    "failed": sr.failed,
                    "errors": sr.errors,
                    "total_cost_usd": sr.total_cost_usd,
                    "run_at": sr.run_at.isoformat(),
                }
                await self._pubsub.publish(
                    topic=self._pubsub_topic,
                    data=json.dumps(completed).encode(),
                )
                logger.info("eval_completed_published", topic=self._pubsub_topic)
            except Exception:  # noqa: BLE001
                logger.warning("eval_completed_publish_failed", topic=self._pubsub_topic)


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------


def _render_text(report: EvalReport) -> str:
    sr = report.suite_result
    lines: list[str] = []
    sep = "═" * 55

    lines.append(sep)
    ts = sr.run_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f" EvalSuite: {sr.suite_name}  |  Run: {ts}")
    lines.append(sep)

    pct = sr.pass_rate * 100
    lines.append(
        f" PASSED  {sr.passed}/{sr.total_cases}  ({pct:.1f}%)"
        f"   Cost: ${sr.total_cost_usd:.4f}"
        f"   Duration: {sr.avg_duration_seconds:.2f}s"
    )
    lines.append("")

    for cr in sr.case_results:
        lines.extend(_render_case_text(cr))

    if report.regression_report:
        lines.extend(_render_regression_text(report.regression_report))

    lines.append(sep)
    return "\n".join(lines)


def _render_case_text(cr: CaseResult) -> list[str]:
    icon = "✓" if cr.passed else "✗"
    lines = [f" {icon} {cr.case_name:<45} {cr.duration_seconds:.2f}s"]
    if not cr.passed:
        if cr.error:
            lines.append(f"   └─ ERROR: {cr.error}")
        for ar in cr.assertion_results:
            if not ar.passed:
                lines.append(f"   └─ {ar.assertion_type}: {ar.detail}")
    return lines


def _render_regression_text(reg: RegressionReport) -> list[str]:
    lines = [""]
    delta_pp = reg.delta * 100
    lines.append(
        f"[REGRESSION] Pass rate dropped {abs(delta_pp):.1f}pp"
        f" vs baseline (threshold: {reg.regression_threshold * 100:.1f}pp)"
    )
    if reg.newly_failing:
        lines.append(f"  Newly failing: {', '.join(reg.newly_failing)}")
    return lines


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------


def _render_json(report: EvalReport) -> str:
    return report.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# JUnit XML renderer
# ---------------------------------------------------------------------------


def _render_junit_xml(report: EvalReport) -> str:
    sr = report.suite_result
    total = sr.total_cases
    failures = sr.failed + sr.errors
    time_str = f"{sr.avg_duration_seconds:.3f}"

    testsuites = ET.Element(
        "testsuites",
        name=sr.suite_name,
        tests=str(total),
        failures=str(failures),
        errors="0",
        time=time_str,
    )
    testsuite = ET.SubElement(
        testsuites,
        "testsuite",
        name=sr.suite_name,
        tests=str(total),
        failures=str(failures),
        errors="0",
        time=time_str,
    )

    for cr in sr.case_results:
        _add_testcase_xml(testsuite, cr)

    ET.indent(testsuites)
    header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    return header + ET.tostring(testsuites, encoding="unicode")


def _add_testcase_xml(parent: ET.Element, cr: CaseResult) -> None:
    tc = ET.SubElement(
        parent,
        "testcase",
        name=cr.case_name,
        time=f"{cr.duration_seconds:.3f}",
    )
    if not cr.passed:
        msg = _first_failure_message(cr)
        ET.SubElement(tc, "failure", message=msg)


def _first_failure_message(cr: CaseResult) -> str:
    if cr.error:
        return f"ERROR: {cr.error}"
    for ar in cr.assertion_results:
        if not ar.passed:
            return f"{ar.assertion_type}: {ar.detail}"
    return "assertion failed"
