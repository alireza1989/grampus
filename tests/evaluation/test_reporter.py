"""Tests for EvalReporter."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.evaluation.assertions import AssertionResult
from grampus.evaluation.baseline import RegressionReport
from grampus.evaluation.reporter import EvalReport, EvalReporter, ReportFormat
from grampus.evaluation.suite import CaseResult, SuiteResult


def _make_suite_result(
    passed: int = 3,
    failed: int = 2,
    cost_usd: float = 0.0042,
    duration: float = 1.23,
) -> SuiteResult:
    total = passed + failed
    cases = []
    for i in range(passed):
        cases.append(
            CaseResult(
                case_id=f"pass-{i}",
                case_name=f"test-passing-{i}",
                passed=True,
                assertion_results=[
                    AssertionResult(
                        passed=True,
                        assertion_type="contains",
                        detail="ok",
                        score=1.0,
                    )
                ],
                duration_seconds=0.1,
            )
        )
    for i in range(failed):
        cases.append(
            CaseResult(
                case_id=f"fail-{i}",
                case_name=f"test-failing-{i}",
                passed=False,
                assertion_results=[
                    AssertionResult(
                        passed=False,
                        assertion_type="tool_was_called",
                        detail="tool 'search' was not called",
                        score=0.0,
                    )
                ],
                duration_seconds=0.2,
            )
        )

    return SuiteResult(
        suite_name="MySuite",
        total_cases=total,
        passed=passed,
        failed=failed,
        errors=0,
        pass_rate=passed / total,
        avg_duration_seconds=duration,
        case_results=cases,
        run_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        total_cost_usd=cost_usd,
    )


def _make_report(with_regression: bool = False) -> EvalReport:
    sr = _make_suite_result()
    reg = None
    if with_regression:
        reg = RegressionReport(
            baseline_id="base-1",
            current_pass_rate=0.4,
            baseline_pass_rate=0.6,
            delta=-0.2,
            regressed=True,
            regression_threshold=0.05,
            newly_failing=["test-failing-0", "test-failing-1"],
            newly_passing=[],
            cost_delta_usd=0.001,
            duration_delta_seconds=0.1,
        )
    return EvalReport(suite_result=sr, regression_report=reg, format=ReportFormat.TEXT)


class TestEvalReporterText:
    def test_render_text_contains_suite_name(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.TEXT)
        assert "MySuite" in text

    def test_render_text_contains_pass_rate(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.TEXT)
        assert "60" in text  # 3/5 = 60%

    def test_render_text_shows_passing_cases_with_checkmark(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.TEXT)
        assert "✓" in text

    def test_render_text_shows_failing_cases_with_cross(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.TEXT)
        assert "✗" in text

    def test_render_text_shows_failing_assertion_detail(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.TEXT)
        assert "tool 'search' was not called" in text

    def test_render_text_shows_regression_warning(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(with_regression=True), fmt=ReportFormat.TEXT)
        assert "REGRESSION" in text


class TestEvalReporterJSON:
    def test_render_json_is_valid_json(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.JSON)
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_render_json_contains_suite_result(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.JSON)
        data = json.loads(text)
        assert "suite_result" in data
        assert data["suite_result"]["suite_name"] == "MySuite"


class TestEvalReporterJUnitXML:
    def test_render_junit_xml_is_valid_xml(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.JUNIT_XML)
        root = ET.fromstring(text)
        assert root is not None

    def test_render_junit_xml_contains_testcase_elements(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.JUNIT_XML)
        root = ET.fromstring(text)
        testcases = root.findall(".//testcase")
        assert len(testcases) == 5  # 3 passing + 2 failing

    def test_render_junit_xml_failure_element_for_failed_cases(self) -> None:
        reporter = EvalReporter()
        text = reporter.render(_make_report(), fmt=ReportFormat.JUNIT_XML)
        root = ET.fromstring(text)
        failures = root.findall(".//failure")
        assert len(failures) == 2


class TestEvalReporterPublish:
    @pytest.mark.asyncio
    async def test_publish_calls_pubsub_when_configured(self) -> None:
        mock_pubsub = MagicMock()
        mock_pubsub.publish = AsyncMock()
        reporter = EvalReporter(pubsub=mock_pubsub, report_topic="test.topic")
        await reporter.publish(_make_report())
        # publish() sends two events: the full report + the completed notification
        assert mock_pubsub.publish.call_count >= 1
        topics_called = {kw["topic"] for _, kw in mock_pubsub.publish.call_args_list}
        assert "test.topic" in topics_called

    @pytest.mark.asyncio
    async def test_publish_noop_when_pubsub_none(self) -> None:
        reporter = EvalReporter(pubsub=None)
        await reporter.publish(_make_report())
