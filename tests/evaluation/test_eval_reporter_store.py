"""Tests for EvalReporter run_store and pubsub_topic integration."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.evaluation.assertions import AssertionResult
from grampus.evaluation.reporter import EvalReport, EvalReporter
from grampus.evaluation.run_store import EvalRunStore
from grampus.evaluation.suite import CaseResult, SuiteResult


def _make_suite_result(
    name: str = "TestSuite",
    pass_rate: float = 0.8,
) -> SuiteResult:
    cases = [
        CaseResult(
            case_id="c1",
            case_name="test-1",
            passed=True,
            assertion_results=[
                AssertionResult(passed=True, assertion_type="contains", detail="ok", score=1.0)
            ],
            duration_seconds=0.1,
        )
    ]
    return SuiteResult(
        suite_name=name,
        total_cases=1,
        passed=1,
        failed=0,
        errors=0,
        pass_rate=pass_rate,
        avg_duration_seconds=0.1,
        case_results=cases,
        run_at=datetime(2026, 6, 1, tzinfo=UTC),
        total_cost_usd=0.001,
    )


class TestEvalReporterRunStore:
    @pytest.mark.asyncio
    async def test_publish_saves_to_run_store(self) -> None:
        store = EvalRunStore()
        reporter = EvalReporter(run_store=store)
        report = EvalReport(suite_result=_make_suite_result())
        await reporter.publish(report)
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].suite_name == "TestSuite"
        assert runs[0].pass_rate == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_publish_stores_case_results(self) -> None:
        store = EvalRunStore()
        reporter = EvalReporter(run_store=store)
        report = EvalReport(suite_result=_make_suite_result())
        await reporter.publish(report)
        run = store.list_runs()[0]
        assert len(run.case_results) == 1
        assert run.case_results[0]["case_name"] == "test-1"

    @pytest.mark.asyncio
    async def test_run_store_failure_does_not_propagate(self) -> None:
        bad_store = MagicMock()
        bad_store.from_suite_result = MagicMock(side_effect=RuntimeError("store down"))
        reporter = EvalReporter(run_store=bad_store)
        report = EvalReport(suite_result=_make_suite_result())
        await reporter.publish(report)  # must not raise

    @pytest.mark.asyncio
    async def test_publish_without_run_store_is_noop(self) -> None:
        reporter = EvalReporter(run_store=None, pubsub=None)
        report = EvalReport(suite_result=_make_suite_result())
        await reporter.publish(report)  # must not raise


class TestEvalReporterPubsubTopic:
    @pytest.mark.asyncio
    async def test_publish_sends_completed_event_to_pubsub_topic(self) -> None:
        mock_pubsub = MagicMock()
        mock_pubsub.publish = AsyncMock()
        reporter = EvalReporter(
            pubsub=mock_pubsub,
            report_topic="grampus.eval.results",
            pubsub_topic="eval.suite.completed",
        )
        report = EvalReport(suite_result=_make_suite_result())
        await reporter.publish(report)

        topics_called = [kw["topic"] for _, kw in mock_pubsub.publish.call_args_list]
        assert "eval.suite.completed" in topics_called
        assert "grampus.eval.results" in topics_called

    @pytest.mark.asyncio
    async def test_pubsub_failure_does_not_propagate(self) -> None:
        mock_pubsub = MagicMock()
        mock_pubsub.publish = AsyncMock(side_effect=RuntimeError("broker down"))
        reporter = EvalReporter(pubsub=mock_pubsub, run_store=EvalRunStore())
        report = EvalReport(suite_result=_make_suite_result())
        await reporter.publish(report)  # must not raise

    @pytest.mark.asyncio
    async def test_pubsub_none_skips_completed_event(self) -> None:
        store = EvalRunStore()
        reporter = EvalReporter(pubsub=None, run_store=store)
        report = EvalReport(suite_result=_make_suite_result())
        await reporter.publish(report)
        assert len(store.list_runs()) == 1  # store still saved
