"""Tests for QualityBaseline."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nexus.evaluation.suite import CaseResult, SuiteResult


def _make_suite_result(
    suite_name: str = "suite",
    passed: int = 4,
    total: int = 5,
    cost_usd: float = 0.01,
    duration: float = 1.0,
    case_results: dict[str, bool] | None = None,
) -> SuiteResult:

    if case_results is None:
        case_results = {f"case-{i}": (i < passed) for i in range(total)}

    cr_list = []
    for cid, p in case_results.items():
        cr_list.append(
            CaseResult(
                case_id=cid,
                case_name=cid,
                passed=p,
                assertion_results=[],
                duration_seconds=duration / total,
            )
        )

    return SuiteResult(
        suite_name=suite_name,
        total_cases=total,
        passed=passed,
        failed=total - passed,
        errors=0,
        pass_rate=passed / total if total > 0 else 0.0,
        avg_duration_seconds=duration,
        case_results=cr_list,
        total_cost_usd=cost_usd,
        run_at=datetime.now(UTC),
    )


class TestQualityBaseline:
    def test_record_stores_run(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline()
        sr = _make_suite_result()
        run = qb.record(sr)
        assert run.pass_rate == pytest.approx(0.8)
        assert len(qb.history()) == 1

    def test_pin_sets_baseline(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline()
        sr = _make_suite_result()
        run = qb.record(sr)
        qb.pin(run.id)
        assert qb.pinned() is not None
        assert qb.pinned().id == run.id  # type: ignore[union-attr]

    def test_pin_latest_pins_most_recent(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline()
        qb.record(_make_suite_result())
        run2 = qb.record(_make_suite_result())
        pinned = qb.pin_latest()
        assert pinned.id == run2.id

    def test_compare_returns_none_when_no_baseline(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline()
        sr = _make_suite_result()
        report = qb.compare(sr)
        assert report is None

    def test_compare_no_regression_when_above_threshold(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline(regression_threshold=0.05)
        baseline_sr = _make_suite_result(passed=4, total=5)  # 80%
        qb.record(baseline_sr)
        qb.pin_latest()
        new_sr = _make_suite_result(passed=4, total=5)  # still 80%
        report = qb.compare(new_sr)
        assert report is not None
        assert report.regressed is False

    def test_compare_regression_detected_when_below_threshold(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline(regression_threshold=0.05)
        baseline_sr = _make_suite_result(passed=10, total=10)  # 100%
        qb.record(baseline_sr)
        qb.pin_latest()
        new_sr = _make_suite_result(passed=8, total=10)  # 80% — delta = -0.2
        report = qb.compare(new_sr)
        assert report is not None
        assert report.regressed is True
        assert report.delta == pytest.approx(-0.2)

    def test_compare_identifies_newly_failing_cases(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline()
        baseline_sr = _make_suite_result(case_results={"case-A": True, "case-B": True})
        qb.record(baseline_sr)
        qb.pin_latest()
        new_sr = _make_suite_result(case_results={"case-A": True, "case-B": False})
        report = qb.compare(new_sr)
        assert report is not None
        assert "case-B" in report.newly_failing

    def test_compare_identifies_newly_passing_cases(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline()
        baseline_sr = _make_suite_result(case_results={"case-A": False, "case-B": True})
        qb.record(baseline_sr)
        qb.pin_latest()
        new_sr = _make_suite_result(case_results={"case-A": True, "case-B": True})
        report = qb.compare(new_sr)
        assert report is not None
        assert "case-A" in report.newly_passing

    def test_history_sorted_by_recorded_at(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline()
        for _ in range(3):
            qb.record(_make_suite_result())
        history = qb.history()
        assert len(history) == 3
        for i in range(len(history) - 1):
            assert history[i].recorded_at <= history[i + 1].recorded_at

    def test_cost_delta_computed_correctly(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        qb = QualityBaseline()
        qb.record(_make_suite_result(cost_usd=0.01))
        qb.pin_latest()
        report = qb.compare(_make_suite_result(cost_usd=0.03))
        assert report is not None
        assert report.cost_delta_usd == pytest.approx(0.02)
