"""Tests for EvalRunStore."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from grampus.evaluation.run_store import EvalRunRecord, EvalRunStore


def _make_record(
    suite_name: str = "MySuite",
    pass_rate: float = 0.8,
    run_at: datetime | None = None,
) -> EvalRunRecord:
    return EvalRunRecord(
        suite_name=suite_name,
        run_at=run_at or datetime(2026, 1, 1, tzinfo=UTC),
        pass_rate=pass_rate,
        passed=8,
        failed=2,
        errors=0,
        total_cases=10,
        total_cost_usd=0.001,
        avg_duration_seconds=0.5,
    )


class TestEvalRunStoreAppendAndList:
    def test_empty_store_returns_empty_list(self) -> None:
        store = EvalRunStore()
        assert store.list_runs() == []

    def test_append_and_list_returns_record(self) -> None:
        store = EvalRunStore()
        rec = _make_record()
        store.append(rec)
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].run_id == rec.run_id

    def test_list_runs_newest_first(self) -> None:
        store = EvalRunStore()
        older = _make_record(run_at=datetime(2026, 1, 1, tzinfo=UTC))
        newer = _make_record(run_at=datetime(2026, 1, 2, tzinfo=UTC))
        store.append(older)
        store.append(newer)
        runs = store.list_runs()
        assert runs[0].run_id == newer.run_id
        assert runs[1].run_id == older.run_id

    def test_list_runs_filter_by_suite_name(self) -> None:
        store = EvalRunStore()
        store.append(_make_record(suite_name="A"))
        store.append(_make_record(suite_name="B"))
        store.append(_make_record(suite_name="A"))
        runs = store.list_runs(suite_name="A")
        assert len(runs) == 2
        assert all(r.suite_name == "A" for r in runs)

    def test_list_runs_limit(self) -> None:
        store = EvalRunStore()
        for _ in range(10):
            store.append(_make_record())
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_max_runs_evicts_oldest(self) -> None:
        store = EvalRunStore(max_runs=3)
        first = _make_record()
        store.append(first)
        store.append(_make_record())
        store.append(_make_record())
        store.append(_make_record())  # first is evicted
        runs = store.list_runs(limit=100)
        assert len(runs) == 3
        ids = [r.run_id for r in runs]
        assert first.run_id not in ids


class TestEvalRunStoreGet:
    def test_get_existing_returns_record(self) -> None:
        store = EvalRunStore()
        rec = _make_record()
        store.append(rec)
        result = store.get(rec.run_id)
        assert result is not None
        assert result.run_id == rec.run_id

    def test_get_missing_returns_none(self) -> None:
        store = EvalRunStore()
        assert store.get("does-not-exist") is None


class TestEvalRunStoreListSuiteNames:
    def test_empty_store_returns_empty(self) -> None:
        store = EvalRunStore()
        assert store.list_suite_names() == []

    def test_returns_unique_sorted_names(self) -> None:
        store = EvalRunStore()
        store.append(_make_record(suite_name="Zoo"))
        store.append(_make_record(suite_name="Alpha"))
        store.append(_make_record(suite_name="Zoo"))
        names = store.list_suite_names()
        assert names == ["Alpha", "Zoo"]


class TestEvalRunStoreFromSuiteResult:
    def test_builds_record_from_suite_result(self) -> None:
        from unittest.mock import MagicMock

        sr = MagicMock()
        sr.suite_name = "SuiteX"
        sr.run_at = datetime(2026, 3, 1, tzinfo=UTC)
        sr.pass_rate = 0.75
        sr.passed = 3
        sr.failed = 1
        sr.errors = 0
        sr.total_cases = 4
        sr.total_cost_usd = 0.005
        sr.avg_duration_seconds = 1.2
        mock_cr = MagicMock()
        mock_cr.model_dump = lambda mode=None: {"case_name": "tc1", "passed": True}
        sr.case_results = [mock_cr]

        store = EvalRunStore()
        rec = store.from_suite_result(sr)

        assert rec.suite_name == "SuiteX"
        assert rec.pass_rate == pytest.approx(0.75)
        assert rec.total_cases == 4
        assert len(rec.case_results) == 1
        assert rec.case_results[0]["case_name"] == "tc1"

    def test_from_suite_result_generates_unique_run_id(self) -> None:
        from unittest.mock import MagicMock

        sr = MagicMock()
        sr.suite_name = "S"
        sr.run_at = datetime(2026, 1, 1, tzinfo=UTC)
        sr.pass_rate = 1.0
        sr.passed = sr.failed = sr.errors = sr.total_cases = 0
        sr.total_cost_usd = 0.0
        sr.avg_duration_seconds = 0.0
        sr.case_results = []

        store = EvalRunStore()
        r1 = store.from_suite_result(sr)
        r2 = store.from_suite_result(sr)
        assert r1.run_id != r2.run_id
