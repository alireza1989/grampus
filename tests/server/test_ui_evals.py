"""Tests for the /ui/evals/ pages."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from grampus.core.types import AgentDefinition
from grampus.evaluation.run_store import EvalRunRecord, EvalRunStore
from grampus.server.app import create_app


def _make_store(*records: EvalRunRecord) -> EvalRunStore:
    store = EvalRunStore()
    for r in records:
        store.append(r)
    return store


def _rec(
    suite_name: str = "MySuite",
    pass_rate: float = 0.8,
    run_at: datetime | None = None,
    run_id: str | None = None,
) -> EvalRunRecord:
    r = EvalRunRecord(
        suite_name=suite_name,
        run_at=run_at or datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        pass_rate=pass_rate,
        passed=8,
        failed=2,
        errors=0,
        total_cases=10,
        total_cost_usd=0.001,
        avg_duration_seconds=0.5,
        case_results=[
            {
                "case_id": "c1",
                "case_name": "test-alpha",
                "passed": True,
                "assertion_results": [
                    {"passed": True, "assertion_type": "contains", "detail": "ok", "score": 1.0}
                ],
                "duration_seconds": 0.1,
                "error": None,
                "tags": [],
            }
        ],
    )
    if run_id is not None:
        object.__setattr__(r, "run_id", run_id)
    return r


@pytest.fixture()
def store() -> EvalRunStore:
    return _make_store(
        _rec("SuiteA", 0.9, datetime(2026, 1, 2, tzinfo=UTC)),
        _rec("SuiteA", 0.7, datetime(2026, 1, 1, tzinfo=UTC)),
        _rec("SuiteB", 0.5, datetime(2026, 1, 1, tzinfo=UTC)),
    )


@pytest.fixture()
def client(store: EvalRunStore) -> TestClient:
    runner = MagicMock()
    agent_def = AgentDefinition(name="test", model="claude-sonnet-4-6")
    app = create_app(runner, agent_def, eval_run_store=store)
    return TestClient(app, raise_server_exceptions=False)


def test_evals_page_returns_200(client: TestClient) -> None:
    resp = client.get("/ui/evals/")
    assert resp.status_code == 200


def test_evals_page_populates_suite_filter(client: TestClient) -> None:
    resp = client.get("/ui/evals/")
    assert "SuiteA" in resp.text
    assert "SuiteB" in resp.text


def test_evals_runs_returns_200(client: TestClient) -> None:
    resp = client.get("/ui/evals/_runs")
    assert resp.status_code == 200


def test_evals_runs_shows_pass_rates(client: TestClient) -> None:
    resp = client.get("/ui/evals/_runs")
    assert "90%" in resp.text or "70%" in resp.text


def test_evals_runs_filter_by_suite(client: TestClient) -> None:
    resp = client.get("/ui/evals/_runs?suite_name=SuiteB")
    assert resp.status_code == 200
    assert "SuiteB" in resp.text
    assert "SuiteA" not in resp.text


def test_evals_runs_regression_badge(store: EvalRunStore) -> None:
    runner = MagicMock()
    agent_def = AgentDefinition(name="test", model="claude-sonnet-4-6")
    # oldest (0.9) → newest (0.6): dropped 30pp → should show regression
    regressing_store = _make_store(
        _rec("S", 0.9, datetime(2026, 1, 1, tzinfo=UTC)),
        _rec("S", 0.6, datetime(2026, 1, 2, tzinfo=UTC)),
    )
    app = create_app(runner, agent_def, eval_run_store=regressing_store)
    c = TestClient(app, raise_server_exceptions=False)
    resp = c.get("/ui/evals/_runs?suite_name=S")
    assert "regression" in resp.text


def test_evals_detail_returns_200(client: TestClient, store: EvalRunStore) -> None:
    run_id = store.list_runs()[0].run_id
    resp = client.get(f"/ui/evals/_detail/{run_id}")
    assert resp.status_code == 200


def test_evals_detail_shows_cases(client: TestClient, store: EvalRunStore) -> None:
    run_id = store.list_runs()[0].run_id
    resp = client.get(f"/ui/evals/_detail/{run_id}")
    assert "test-alpha" in resp.text


def test_evals_detail_404_for_missing(client: TestClient) -> None:
    resp = client.get("/ui/evals/_detail/no-such-id")
    assert resp.status_code == 404


def test_evals_trend_returns_200(client: TestClient) -> None:
    resp = client.get("/ui/evals/_trend?suite_name=SuiteA")
    assert resp.status_code == 200


def test_evals_trend_shows_bars(client: TestClient) -> None:
    resp = client.get("/ui/evals/_trend?suite_name=SuiteA")
    assert "█" in resp.text or "░" in resp.text


def test_evals_no_store_shows_empty_state() -> None:
    runner = MagicMock()
    agent_def = AgentDefinition(name="test", model="claude-sonnet-4-6")
    app = create_app(runner, agent_def)  # no eval_run_store
    c = TestClient(app, raise_server_exceptions=False)
    resp = c.get("/ui/evals/_runs")
    assert resp.status_code == 200
    assert "No eval runs" in resp.text
