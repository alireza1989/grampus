"""Tests for /evals/* REST endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from grampus.core.types import AgentDefinition
from grampus.evaluation.run_store import EvalRunRecord, EvalRunStore
from grampus.server.app import create_app


def _make_record(suite_name: str = "MySuite", pass_rate: float = 0.8) -> EvalRunRecord:
    return EvalRunRecord(
        suite_name=suite_name,
        run_at=datetime(2026, 1, 1, tzinfo=UTC),
        pass_rate=pass_rate,
        passed=8,
        failed=2,
        errors=0,
        total_cases=10,
        total_cost_usd=0.001,
        avg_duration_seconds=0.5,
        case_results=[{"case_name": "tc1", "passed": True}],
    )


def _make_client(store: EvalRunStore | None = None) -> TestClient:
    runner = MagicMock()
    agent_def = AgentDefinition(name="test", model="claude-sonnet-4-6")
    app = create_app(runner, agent_def, eval_run_store=store)
    return TestClient(app, raise_server_exceptions=False)


def test_list_runs_no_store_returns_empty() -> None:
    client = _make_client(None)
    resp = client.get("/evals/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"] == []
    assert data["count"] == 0


def test_list_runs_returns_records() -> None:
    store = EvalRunStore()
    store.append(_make_record("A"))
    store.append(_make_record("B"))
    client = _make_client(store)
    resp = client.get("/evals/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2


def test_list_runs_filter_by_suite() -> None:
    store = EvalRunStore()
    store.append(_make_record("A"))
    store.append(_make_record("B"))
    client = _make_client(store)
    resp = client.get("/evals/runs?suite_name=A")
    data = resp.json()
    assert data["count"] == 1
    assert data["runs"][0]["suite_name"] == "A"


def test_get_run_returns_record() -> None:
    store = EvalRunStore()
    rec = _make_record()
    store.append(rec)
    client = _make_client(store)
    resp = client.get(f"/evals/runs/{rec.run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == rec.run_id
    assert data["suite_name"] == "MySuite"


def test_get_run_404_for_missing() -> None:
    store = EvalRunStore()
    client = _make_client(store)
    resp = client.get("/evals/runs/does-not-exist")
    assert resp.status_code == 404


def test_get_run_503_when_no_store() -> None:
    client = _make_client(None)
    resp = client.get("/evals/runs/some-id")
    assert resp.status_code == 404


def test_export_run_returns_json_file() -> None:
    store = EvalRunStore()
    rec = _make_record()
    store.append(rec)
    client = _make_client(store)
    resp = client.get(f"/evals/runs/{rec.run_id}/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert "attachment" in resp.headers.get("content-disposition", "")
    data = json.loads(resp.content)
    assert data["run_id"] == rec.run_id


def test_export_run_contains_case_results() -> None:
    store = EvalRunStore()
    rec = _make_record()
    store.append(rec)
    client = _make_client(store)
    resp = client.get(f"/evals/runs/{rec.run_id}/export")
    data = json.loads(resp.content)
    assert len(data["case_results"]) == 1
    assert data["case_results"][0]["case_name"] == "tc1"


def test_export_run_404_for_missing() -> None:
    store = EvalRunStore()
    client = _make_client(store)
    resp = client.get("/evals/runs/no-such/export")
    assert resp.status_code == 404
