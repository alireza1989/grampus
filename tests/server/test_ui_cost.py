"""Tests for the /ui/cost/ pages."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nexus.core.types import AgentDefinition
from nexus.observability.metrics import NexusMetrics
from nexus.server.app import create_app


def _make_client(metrics: NexusMetrics | None) -> TestClient:
    runner = MagicMock()
    agent_def = AgentDefinition(name="test", model="claude-sonnet-4-6")
    app = create_app(runner, agent_def, nexus_metrics=metrics)
    return TestClient(app, raise_server_exceptions=False)


def test_cost_page_returns_200() -> None:
    client = _make_client(None)
    resp = client.get("/ui/cost/")
    assert resp.status_code == 200


def test_cost_page_links_to_summary_partial() -> None:
    client = _make_client(None)
    resp = client.get("/ui/cost/")
    assert "/ui/cost/_summary" in resp.text


def test_cost_summary_empty_state_when_no_metrics() -> None:
    client = _make_client(None)
    resp = client.get("/ui/cost/_summary")
    assert resp.status_code == 200
    assert "No metrics configured" in resp.text


def test_cost_summary_shows_stat_cards() -> None:
    m = NexusMetrics(agent_id="agent-1")
    m.record_llm_call(
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0042,
        latency_ms=200.0,
    )
    client = _make_client(m)
    resp = client.get("/ui/cost/_summary")
    assert resp.status_code == 200
    assert "0.0042" in resp.text
    assert "150" in resp.text  # total tokens


def test_cost_summary_shows_model_table() -> None:
    m = NexusMetrics(agent_id="agent-1")
    m.record_llm_call(
        model="claude-opus-4-7",
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.012,
        latency_ms=300.0,
    )
    client = _make_client(m)
    resp = client.get("/ui/cost/_summary")
    assert "claude-opus-4-7" in resp.text
    assert "0.012" in resp.text


def test_cost_summary_auto_refresh_attribute() -> None:
    client = _make_client(None)
    resp = client.get("/ui/cost/")
    assert "every 30s" in resp.text
