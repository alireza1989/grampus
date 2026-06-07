"""Tests for the Nexus web UI shell — sidebar, nav, static files, empty states."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nexus.core.types import AgentDefinition
from nexus.server.app import create_app


@pytest.fixture()
def app() -> MagicMock:
    runner = MagicMock()
    agent_def = AgentDefinition(name="test", model="claude-sonnet-4-6")
    return create_app(runner, agent_def)  # type: ignore[return-value]


@pytest.fixture()
def client(app: MagicMock) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)  # type: ignore[arg-type]


def test_dashboard_returns_200(client: TestClient) -> None:
    resp = client.get("/ui/")
    assert resp.status_code == 200


def test_dashboard_html_contains_sidebar(client: TestClient) -> None:
    resp = client.get("/ui/")
    assert "sidebar" in resp.text


def test_dashboard_html_contains_nav_links(client: TestClient) -> None:
    resp = client.get("/ui/")
    assert "/ui/memory/" in resp.text
    assert "/ui/" in resp.text


def test_dashboard_active_page_dashboard(client: TestClient) -> None:
    resp = client.get("/ui/")
    assert 'class="active"' in resp.text


def test_dashboard_stats_partial_returns_200(client: TestClient) -> None:
    resp = client.get("/ui/_stats")
    assert resp.status_code == 200


def test_dashboard_stats_no_metrics_shows_dash(client: TestClient) -> None:
    resp = client.get("/ui/_stats")
    assert "—" in resp.text


def test_static_css_served(client: TestClient) -> None:
    resp = client.get("/ui/static/style.css")
    assert resp.status_code == 200


def test_memory_page_returns_200(client: TestClient) -> None:
    resp = client.get("/ui/memory/")
    assert resp.status_code == 200


def test_memory_active_page_memory(client: TestClient) -> None:
    resp = client.get("/ui/memory/")
    assert "active" in resp.text


def test_memory_rows_partial_empty_state(client: TestClient) -> None:
    resp = client.get("/ui/memory/_rows")
    assert resp.status_code == 200
    assert "No memory entries found" in resp.text


def test_memory_rows_filter_params_accepted(client: TestClient) -> None:
    resp = client.get("/ui/memory/_rows?agent_id=test&memory_type=episodic&q=hello&min_trust=0.5")
    assert resp.status_code == 200


def test_memory_detail_missing_entry_returns_404(client: TestClient) -> None:
    resp = client.get("/ui/memory/_detail/nonexistent")
    assert resp.status_code == 404


def test_memory_delete_missing_entry_returns_404(client: TestClient) -> None:
    resp = client.delete("/ui/memory/nonexistent?memory_type=episodic")
    assert resp.status_code == 404
