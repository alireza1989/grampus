"""Tests for the Nexus web UI memory inspector with a mock MemoryManager."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from grampus.core.types import AgentDefinition
from grampus.server.app import create_app

_FAKE_RECORDS = [
    {
        "id": "aaaa-1111",
        "agent_id": "agent-x",
        "memory_type": "episodic",
        "content": "The user prefers dark mode.",
        "trust_score": 0.9,
        "created_at": datetime(2026, 6, 9, 14, 23, tzinfo=UTC),
        "last_accessed": None,
        "metadata": {"session_id": "s1"},
        "provenance": {"source_type": "USER_INPUT", "trust_level": 0.9, "content_hash": "abc123"},
    },
    {
        "id": "bbbb-2222",
        "agent_id": "agent-y",
        "memory_type": "semantic",
        "content": "Python is a programming language.",
        "trust_score": 0.7,
        "created_at": datetime(2026, 6, 8, 10, 0, tzinfo=UTC),
        "last_accessed": datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
        "metadata": {},
        "provenance": None,
    },
    {
        "id": "cccc-3333",
        "agent_id": "agent-x",
        "memory_type": "procedural",
        "content": "Step 1: fetch data. Step 2: process.",
        "trust_score": 0.4,
        "created_at": datetime(2026, 6, 7, 8, 0, tzinfo=UTC),
        "last_accessed": None,
        "metadata": {},
        "provenance": None,
    },
]


@pytest.fixture()
def mock_manager() -> MagicMock:
    manager = MagicMock()
    manager.list_records = AsyncMock(return_value=_FAKE_RECORDS)
    manager.forget = AsyncMock(return_value=None)
    return manager


@pytest.fixture()
def client(mock_manager: MagicMock) -> TestClient:
    runner = MagicMock()
    agent_def = AgentDefinition(name="test", model="claude-sonnet-4-6")
    app = create_app(runner, agent_def, memory_manager=mock_manager)
    return TestClient(app, raise_server_exceptions=False)


def test_memory_rows_shows_entries(client: TestClient) -> None:
    resp = client.get("/ui/memory/_rows")
    assert resp.status_code == 200
    assert "aaaa-1111" in resp.text or "aaaa" in resp.text
    assert "bbbb-2222" in resp.text or "bbbb" in resp.text


def test_memory_rows_filter_by_agent(client: TestClient, mock_manager: MagicMock) -> None:
    client.get("/ui/memory/_rows?agent_id=agent-x")
    call_kwargs = mock_manager.list_records.call_args.kwargs
    assert call_kwargs.get("agent_id") == "agent-x"


def test_memory_rows_filter_by_type(client: TestClient, mock_manager: MagicMock) -> None:
    client.get("/ui/memory/_rows?memory_type=episodic")
    call_kwargs = mock_manager.list_records.call_args.kwargs
    assert call_kwargs.get("memory_type") == "episodic"


def test_memory_rows_filter_by_query(client: TestClient, mock_manager: MagicMock) -> None:
    client.get("/ui/memory/_rows?q=dark+mode")
    call_kwargs = mock_manager.list_records.call_args.kwargs
    assert call_kwargs.get("query") == "dark mode"


def test_memory_rows_filter_by_trust(client: TestClient, mock_manager: MagicMock) -> None:
    client.get("/ui/memory/_rows?min_trust=0.8")
    call_kwargs = mock_manager.list_records.call_args.kwargs
    assert call_kwargs.get("min_trust") == pytest.approx(0.8)


def test_memory_detail_shows_full_content(client: TestClient) -> None:
    resp = client.get("/ui/memory/_detail/aaaa-1111")
    assert resp.status_code == 200
    assert "The user prefers dark mode." in resp.text


def test_memory_detail_shows_provenance(client: TestClient) -> None:
    resp = client.get("/ui/memory/_detail/aaaa-1111")
    assert resp.status_code == 200
    assert "USER_INPUT" in resp.text or "source_type" in resp.text


def test_memory_delete_calls_manager_forget(client: TestClient, mock_manager: MagicMock) -> None:
    resp = client.delete("/ui/memory/aaaa-1111?memory_type=episodic")
    assert resp.status_code == 200
    mock_manager.forget.assert_awaited_once_with("aaaa-1111", memory_type="episodic")


def test_memory_delete_returns_empty_body(client: TestClient, mock_manager: MagicMock) -> None:
    resp = client.delete("/ui/memory/aaaa-1111?memory_type=episodic")
    assert resp.status_code == 200
    assert resp.text.strip() == ""


def test_memory_manager_exception_shows_empty_state(mock_manager: MagicMock) -> None:
    mock_manager.list_records = AsyncMock(side_effect=RuntimeError("backend down"))
    runner = MagicMock()
    agent_def = AgentDefinition(name="test", model="claude-sonnet-4-6")
    app = create_app(runner, agent_def, memory_manager=mock_manager)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/ui/memory/_rows")
    assert resp.status_code == 200
    assert "No memory entries found" in resp.text
