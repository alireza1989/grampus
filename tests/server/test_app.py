"""Tests for grampus.server.app FastAPI application factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from grampus.core.errors import GrampusError, OrchestrationError
from grampus.core.types import AgentDefinition


@pytest.fixture
def mock_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-sonnet-4-6")


@pytest.fixture
def mock_runner() -> MagicMock:
    return MagicMock()


class TestCreateApp:
    def test_create_app_returns_fastapi_instance(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from fastapi import FastAPI

        from grampus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        assert isinstance(app, FastAPI)

    def test_app_stores_runner_on_state(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from grampus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        assert app.state.runner is mock_runner

    def test_app_stores_agent_def_on_state(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from grampus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        assert app.state.agent_def is mock_agent_def

    def test_app_stores_memory_manager_on_state(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from grampus.server.app import create_app

        mm = MagicMock()
        app = create_app(mock_runner, mock_agent_def, memory_manager=mm)
        assert app.state.memory_manager is mm

    def test_app_memory_manager_none_by_default(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from grampus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)
        assert app.state.memory_manager is None

    def test_app_has_grampus_error_handler(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from fastapi.testclient import TestClient

        from grampus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)

        @app.get("/test-error")
        async def _trigger() -> None:
            raise OrchestrationError("boom", code="TEST_CODE", hint="fix it")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-error")
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "TEST_CODE"
        assert body["hint"] == "fix it"

    def test_app_grampus_error_includes_message(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from fastapi.testclient import TestClient

        from grampus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)

        @app.get("/test-msg")
        async def _trigger2() -> None:
            raise GrampusError("something bad", code="BAD")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-msg")
        assert resp.status_code == 400
        assert "something bad" in resp.json()["error"]

    def test_app_generic_exception_returns_500(
        self, mock_runner: MagicMock, mock_agent_def: AgentDefinition
    ) -> None:
        from fastapi.testclient import TestClient

        from grampus.server.app import create_app

        app = create_app(mock_runner, mock_agent_def)

        @app.get("/test-500")
        async def _trigger3() -> None:
            raise RuntimeError("unexpected")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-500")
        assert resp.status_code == 500
        assert resp.json()["code"] == "INTERNAL_ERROR"
