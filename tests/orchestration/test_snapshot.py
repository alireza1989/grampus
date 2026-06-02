"""Tests for Phase D6 — Agent State Snapshots."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.core.errors import SnapshotError
from nexus.core.types import AgentDefinition, AgentState, AgentStatus, TokenUsage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent_state(
    agent_id: str = "test-agent",
    session_id: str = "sess-abc123",
    step: int = 3,
    status: AgentStatus = AgentStatus.COMPLETED,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        session_id=session_id,
        messages=[],
        status=status,
        current_step=step,
        total_token_usage=TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            model="test-model",
        ),
        metadata={},
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, 12, 5, 0, tzinfo=UTC),
    )


def _make_snapshot(**kwargs: Any) -> Any:
    from nexus.orchestration.snapshot import StateSnapshot

    defaults = dict(
        agent_id="test-agent",
        session_id="sess-abc123",
        state=_make_agent_state(),
        description="Test snapshot",
        tags=["test"],
        source_environment="ci",
    )
    defaults.update(kwargs)
    return StateSnapshot(**defaults)


# ===========================================================================
# TestStateSnapshot
# ===========================================================================


class TestStateSnapshot:
    def test_fields_default_populated(self) -> None:
        from nexus.orchestration.snapshot import StateSnapshot

        snap = StateSnapshot(
            agent_id="a",
            session_id="s",
            state=_make_agent_state("a", "s"),
        )
        uuid.UUID(snap.snapshot_id)  # must be valid UUID
        assert snap.created_at is not None
        assert snap.schema_version == "1.0"

    def test_json_round_trip(self) -> None:
        from nexus.orchestration.snapshot import StateSnapshot

        original = _make_snapshot()
        serialized = original.model_dump_json(indent=2)
        restored = StateSnapshot.model_validate(json.loads(serialized))
        assert restored.snapshot_id == original.snapshot_id
        assert restored.agent_id == original.agent_id
        assert restored.session_id == original.session_id
        assert restored.state.current_step == original.state.current_step

    def test_nexus_version_populated(self) -> None:
        snap = _make_snapshot()
        assert snap.nexus_version
        assert len(snap.nexus_version) > 0

    def test_tags_default_empty(self) -> None:
        from nexus.orchestration.snapshot import StateSnapshot

        snap = StateSnapshot(
            agent_id="a",
            session_id="s",
            state=_make_agent_state("a", "s"),
        )
        assert snap.tags == []

    def test_description_default_empty(self) -> None:
        from nexus.orchestration.snapshot import StateSnapshot

        snap = StateSnapshot(
            agent_id="a",
            session_id="s",
            state=_make_agent_state("a", "s"),
        )
        assert snap.description == ""


# ===========================================================================
# TestSnapshotManagerFileOps
# ===========================================================================


class TestSnapshotManagerFileOps:
    def test_to_file_creates_file(self, tmp_path: Any) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        snap = _make_snapshot()
        out = tmp_path / "snap.json"
        SnapshotManager.to_file(snap, out)
        assert out.exists()

    def test_to_file_creates_parent_dirs(self, tmp_path: Any) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        snap = _make_snapshot()
        out = tmp_path / "sub" / "dir" / "snap.json"
        SnapshotManager.to_file(snap, out)
        assert out.exists()

    def test_from_file_round_trips_snapshot(self, tmp_path: Any) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        snap = _make_snapshot()
        out = tmp_path / "snap.json"
        SnapshotManager.to_file(snap, out)
        restored = SnapshotManager.from_file(out)
        assert restored.snapshot_id == snap.snapshot_id
        assert restored.agent_id == snap.agent_id
        assert restored.state.current_step == snap.state.current_step

    def test_from_file_missing_file_raises(self, tmp_path: Any) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        with pytest.raises(SnapshotError) as exc_info:
            SnapshotManager.from_file(tmp_path / "nonexistent.json")
        assert exc_info.value.code == "FILE_NOT_FOUND"

    def test_from_file_invalid_json_raises(self, tmp_path: Any) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        bad = tmp_path / "bad.json"
        bad.write_text("not-valid-json{{{", encoding="utf-8")
        with pytest.raises(SnapshotError) as exc_info:
            SnapshotManager.from_file(bad)
        assert exc_info.value.code == "INVALID_SNAPSHOT"

    def test_from_dict_valid_data_returns_snapshot(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        snap = _make_snapshot()
        data = json.loads(snap.model_dump_json())
        restored = SnapshotManager.from_dict(data)
        assert restored.snapshot_id == snap.snapshot_id

    def test_from_dict_invalid_data_raises(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        with pytest.raises(SnapshotError) as exc_info:
            SnapshotManager.from_dict({"missing": "required_fields"})
        assert exc_info.value.code == "INVALID_SNAPSHOT"


# ===========================================================================
# TestSnapshotManagerExport
# ===========================================================================


class TestSnapshotManagerExport:
    def _mock_store(self, state: AgentState | None = None) -> MagicMock:
        store = MagicMock()
        store.get = AsyncMock(return_value=(state, "etag-123"))
        store.save = AsyncMock(return_value=None)
        return store

    @pytest.mark.asyncio
    async def test_export_session_calls_store_get(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        state = _make_agent_state()
        store = self._mock_store(state)
        mgr = SnapshotManager(state_store=store)
        await mgr.export_session("test-agent", "sess-abc123")
        store.get.assert_called_once_with("runner", "agent:test-agent:sess-abc123", AgentState)

    @pytest.mark.asyncio
    async def test_export_session_returns_snapshot_with_correct_agent_id(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        state = _make_agent_state(agent_id="my-agent")
        store = self._mock_store(state)
        mgr = SnapshotManager(state_store=store)
        snap = await mgr.export_session("my-agent", "sess-1")
        assert snap.agent_id == "my-agent"
        assert snap.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_export_session_state_not_found_raises(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        store = self._mock_store(None)
        mgr = SnapshotManager(state_store=store)
        with pytest.raises(SnapshotError) as exc_info:
            await mgr.export_session("test-agent", "sess-missing")
        assert exc_info.value.code == "STATE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_export_session_no_store_raises(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        mgr = SnapshotManager(None)
        with pytest.raises(SnapshotError) as exc_info:
            await mgr.export_session("test-agent", "sess-1")
        assert exc_info.value.code == "NO_STATE_STORE"

    @pytest.mark.asyncio
    async def test_export_session_description_propagated(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        store = self._mock_store(_make_agent_state())
        mgr = SnapshotManager(state_store=store)
        snap = await mgr.export_session("a", "s", description="my desc")
        assert snap.description == "my desc"

    @pytest.mark.asyncio
    async def test_export_session_tags_propagated(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        store = self._mock_store(_make_agent_state())
        mgr = SnapshotManager(state_store=store)
        snap = await mgr.export_session("a", "s", tags=["prod", "incident-42"])
        assert snap.tags == ["prod", "incident-42"]

    @pytest.mark.asyncio
    async def test_export_session_source_environment_propagated(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        store = self._mock_store(_make_agent_state())
        mgr = SnapshotManager(state_store=store)
        snap = await mgr.export_session("a", "s", source_environment="production")
        assert snap.source_environment == "production"


# ===========================================================================
# TestSnapshotManagerRestore
# ===========================================================================


class TestSnapshotManagerRestore:
    def _mock_store(self) -> MagicMock:
        store = MagicMock()
        store.save = AsyncMock(return_value=None)
        return store

    @pytest.mark.asyncio
    async def test_restore_writes_to_store(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        store = self._mock_store()
        mgr = SnapshotManager(state_store=store)
        snap = _make_snapshot()
        await mgr.restore_snapshot(snap)
        store.save.assert_called_once()
        call_args = store.save.call_args
        assert call_args[0][0] == "runner"
        assert call_args[0][1] == f"agent:{snap.agent_id}:{snap.session_id}"

    @pytest.mark.asyncio
    async def test_restore_no_store_raises(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        mgr = SnapshotManager(None)
        with pytest.raises(SnapshotError) as exc_info:
            await mgr.restore_snapshot(_make_snapshot())
        assert exc_info.value.code == "NO_STATE_STORE"

    @pytest.mark.asyncio
    async def test_restore_with_session_id_override(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        store = self._mock_store()
        mgr = SnapshotManager(state_store=store)
        snap = _make_snapshot()
        await mgr.restore_snapshot(snap, session_id_override="new-session")
        call_args = store.save.call_args
        assert call_args[0][1] == f"agent:{snap.agent_id}:new-session"

    @pytest.mark.asyncio
    async def test_restore_updated_at_refreshed_on_override(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        store = self._mock_store()
        mgr = SnapshotManager(state_store=store)
        snap = _make_snapshot()
        original_updated_at = snap.state.updated_at
        before = datetime.now(UTC)
        await mgr.restore_snapshot(snap, session_id_override="new-session")
        saved_state: AgentState = store.save.call_args[0][2]
        assert saved_state.updated_at >= before
        assert saved_state.updated_at != original_updated_at

    @pytest.mark.asyncio
    async def test_restore_original_session_unchanged_without_override(self) -> None:
        from nexus.orchestration.snapshot import SnapshotManager

        store = self._mock_store()
        mgr = SnapshotManager(state_store=store)
        snap = _make_snapshot()
        await mgr.restore_snapshot(snap)
        saved_state: AgentState = store.save.call_args[0][2]
        assert saved_state.session_id == snap.session_id


# ===========================================================================
# TestSnapshotEndpoints
# ===========================================================================


def _make_app(
    state: AgentState | None = None,
    has_store: bool = True,
    store_get_raises: bool = False,
) -> Any:
    """Build a minimal FastAPI app with mock state for endpoint tests."""
    from fastapi import FastAPI

    from nexus.server.routes import create_router

    app = FastAPI()
    app.include_router(create_router(has_memory=False))

    mock_runner = MagicMock()
    mock_agent_def = AgentDefinition(name="test", model="m")

    if has_store:
        mock_store = MagicMock()
        mock_store.get = AsyncMock(return_value=(state or _make_agent_state(), "etag"))
        mock_store.save = AsyncMock(return_value=None)
        app.state.state_store = mock_store
    else:
        if hasattr(app.state, "state_store"):
            del app.state.state_store

    app.state.runner = mock_runner
    app.state.agent_def = mock_agent_def
    return app


class TestSnapshotEndpoints:
    def test_get_snapshot_returns_200(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app(state=_make_agent_state())
        client = TestClient(app)
        resp = client.get("/agents/sess-abc123/snapshot")
        assert resp.status_code == 200

    def test_get_snapshot_content_disposition_header(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app(state=_make_agent_state())
        client = TestClient(app)
        resp = client.get("/agents/sess-abc123/snapshot")
        cd = resp.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "filename=" in cd

    def test_get_snapshot_session_not_found_returns_404(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        # Override store to return None
        app.state.state_store.get = AsyncMock(return_value=(None, None))
        client = TestClient(app)
        resp = client.get("/agents/missing-session/snapshot")
        assert resp.status_code == 404

    def test_get_snapshot_no_store_returns_503(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app(has_store=False)
        client = TestClient(app)
        resp = client.get("/agents/sess-abc123/snapshot")
        assert resp.status_code == 503

    def test_restore_snapshot_returns_200(self) -> None:
        from fastapi.testclient import TestClient

        snap = _make_snapshot()
        app = _make_app()
        client = TestClient(app)
        body = {"snapshot": json.loads(snap.model_dump_json())}
        resp = client.post("/agents/snapshot/restore", json=body)
        assert resp.status_code == 200

    def test_restore_snapshot_calls_store_save(self) -> None:
        from fastapi.testclient import TestClient

        snap = _make_snapshot()
        app = _make_app()
        client = TestClient(app)
        body = {"snapshot": json.loads(snap.model_dump_json())}
        client.post("/agents/snapshot/restore", json=body)
        app.state.state_store.save.assert_called_once()

    def test_restore_snapshot_with_session_override(self) -> None:
        from fastapi.testclient import TestClient

        snap = _make_snapshot()
        app = _make_app()
        client = TestClient(app)
        body = {
            "snapshot": json.loads(snap.model_dump_json()),
            "session_id_override": "overridden-session",
        }
        resp = client.post("/agents/snapshot/restore", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "overridden-session"

    def test_restore_snapshot_invalid_body_returns_400(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        client = TestClient(app)
        body = {"snapshot": {"not": "a valid snapshot"}}
        resp = client.post("/agents/snapshot/restore", json=body)
        assert resp.status_code == 400

    def test_restore_snapshot_no_store_returns_503(self) -> None:
        from fastapi.testclient import TestClient

        snap = _make_snapshot()
        app = _make_app(has_store=False)
        client = TestClient(app)
        body = {"snapshot": json.loads(snap.model_dump_json())}
        resp = client.post("/agents/snapshot/restore", json=body)
        assert resp.status_code == 503


# ===========================================================================
# TestSnapshotCLI
# ===========================================================================


class TestSnapshotCLI:
    def test_show_command_prints_summary(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from nexus.cli.commands.state import state
        from nexus.orchestration.snapshot import SnapshotManager

        snap = _make_snapshot(agent_id="my-agent", session_id="sess-xyz")
        snap_file = tmp_path / "snap.json"
        SnapshotManager.to_file(snap, snap_file)

        runner = CliRunner()
        result = runner.invoke(state, ["show", str(snap_file)])
        assert result.exit_code == 0, result.output
        assert "my-agent" in result.output

    def test_show_command_json_format(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from nexus.cli.commands.state import state
        from nexus.orchestration.snapshot import SnapshotManager

        snap = _make_snapshot()
        snap_file = tmp_path / "snap.json"
        SnapshotManager.to_file(snap, snap_file)

        runner = CliRunner()
        result = runner.invoke(state, ["show", str(snap_file), "--format", "json"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["agent_id"] == snap.agent_id

    def test_show_command_missing_file_exits_nonzero(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from nexus.cli.commands.state import state

        runner = CliRunner()
        result = runner.invoke(state, ["show", str(tmp_path / "nonexistent.json")])
        assert result.exit_code != 0

    def test_import_dry_run_prints_summary_no_write(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from nexus.cli.commands.state import state
        from nexus.orchestration.snapshot import SnapshotManager

        snap = _make_snapshot()
        snap_file = tmp_path / "snap.json"
        SnapshotManager.to_file(snap, snap_file)

        runner = CliRunner()
        with patch("nexus.cli.commands.state.SnapshotManager") as mock_mgr_cls:
            mock_mgr = MagicMock()
            mock_mgr_cls.return_value = mock_mgr
            mock_mgr_cls.from_file = SnapshotManager.from_file  # keep real from_file
            result = runner.invoke(state, ["import", str(snap_file), "--dry-run"])

        assert result.exit_code == 0, result.output
        # store.save should NOT have been called during dry-run
        mock_mgr.restore_snapshot.assert_not_called()

    def test_import_dry_run_shows_all_fields(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from nexus.cli.commands.state import state
        from nexus.orchestration.snapshot import SnapshotManager

        snap = _make_snapshot(
            agent_id="my-agent",
            session_id="sess-abc",
            description="Test description",
            tags=["prod"],
            source_environment="staging",
        )
        snap_file = tmp_path / "snap.json"
        SnapshotManager.to_file(snap, snap_file)

        runner = CliRunner()
        result = runner.invoke(state, ["import", str(snap_file), "--dry-run"])
        assert result.exit_code == 0, result.output
        output = result.output
        assert "my-agent" in output
        assert "sess-abc" in output
        assert snap.snapshot_id in output
