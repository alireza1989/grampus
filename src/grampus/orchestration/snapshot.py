"""Agent state snapshot — export, persist, and restore AgentState across environments."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from grampus.core.errors import SnapshotError
from grampus.core.types import AgentState


def _grampus_version() -> str:
    try:
        from importlib.metadata import version

        return version("grampus-ai")
    except Exception:
        return "0.1.0"


class StateSnapshot(BaseModel):
    """Portable, versioned export of an AgentState.

    Designed for debugging, migration between environments, test fixture
    creation, and disaster recovery. Schema-versioned for forward compatibility.
    """

    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "1.0"
    grampus_version: str = Field(default_factory=_grampus_version)
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_id: str
    session_id: str
    state: AgentState
    event_log_count: int = 0
    tags: list[str] = Field(default_factory=list)
    source_environment: str = ""


class SnapshotManager:
    """Export and restore AgentState snapshots.

    Works standalone (state_store=None) for file-only operations, or with a
    DaprStateStore for live export/restore against a running Nexus server.

    Args:
        state_store: Optional Dapr state store. Required for export_session()
            and restore_snapshot() to reach live state. None is valid for
            from_file() / to_file() / from_dict() operations.
    """

    def __init__(self, state_store: Any | None = None) -> None:
        self._store = state_store

    async def export_session(
        self,
        agent_id: str,
        session_id: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
        source_environment: str = "",
    ) -> StateSnapshot:
        """Load live state from the store and wrap it in a StateSnapshot.

        Args:
            agent_id: The agent identifier.
            session_id: The session identifier.
            description: Human-readable description embedded in the snapshot.
            tags: User-supplied labels.
            source_environment: Label for the originating environment.

        Returns:
            A StateSnapshot containing the current AgentState.

        Raises:
            SnapshotError: If no state store is configured or the session is not found.
        """
        if self._store is None:
            raise SnapshotError(
                "Cannot export live session without a state_store.",
                code="NO_STATE_STORE",
                hint="Pass a DaprStateStore to SnapshotManager or use from_file().",
            )
        entity_id = f"agent:{agent_id}:{session_id}"
        state, _ = await self._store.get("runner", entity_id, AgentState)
        if state is None:
            raise SnapshotError(
                f"No state found for agent='{agent_id}' session='{session_id}'.",
                code="STATE_NOT_FOUND",
                hint="Verify the agent has run at least once and state_store is configured.",
            )
        return StateSnapshot(
            agent_id=agent_id,
            session_id=session_id,
            state=state,
            description=description,
            tags=tags or [],
            source_environment=source_environment,
        )

    async def restore_snapshot(
        self,
        snapshot: StateSnapshot,
        *,
        session_id_override: str | None = None,
    ) -> None:
        """Write the snapshot's state back to the state store.

        Args:
            snapshot: The snapshot to restore.
            session_id_override: If given, write to this session ID instead
                of the snapshot's original session ID.

        Raises:
            SnapshotError: If no state store is configured.
        """
        if self._store is None:
            raise SnapshotError(
                "Cannot restore snapshot without a state_store.",
                code="NO_STATE_STORE",
            )
        target_session = session_id_override or snapshot.session_id
        state = (
            snapshot.state.model_copy(
                update={"session_id": target_session, "updated_at": datetime.now(UTC)}
            )
            if session_id_override
            else snapshot.state
        )
        entity_id = f"agent:{snapshot.agent_id}:{target_session}"
        await self._store.save("runner", entity_id, state)

    @staticmethod
    def to_file(snapshot: StateSnapshot, path: Path) -> None:
        """Write snapshot to a JSON file. Creates parent directories if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def from_file(path: Path) -> StateSnapshot:
        """Load and validate a snapshot from a JSON file.

        Raises:
            SnapshotError: If the file does not exist or contains invalid data.
        """
        if not path.exists():
            raise SnapshotError(
                f"Snapshot file not found: {path}",
                code="FILE_NOT_FOUND",
                hint="Check the path and try again.",
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SnapshotManager.from_dict(data)
        except SnapshotError:
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            raise SnapshotError(
                f"Invalid snapshot file '{path}': {exc}",
                code="INVALID_SNAPSHOT",
                hint="Ensure the file was exported by Nexus and has not been manually edited.",
            ) from exc

    @staticmethod
    def from_dict(data: dict[str, Any]) -> StateSnapshot:
        """Parse and validate a snapshot from a dict (e.g. from HTTP body).

        Raises:
            SnapshotError: If validation fails.
        """
        try:
            return StateSnapshot.model_validate(data)
        except Exception as exc:
            raise SnapshotError(
                f"Snapshot validation failed: {exc}",
                code="INVALID_SNAPSHOT",
            ) from exc
