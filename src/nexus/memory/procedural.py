"""Procedural memory: store and retrieve learned workflows backed by Dapr state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.types import Procedure

_log = get_logger(__name__)

_ENTITY = "procedure"
_INDEX_KEY = "_index"


class ProceduralMemory:
    """CRUD store for learned procedures backed by a DaprStateStore.

    Key layout (within the agent's namespace):
    - ``procedure:{procedure_id}`` — individual procedure
    - ``procedure:_index`` — JSON list of procedure IDs for this agent

    Args:
        state_store: A DaprStateStore (or duck-typed equivalent).
        agent_id: Scopes all keys to this agent.
    """

    def __init__(self, state_store: Any, *, agent_id: str) -> None:
        self._store = state_store
        self._agent_id = agent_id
        self._index: list[str] = []

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def store(self, procedure: Procedure) -> Procedure:
        """Persist *procedure* and register it in the index.

        Returns the stored procedure unchanged.
        """
        await self._save_procedure(procedure)
        self._index.append(procedure.id)
        await self._save_index()
        _log.debug(
            "procedure_stored",
            procedure_id=procedure.id,
            name=procedure.name,
            agent=self._agent_id,
        )
        return procedure

    async def get(self, procedure_id: str) -> Procedure | None:
        """Load a single procedure by ID. Returns None if not found."""
        result, _ = await self._store.get(_ENTITY, procedure_id, Procedure)
        return result  # type: ignore[no-any-return]

    async def delete(self, procedure_id: str) -> None:
        """Remove a procedure and its entry from the index."""
        await self._store.delete(_ENTITY, procedure_id)
        if procedure_id in self._index:
            self._index.remove(procedure_id)
            await self._save_index()
        _log.debug("procedure_deleted", procedure_id=procedure_id, agent=self._agent_id)

    async def list_all(self) -> list[Procedure]:
        """Return all procedures for this agent."""
        if not self._index:
            return []
        procedures: list[Procedure] = []
        for pid in list(self._index):
            procedure = await self.get(pid)
            if procedure is not None:
                procedures.append(procedure)
        return procedures

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------

    async def record_outcome(self, procedure_id: str, *, success: bool) -> None:
        """Increment success or failure counter and update last_used timestamp.

        Does nothing if *procedure_id* is not found.
        """
        procedure = await self.get(procedure_id)
        if procedure is None:
            _log.debug("procedure_outcome_skipped_missing", procedure_id=procedure_id)
            return

        now = datetime.now(UTC)
        if success:
            updated = procedure.model_copy(
                update={"success_count": procedure.success_count + 1, "last_used": now}
            )
        else:
            updated = procedure.model_copy(
                update={"failure_count": procedure.failure_count + 1, "last_used": now}
            )

        await self._save_procedure(updated)
        _log.debug(
            "procedure_outcome_recorded",
            procedure_id=procedure_id,
            success=success,
            agent=self._agent_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _save_procedure(self, procedure: Procedure) -> None:
        await self._store.save(_ENTITY, procedure.id, procedure)

    async def _save_index(self) -> None:
        data = json.dumps(self._index).encode()
        await self._store.save(_ENTITY, _INDEX_KEY, data)
