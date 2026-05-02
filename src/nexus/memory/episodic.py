"""Episodic memory: cross-session records with embeddings and importance scoring."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.types import EpisodicRecord

_log = get_logger(__name__)

_ENTITY = "episodic"
_INDEX_KEY = "_index"
_MAX_IMPORTANCE_WORDS = 200


class EpisodicMemory:
    """CRUD store for episodic memory records backed by a DaprStateStore.

    Key layout (within the agent's namespace):
    - ``episodic:{record_id}`` — individual record
    - ``episodic:_index`` — JSON list of record IDs for this agent

    Importance scoring proxy: ``min(word_count / 200, 1.0)`` — longer
    content is treated as more important. Phase 4 consolidation upgrades this
    with LLM-based extraction.
    """

    def __init__(
        self,
        state_store: Any,
        embedding_service: Any,
        *,
        agent_id: str,
    ) -> None:
        self._store = state_store
        self._embeddings = embedding_service
        self._agent_id = agent_id
        self._index: list[str] = []

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        *,
        session_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        provenance: str | None = None,
    ) -> EpisodicRecord:
        """Create and persist a new episodic record.

        If the embedding API fails the record is still saved with
        ``embedding=None`` (graceful degradation).
        """
        record_id = str(uuid.uuid4())
        embedding: list[float] | None = None
        try:
            embedding = await self._embeddings.embed(content)
        except Exception:
            _log.warning("episodic_embedding_failed", record_id=record_id)

        record = EpisodicRecord(
            id=record_id,
            agent_id=self._agent_id,
            user_id=user_id,
            session_id=session_id,
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            importance_score=_importance_score(content),
            provenance=provenance,
        )
        await self._save_record(record)
        self._index.append(record_id)
        await self._save_index()
        _log.debug("episodic_stored", agent=self._agent_id, record_id=record_id)
        return record

    async def get(self, record_id: str) -> EpisodicRecord | None:
        """Load a single record by ID. Returns None if not found."""
        result, _ = await self._store.get(_ENTITY, record_id, EpisodicRecord)
        return result  # type: ignore[no-any-return]

    async def delete(self, record_id: str) -> None:
        """Remove a record and its ID from the index."""
        await self._store.delete(_ENTITY, record_id)
        if record_id in self._index:
            self._index.remove(record_id)
            await self._save_index()
        _log.debug("episodic_deleted", agent=self._agent_id, record_id=record_id)

    async def update_metadata(self, record_id: str, metadata: dict[str, Any]) -> None:
        """Merge *metadata* into an existing record's metadata dict."""
        record, etag = await self._store.get(_ENTITY, record_id, EpisodicRecord)
        if record is None:
            return
        merged = {**record.metadata, **metadata}
        updated = record.model_copy(update={"metadata": merged})
        await self._save_record(updated, etag=etag if etag else None)
        _log.debug("episodic_metadata_updated", record_id=record_id)

    async def update_access(self, record_id: str) -> None:
        """Increment access_count and set last_accessed on an existing record."""
        record, etag = await self._store.get(_ENTITY, record_id, EpisodicRecord)
        if record is None:
            return
        updated = record.model_copy(
            update={
                "access_count": record.access_count + 1,
                "last_accessed": datetime.now(UTC),
            }
        )
        await self._save_record(updated, etag=etag if etag else None)
        _log.debug("episodic_access_updated", record_id=record_id)

    async def list_all(self) -> list[EpisodicRecord]:
        """Return all records for this agent (loads each individually)."""
        if not self._index:
            return []
        records: list[EpisodicRecord] = []
        for rid in list(self._index):
            rec = await self.get(rid)
            if rec is not None:
                records.append(rec)
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _save_record(self, record: EpisodicRecord, *, etag: str | None = None) -> None:
        await self._store.save(_ENTITY, record.id, record, etag=etag)

    async def _save_index(self) -> None:
        data = json.dumps(self._index).encode()
        await self._store.save(_ENTITY, _INDEX_KEY, data)


def _importance_score(content: str) -> float:
    """Simple word-count proxy for importance. Phase 4 replaces with LLM scoring."""
    words = len(content.split())
    return min(words / _MAX_IMPORTANCE_WORDS, 1.0)
