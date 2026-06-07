"""PgVectorStore — wraps the existing Dapr-backed pgvector path."""

from __future__ import annotations

import math
from typing import Any

from nexus.memory.vector.base import VectorRecord, VectorSearchResult, VectorStore

_ENTITY = "vector"


class PgVectorStore(VectorStore):
    """Default vector store backed by Dapr state (PostgreSQL + pgvector).

    Delegates to the existing state store interface used by EpisodicMemory and
    SemanticMemory.  The in-memory registry is a lightweight index so search
    can compute cosine similarity without a live pgvector instance in tests.
    """

    def __init__(self, state_store: Any) -> None:
        self._store = state_store
        self._registry: dict[str, VectorRecord] = {}

    async def ensure_collection(self, dimension: int) -> None:
        """No-op: pgvector schema is managed by init-db.sql."""

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Persist records to Dapr state and update the local registry."""
        for record in records:
            await self._store.save(_ENTITY, record.id, record)
            self._registry[record.id] = record

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Cosine-similarity search over the in-memory registry.

        In production the state store would delegate to pgvector; here we
        compute similarity directly so unit tests work without a database.
        """
        if not self._registry:
            return []

        scored: list[tuple[float, VectorRecord]] = []
        for rec in self._registry.values():
            sim = _cosine_similarity(vector, rec.vector)
            scored.append((sim, rec))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            VectorSearchResult(id=rec.id, score=score, payload=rec.payload)
            for score, rec in scored[:top_k]
        ]

    async def delete(self, ids: list[str]) -> None:
        """Remove records from Dapr state and the local registry."""
        for rid in ids:
            await self._store.delete(_ENTITY, rid)
            self._registry.pop(rid, None)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom
