"""Tests for grampus.memory.vector.base — VectorStore ABC and data models."""

from __future__ import annotations

import pytest

from grampus.memory.vector.base import (
    VectorRecord,
    VectorSearchResult,
    VectorStore,
    VectorStoreType,
)


def test_vector_record_payload_defaults_empty() -> None:
    record = VectorRecord(id="r1", vector=[0.1, 0.2, 0.3])
    assert record.payload == {}


def test_vector_record_with_payload() -> None:
    record = VectorRecord(id="r2", vector=[0.1], payload={"agent_id": "a1"})
    assert record.payload["agent_id"] == "a1"


def test_vector_search_result_fields() -> None:
    result = VectorSearchResult(id="r1", score=0.95, payload={"key": "val"})
    assert result.id == "r1"
    assert result.score == 0.95
    assert result.payload["key"] == "val"


def test_vector_search_result_payload_defaults_empty() -> None:
    result = VectorSearchResult(id="r1", score=0.5)
    assert result.payload == {}


def test_vector_store_type_enum_values() -> None:
    assert VectorStoreType.PGVECTOR == "pgvector"
    assert VectorStoreType.PINECONE == "pinecone"
    assert VectorStoreType.WEAVIATE == "weaviate"
    assert VectorStoreType.QDRANT == "qdrant"


def test_vector_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


async def test_vector_store_close_default_noop() -> None:
    """close() has a default no-op implementation — subclasses need not override."""

    class MinimalStore(VectorStore):
        async def ensure_collection(self, dimension: int) -> None: ...

        async def upsert(self, records: list[VectorRecord]) -> None: ...

        async def search(
            self,
            vector: list[float],
            top_k: int,
            filter: dict | None = None,
        ) -> list[VectorSearchResult]:
            return []

        async def delete(self, ids: list[str]) -> None: ...

    store = MinimalStore()
    await store.close()  # must not raise
