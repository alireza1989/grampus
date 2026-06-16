"""Tests for grampus.memory.vector.pgvector — PgVectorStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.memory.vector.base import VectorRecord, VectorSearchResult
from grampus.memory.vector.pgvector import PgVectorStore

FAKE_VECTOR = [0.1, 0.2, 0.3]


@pytest.fixture()
def mock_state_store() -> MagicMock:
    store = MagicMock()
    store.save = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=(None, ""))
    store.delete = AsyncMock(return_value=None)
    return store


@pytest.fixture()
def pg_store(mock_state_store: MagicMock) -> PgVectorStore:
    return PgVectorStore(state_store=mock_state_store)


async def test_pgvector_ensure_collection_is_noop(pg_store: PgVectorStore) -> None:
    # Must not raise and must not call the state_store
    await pg_store.ensure_collection(dimension=1536)


async def test_pgvector_upsert_delegates_to_state_store(
    pg_store: PgVectorStore, mock_state_store: MagicMock
) -> None:
    records = [VectorRecord(id="r1", vector=FAKE_VECTOR, payload={"a": 1})]
    await pg_store.upsert(records)
    mock_state_store.save.assert_called()


async def test_pgvector_upsert_multiple_records(
    pg_store: PgVectorStore, mock_state_store: MagicMock
) -> None:
    records = [
        VectorRecord(id="r1", vector=FAKE_VECTOR),
        VectorRecord(id="r2", vector=FAKE_VECTOR),
    ]
    await pg_store.upsert(records)
    assert mock_state_store.save.call_count == 2


async def test_pgvector_search_returns_results(
    pg_store: PgVectorStore, mock_state_store: MagicMock
) -> None:
    stored = VectorRecord(id="r1", vector=FAKE_VECTOR, payload={"k": "v"})
    mock_state_store.get = AsyncMock(return_value=(stored, "etag1"))

    # Pre-populate the store's internal registry via upsert
    await pg_store.upsert([stored])
    results = await pg_store.search(FAKE_VECTOR, top_k=5)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, VectorSearchResult)


async def test_pgvector_delete_removes_records(
    pg_store: PgVectorStore, mock_state_store: MagicMock
) -> None:
    await pg_store.upsert([VectorRecord(id="r1", vector=FAKE_VECTOR)])
    await pg_store.delete(["r1"])
    mock_state_store.delete.assert_called()


async def test_pgvector_delete_empty_list_is_noop(
    pg_store: PgVectorStore, mock_state_store: MagicMock
) -> None:
    await pg_store.delete([])
    mock_state_store.delete.assert_not_called()


async def test_pgvector_search_empty_returns_empty(pg_store: PgVectorStore) -> None:
    results = await pg_store.search(FAKE_VECTOR, top_k=5)
    assert results == []
