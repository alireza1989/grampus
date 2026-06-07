"""Tests for nexus.memory.vector.weaviate — WeaviateVectorStore."""

from __future__ import annotations

import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.core.errors import ToolError
from nexus.memory.vector.base import VectorRecord, VectorSearchResult
from nexus.memory.vector.weaviate import WeaviateVectorStore, _to_weaviate_uuid

FAKE_VECTOR = [0.1] * 8


# ---------------------------------------------------------------------------
# UUID helpers
# ---------------------------------------------------------------------------


def test_to_weaviate_uuid_deterministic() -> None:
    uid1 = _to_weaviate_uuid("nexus-id-123")
    uid2 = _to_weaviate_uuid("nexus-id-123")
    assert uid1 == uid2


def test_to_weaviate_uuid_different_ids() -> None:
    uid1 = _to_weaviate_uuid("id-a")
    uid2 = _to_weaviate_uuid("id-b")
    assert uid1 != uid2


def test_to_weaviate_uuid_is_valid_uuid() -> None:
    uid = _to_weaviate_uuid("test-id")
    # Should not raise
    parsed = uuid.UUID(uid)
    assert parsed.version == 5


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_weaviate_obj(nexus_id: str, score: float = 0.85) -> MagicMock:
    obj = MagicMock()
    obj.properties = {"_nexus_id": nexus_id, "agent_id": "a1"}
    obj.metadata = MagicMock()
    obj.metadata.score = score
    obj.metadata.distance = None
    obj.uuid = _to_weaviate_uuid(nexus_id)
    return obj


def _make_collection(objects: list[MagicMock] | None = None) -> MagicMock:
    col = MagicMock()

    # data sub-object
    col.data = MagicMock()
    col.data.insert_many = AsyncMock(return_value=MagicMock(errors={}))
    col.data.replace = AsyncMock(return_value=None)
    col.data.delete_by_id = AsyncMock(return_value=None)

    # query sub-object
    query_result = MagicMock()
    query_result.objects = objects or []
    col.query = MagicMock()
    col.query.near_vector = AsyncMock(return_value=query_result)

    return col


def _make_weaviate_client(col: MagicMock, exists: bool = False) -> MagicMock:
    client = MagicMock()
    client.collections = MagicMock()
    client.collections.exists = AsyncMock(return_value=exists)
    client.collections.create = AsyncMock(return_value=None)
    client.collections.get = MagicMock(return_value=col)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _make_store(client: MagicMock) -> WeaviateVectorStore:
    return WeaviateVectorStore(collection_name="NexusMemory", _client=client)


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_inserts_new_objects() -> None:
    col = _make_collection()
    client = _make_weaviate_client(col)
    store = _make_store(client)

    records = [VectorRecord(id="r1", vector=FAKE_VECTOR, payload={"x": 1})]
    await store.upsert(records)
    col.data.insert_many.assert_called_once()


async def test_upsert_replaces_existing_objects() -> None:
    """When insert_many returns errors with ObjectAlreadyExistsException, replace is called."""
    col = _make_collection()
    # Simulate insert error for r1
    error_mock = MagicMock()
    error_mock.error = MagicMock()
    error_mock.error.__class__.__name__ = "ObjectAlreadyExistsException"
    error_mock.original_uuid = _to_weaviate_uuid("r1")
    col.data.insert_many = AsyncMock(return_value=MagicMock(errors={"0": error_mock}))

    client = _make_weaviate_client(col)
    store = _make_store(client)

    records = [VectorRecord(id="r1", vector=FAKE_VECTOR)]
    await store.upsert(records)
    # replace should be called for the duplicate
    col.data.replace.assert_called_once()


async def test_upsert_stores_nexus_id_in_payload() -> None:
    col = _make_collection()
    client = _make_weaviate_client(col)
    store = _make_store(client)

    records = [VectorRecord(id="my-nexus-id", vector=FAKE_VECTOR)]
    await store.upsert(records)

    call_args = col.data.insert_many.call_args
    data_objects = call_args[0][0]  # first positional arg
    assert any(obj.properties.get("_nexus_id") == "my-nexus-id" for obj in data_objects)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_maps_nexus_id_from_payload() -> None:
    obj = _make_weaviate_obj("nexus-id-99", score=0.9)
    col = _make_collection(objects=[obj])
    client = _make_weaviate_client(col)
    store = _make_store(client)

    results = await store.search(FAKE_VECTOR, top_k=5)
    assert len(results) == 1
    assert results[0].id == "nexus-id-99"


async def test_search_returns_score() -> None:
    obj = _make_weaviate_obj("r1", score=0.75)
    col = _make_collection(objects=[obj])
    client = _make_weaviate_client(col)
    store = _make_store(client)

    results = await store.search(FAKE_VECTOR, top_k=5)
    assert results[0].score == pytest.approx(0.75)


async def test_search_returns_vector_search_result_type() -> None:
    obj = _make_weaviate_obj("r1")
    col = _make_collection(objects=[obj])
    client = _make_weaviate_client(col)
    store = _make_store(client)

    results = await store.search(FAKE_VECTOR, top_k=5)
    assert all(isinstance(r, VectorSearchResult) for r in results)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_calls_delete_by_id() -> None:
    col = _make_collection()
    client = _make_weaviate_client(col)
    store = _make_store(client)

    await store.delete(["r1"])
    col.data.delete_by_id.assert_called_once_with(_to_weaviate_uuid("r1"))


async def test_delete_silent_on_not_found() -> None:
    col = _make_collection()

    # Simulate a not-found exception
    class _NotFoundErr(Exception):
        pass

    col.data.delete_by_id = AsyncMock(side_effect=_NotFoundErr("not found"))
    client = _make_weaviate_client(col)
    store = _make_store(client)

    # Should not raise
    await store.delete(["nonexistent"])


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


async def test_ensure_collection_creates_if_missing() -> None:
    col = _make_collection()
    client = _make_weaviate_client(col, exists=False)
    store = _make_store(client)

    await store.ensure_collection(dimension=1536)
    client.collections.create.assert_called_once()


async def test_ensure_collection_idempotent() -> None:
    """If collection already exists, create should not be called."""
    col = _make_collection()
    client = _make_weaviate_client(col, exists=True)
    store = _make_store(client)

    await store.ensure_collection(dimension=1536)
    client.collections.create.assert_not_called()


# ---------------------------------------------------------------------------
# missing SDK
# ---------------------------------------------------------------------------


async def test_missing_sdk_raises_tool_error() -> None:
    store = WeaviateVectorStore(collection_name="NexusMemory")
    with patch.dict(sys.modules, {"weaviate": None}), pytest.raises(ToolError, match="weaviate"):
        await store.upsert([VectorRecord(id="r1", vector=FAKE_VECTOR)])
