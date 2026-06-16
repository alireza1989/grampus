"""Tests for grampus.memory.vector.qdrant — QdrantVectorStore."""

from __future__ import annotations

import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grampus.core.errors import ToolError
from grampus.memory.vector.base import VectorRecord, VectorSearchResult
from grampus.memory.vector.qdrant import QdrantVectorStore, _to_qdrant_uuid

FAKE_VECTOR = [0.1] * 8


def _make_scored_point(grampus_id: str, score: float = 0.88) -> MagicMock:
    pt = MagicMock()
    pt.score = score
    pt.payload = {"_grampus_id": grampus_id, "agent_id": "a1"}
    pt.id = _to_qdrant_uuid(grampus_id)
    return pt


def _make_qdrant_client(
    points: list[MagicMock] | None = None, collections_names: list[str] | None = None
) -> MagicMock:
    client = MagicMock()
    client.upsert = AsyncMock(return_value=None)

    query_result = MagicMock()
    query_result.points = points or []
    client.query_points = AsyncMock(return_value=query_result)

    client.delete = AsyncMock(return_value=None)
    client.close = AsyncMock(return_value=None)

    col_names = collections_names or []
    collection_list = MagicMock()
    col_mocks = []
    for n in col_names:
        col_mock = MagicMock()
        col_mock.name = n
        col_mocks.append(col_mock)
    collection_list.collections = col_mocks
    client.get_collections = AsyncMock(return_value=collection_list)
    client.create_collection = AsyncMock(return_value=None)

    return client


def _make_store(client: MagicMock) -> QdrantVectorStore:
    return QdrantVectorStore(collection_name="grampus_memory", _client=client)


# ---------------------------------------------------------------------------
# UUID helper
# ---------------------------------------------------------------------------


def test_to_qdrant_uuid_deterministic() -> None:
    u1 = _to_qdrant_uuid("grampus-id-abc")
    u2 = _to_qdrant_uuid("grampus-id-abc")
    assert u1 == u2


def test_to_qdrant_uuid_different_ids() -> None:
    assert _to_qdrant_uuid("id-a") != _to_qdrant_uuid("id-b")


def test_to_qdrant_uuid_is_valid_uuid() -> None:
    parsed = uuid.UUID(_to_qdrant_uuid("test"))
    assert parsed.version == 5


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_upsert_with_point_structs() -> None:
    client = _make_qdrant_client()
    store = _make_store(client)

    await store.upsert([VectorRecord(id="r1", vector=FAKE_VECTOR, payload={"k": "v"})])
    client.upsert.assert_called_once()
    call_kwargs = client.upsert.call_args[1]
    assert call_kwargs["collection_name"] == "grampus_memory"
    assert len(call_kwargs["points"]) == 1


async def test_upsert_stores_grampus_id_in_payload() -> None:
    client = _make_qdrant_client()
    store = _make_store(client)

    await store.upsert([VectorRecord(id="my-grampus-id", vector=FAKE_VECTOR)])
    points = client.upsert.call_args[1]["points"]
    assert points[0].payload["_grampus_id"] == "my-grampus-id"


async def test_upsert_uses_uuid5_for_point_id() -> None:
    client = _make_qdrant_client()
    store = _make_store(client)

    await store.upsert([VectorRecord(id="r1", vector=FAKE_VECTOR)])
    points = client.upsert.call_args[1]["points"]
    assert points[0].id == _to_qdrant_uuid("r1")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_uses_query_points_not_search() -> None:
    """Must use query_points(), NOT the deprecated search() method."""
    client = _make_qdrant_client()
    store = _make_store(client)

    await store.search(FAKE_VECTOR, top_k=5)
    client.query_points.assert_called_once()
    client.search.assert_not_called()  # type: ignore[attr-defined]


async def test_search_maps_grampus_id_from_payload() -> None:
    pt = _make_scored_point("grampus-abc", score=0.9)
    client = _make_qdrant_client(points=[pt])
    store = _make_store(client)

    results = await store.search(FAKE_VECTOR, top_k=5)
    assert results[0].id == "grampus-abc"


async def test_search_returns_score() -> None:
    pt = _make_scored_point("r1", score=0.77)
    client = _make_qdrant_client(points=[pt])
    store = _make_store(client)

    results = await store.search(FAKE_VECTOR, top_k=5)
    assert results[0].score == pytest.approx(0.77)


async def test_search_returns_vector_search_result_type() -> None:
    pt = _make_scored_point("r1")
    client = _make_qdrant_client(points=[pt])
    store = _make_store(client)

    results = await store.search(FAKE_VECTOR, top_k=5)
    assert all(isinstance(r, VectorSearchResult) for r in results)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_uses_point_ids_list() -> None:
    client = _make_qdrant_client()
    store = _make_store(client)

    await store.delete(["r1", "r2"])
    client.delete.assert_called_once()
    call_kwargs = client.delete.call_args[1]
    assert call_kwargs["collection_name"] == "grampus_memory"


async def test_delete_uses_uuid5_mapping() -> None:
    client = _make_qdrant_client()
    store = _make_store(client)

    await store.delete(["r1"])
    call_kwargs = client.delete.call_args[1]
    selector = call_kwargs["points_selector"]
    assert _to_qdrant_uuid("r1") in selector.points


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


async def test_ensure_collection_creates_if_missing() -> None:
    client = _make_qdrant_client(collections_names=[])
    store = _make_store(client)

    await store.ensure_collection(dimension=1536)
    client.create_collection.assert_called_once()
    call_kwargs = client.create_collection.call_args[1]
    assert call_kwargs["collection_name"] == "grampus_memory"


async def test_ensure_collection_idempotent() -> None:
    """Already exists → create_collection should NOT be called."""
    client = _make_qdrant_client(collections_names=["grampus_memory"])
    store = _make_store(client)

    await store.ensure_collection(dimension=1536)
    client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_calls_client_close() -> None:
    client = _make_qdrant_client()
    store = _make_store(client)

    await store.close()
    client.close.assert_called_once()


# ---------------------------------------------------------------------------
# missing SDK
# ---------------------------------------------------------------------------


async def test_missing_sdk_raises_tool_error() -> None:
    store = QdrantVectorStore(collection_name="grampus_memory")
    with patch.dict(sys.modules, {"qdrant_client": None}), pytest.raises(ToolError, match="qdrant"):
        await store.upsert([VectorRecord(id="r1", vector=FAKE_VECTOR)])
