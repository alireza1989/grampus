"""Tests for nexus.memory.vector.pinecone — PineconeVectorStore."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.core.errors import ToolError
from nexus.memory.vector.base import VectorRecord, VectorSearchResult
from nexus.memory.vector.pinecone import PineconeVectorStore

FAKE_VECTOR = [0.1] * 8


def _make_store(_client: MagicMock | None = None) -> PineconeVectorStore:
    return PineconeVectorStore(
        api_key="pk-test",
        index_host="https://idx.example.pinecone.io",
        namespace="nexus",
        _client=_client,
    )


def _make_mock_index() -> MagicMock:
    """Build a mock Pinecone async index context manager."""
    match = MagicMock()
    match.id = "r1"
    match.score = 0.9
    match.metadata = {"agent_id": "a1"}

    query_response = MagicMock()
    query_response.matches = [match]

    idx = MagicMock()
    idx.upsert = AsyncMock(return_value=None)
    idx.query = AsyncMock(return_value=query_response)
    idx.delete = AsyncMock(return_value=None)

    # Support async context manager on IndexAsyncio
    idx_ctx = MagicMock()
    idx_ctx.__aenter__ = AsyncMock(return_value=idx)
    idx_ctx.__aexit__ = AsyncMock(return_value=None)
    return idx_ctx, idx


def _make_mock_client(idx_ctx: MagicMock, idx: MagicMock) -> MagicMock:
    client = MagicMock()
    client.IndexAsyncio = MagicMock(return_value=idx_ctx)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


async def test_upsert_calls_index_upsert_with_correct_args() -> None:
    idx_ctx, idx = _make_mock_index()
    client = _make_mock_client(idx_ctx, idx)
    store = _make_store(_client=client)

    records = [VectorRecord(id="r1", vector=FAKE_VECTOR, payload={"x": 1})]
    await store.upsert(records)
    idx.upsert.assert_called_once()
    call_kwargs = idx.upsert.call_args
    assert call_kwargs is not None


async def test_upsert_passes_namespace() -> None:
    idx_ctx, idx = _make_mock_index()
    client = _make_mock_client(idx_ctx, idx)
    store = PineconeVectorStore(
        api_key="pk",
        index_host="https://idx.example.io",
        namespace="my-ns",
        _client=client,
    )
    await store.upsert([VectorRecord(id="r1", vector=FAKE_VECTOR)])
    call_kwargs = idx.upsert.call_args[1]
    assert call_kwargs.get("namespace") == "my-ns"


async def test_search_returns_vector_search_results() -> None:
    idx_ctx, idx = _make_mock_index()
    client = _make_mock_client(idx_ctx, idx)
    store = _make_store(_client=client)

    results = await store.search(FAKE_VECTOR, top_k=3)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], VectorSearchResult)


async def test_search_maps_score_and_metadata() -> None:
    idx_ctx, idx = _make_mock_index()
    client = _make_mock_client(idx_ctx, idx)
    store = _make_store(_client=client)

    results = await store.search(FAKE_VECTOR, top_k=3)
    assert results[0].id == "r1"
    assert results[0].score == pytest.approx(0.9)
    assert results[0].payload["agent_id"] == "a1"


async def test_search_passes_filter() -> None:
    idx_ctx, idx = _make_mock_index()
    client = _make_mock_client(idx_ctx, idx)
    store = _make_store(_client=client)

    filter_dict = {"agent_id": {"$eq": "a1"}}
    await store.search(FAKE_VECTOR, top_k=3, filter=filter_dict)
    call_kwargs = idx.query.call_args[1]
    assert call_kwargs.get("filter") == filter_dict


async def test_delete_calls_index_delete_with_ids() -> None:
    idx_ctx, idx = _make_mock_index()
    client = _make_mock_client(idx_ctx, idx)
    store = _make_store(_client=client)

    await store.delete(["r1", "r2"])
    idx.delete.assert_called_once()
    call_kwargs = idx.delete.call_args[1]
    assert "ids" in call_kwargs or idx.delete.call_args[0]


async def test_missing_sdk_raises_tool_error() -> None:
    """When pinecone is not installed, lazy import raises ToolError with install hint."""
    store = PineconeVectorStore(
        api_key="pk",
        index_host="https://idx.example.io",
    )
    # Simulate missing SDK by patching the import inside the method
    with patch.dict(sys.modules, {"pinecone": None}), pytest.raises(ToolError, match="pinecone"):
        await store.upsert([VectorRecord(id="r1", vector=FAKE_VECTOR)])


async def test_ensure_collection_with_injected_client() -> None:
    idx_ctx, idx = _make_mock_index()
    client = _make_mock_client(idx_ctx, idx)

    # describe_index / create_index responses
    desc = MagicMock()
    desc.status = MagicMock()
    desc.status.ready = True
    client.describe_index = AsyncMock(return_value=desc)
    client.create_index = AsyncMock(return_value=None)

    store = _make_store(_client=client)
    # Must not raise
    await store.ensure_collection(dimension=1536)
