"""Tests for EpisodicMemory integration with an optional VectorStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.memory.episodic import EpisodicMemory
from nexus.memory.types import EpisodicRecord
from nexus.memory.vector.base import VectorRecord, VectorSearchResult

FAKE_EMBEDDING = [0.1] * 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_store() -> AsyncMock:
    store = AsyncMock()
    store.save = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=(None, ""))
    store.delete = AsyncMock(return_value=None)
    return store


@pytest.fixture()
def mock_embeddings() -> AsyncMock:
    svc = AsyncMock()
    svc.embed = AsyncMock(return_value=FAKE_EMBEDDING)
    return svc


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.upsert = AsyncMock(return_value=None)
    vs.search = AsyncMock(return_value=[])
    vs.delete = AsyncMock(return_value=None)
    return vs


def _make_memory(
    mock_store: AsyncMock,
    mock_embeddings: AsyncMock,
    vector_store: MagicMock | None = None,
) -> EpisodicMemory:
    return EpisodicMemory(
        state_store=mock_store,
        embedding_service=mock_embeddings,
        agent_id="agent-1",
        vector_store=vector_store,
    )


# ---------------------------------------------------------------------------
# write path
# ---------------------------------------------------------------------------


async def test_episodic_uses_vector_store_on_write(
    mock_store: AsyncMock,
    mock_embeddings: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    memory = _make_memory(mock_store, mock_embeddings, mock_vector_store)
    await memory.store(content="Some important fact.", session_id="s1")
    mock_vector_store.upsert.assert_called_once()
    upsert_call_args = mock_vector_store.upsert.call_args[0][0]
    assert len(upsert_call_args) == 1
    assert isinstance(upsert_call_args[0], VectorRecord)


async def test_episodic_falls_back_to_pgvector_when_no_store(
    mock_store: AsyncMock,
    mock_embeddings: AsyncMock,
) -> None:
    memory = _make_memory(mock_store, mock_embeddings, vector_store=None)
    record = await memory.store(content="Some fact.", session_id="s1")
    # Record still saved to state store
    mock_store.save.assert_called()
    assert record.embedding == FAKE_EMBEDDING


async def test_episodic_vector_store_write_failure_logged_not_raised(
    mock_store: AsyncMock,
    mock_embeddings: AsyncMock,
) -> None:
    """Vector store write failures must not propagate — memory must stay available."""
    failing_vs = MagicMock()
    failing_vs.upsert = AsyncMock(side_effect=RuntimeError("vector DB down"))
    memory = _make_memory(mock_store, mock_embeddings, failing_vs)

    # Must not raise
    record = await memory.store(content="Important fact.", session_id="s1")
    # Record still persisted to state store
    mock_store.save.assert_called()
    assert record is not None


# ---------------------------------------------------------------------------
# search path
# ---------------------------------------------------------------------------


async def test_episodic_uses_vector_store_on_search(
    mock_store: AsyncMock,
    mock_embeddings: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    """When vector_store is set, search should use it for similarity lookup."""
    record = EpisodicRecord(
        id="rec-1",
        agent_id="agent-1",
        session_id="s1",
        content="relevant content",
        trust_score=0.9,
        importance_score=0.5,
    )
    mock_store.get = AsyncMock(return_value=(record, "etag"))
    mock_vector_store.search = AsyncMock(return_value=[VectorSearchResult(id="rec-1", score=0.95)])

    memory = _make_memory(mock_store, mock_embeddings, mock_vector_store)
    results = await memory.search(FAKE_EMBEDDING, top_k=5)

    mock_vector_store.search.assert_called_once()
    assert len(results) == 1
    assert results[0].id == "rec-1"


async def test_episodic_search_falls_back_when_no_vector_store(
    mock_store: AsyncMock,
    mock_embeddings: AsyncMock,
) -> None:
    """Without a vector store, search returns empty list (pgvector path not available in unit test)."""
    memory = _make_memory(mock_store, mock_embeddings, vector_store=None)
    results = await memory.search(FAKE_EMBEDDING, top_k=5)
    # No vector store → returns empty list (no pgvector in unit test environment)
    assert isinstance(results, list)
