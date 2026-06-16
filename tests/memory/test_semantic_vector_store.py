"""Tests for SemanticMemory integration with an optional VectorStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.memory.semantic import SemanticMemory
from grampus.memory.types import SemanticFact
from grampus.memory.vector.base import VectorRecord, VectorSearchResult

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
def mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.upsert = AsyncMock(return_value=None)
    vs.search = AsyncMock(return_value=[])
    return vs


def _make_fact(fact_id: str = "fact-1") -> SemanticFact:
    return SemanticFact(
        id=fact_id,
        subject="Python",
        predicate="is",
        object_value="a language",
        confidence=0.9,
        embedding=FAKE_EMBEDDING,
    )


def _make_memory(
    mock_store: AsyncMock,
    vector_store: MagicMock | None = None,
) -> SemanticMemory:
    return SemanticMemory(
        state_store=mock_store,
        agent_id="agent-1",
        vector_store=vector_store,
    )


# ---------------------------------------------------------------------------
# write path
# ---------------------------------------------------------------------------


async def test_semantic_uses_vector_store_on_write(
    mock_store: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    memory = _make_memory(mock_store, mock_vector_store)
    fact = _make_fact()
    await memory.store(fact)
    mock_vector_store.upsert.assert_called_once()
    upsert_args = mock_vector_store.upsert.call_args[0][0]
    assert isinstance(upsert_args[0], VectorRecord)
    assert upsert_args[0].id == fact.id


async def test_semantic_falls_back_to_pgvector_when_no_store(
    mock_store: AsyncMock,
) -> None:
    memory = _make_memory(mock_store, vector_store=None)
    fact = _make_fact()
    stored = await memory.store(fact)
    mock_store.save.assert_called()
    assert stored.id == fact.id


async def test_semantic_vector_store_write_failure_does_not_raise(
    mock_store: AsyncMock,
) -> None:
    """Vector store upsert failure must not propagate."""
    failing_vs = MagicMock()
    failing_vs.upsert = AsyncMock(side_effect=RuntimeError("vector DB down"))
    memory = _make_memory(mock_store, failing_vs)
    fact = _make_fact()

    # Must not raise
    stored = await memory.store(fact)
    mock_store.save.assert_called()
    assert stored is not None


# ---------------------------------------------------------------------------
# search path
# ---------------------------------------------------------------------------


async def test_semantic_uses_vector_store_on_search(
    mock_store: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    fact = _make_fact("fact-99")
    mock_store.get = AsyncMock(return_value=(fact, "etag"))
    mock_vector_store.search = AsyncMock(
        return_value=[VectorSearchResult(id="fact-99", score=0.92)]
    )

    memory = _make_memory(mock_store, mock_vector_store)
    results = await memory.search_similar(FAKE_EMBEDDING, top_k=5)

    mock_vector_store.search.assert_called_once()
    assert len(results) == 1
    assert results[0].id == "fact-99"


async def test_semantic_search_falls_back_when_no_vector_store(
    mock_store: AsyncMock,
) -> None:
    memory = _make_memory(mock_store, vector_store=None)
    results = await memory.search_similar(FAKE_EMBEDDING, top_k=5)
    assert isinstance(results, list)
