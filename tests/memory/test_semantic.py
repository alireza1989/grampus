"""Tests for grampus.memory.semantic — SemanticMemory CRUD and deduplication."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from grampus.memory.semantic import SemanticMemory
from grampus.memory.types import SemanticFact


def make_fact(
    subject: str = "user",
    predicate: str = "prefers",
    object_value: str = "dark mode",
    confidence: float = 0.9,
    source_ids: list[str] | None = None,
    fact_id: str | None = None,
) -> SemanticFact:
    return SemanticFact(
        id=fact_id or str(uuid.uuid4()),
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        confidence=confidence,
        source_episode_ids=source_ids or [],
    )


@pytest.fixture()
def mock_store() -> AsyncMock:
    store = AsyncMock()
    store.save = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=(None, ""))
    store.delete = AsyncMock(return_value=None)
    return store


@pytest.fixture()
def memory(mock_store: AsyncMock) -> SemanticMemory:
    return SemanticMemory(state_store=mock_store, agent_id="agent-1")


class TestSemanticMemoryStore:
    async def test_store_returns_semantic_fact(self, memory: SemanticMemory) -> None:
        fact = make_fact()
        result = await memory.store(fact)
        assert isinstance(result, SemanticFact)

    async def test_store_preserves_subject_predicate_object(self, memory: SemanticMemory) -> None:
        fact = make_fact(subject="alice", predicate="works_at", object_value="Acme")
        result = await memory.store(fact)
        assert result.subject == "alice"
        assert result.predicate == "works_at"
        assert result.object_value == "Acme"

    async def test_store_persists_to_state_store(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        await memory.store(make_fact())
        mock_store.save.assert_called()

    async def test_store_adds_to_index(self, memory: SemanticMemory, mock_store: AsyncMock) -> None:
        await memory.store(make_fact())
        # One save for the fact, one for the index
        assert mock_store.save.call_count >= 2

    async def test_store_new_fact_appears_in_index(self, memory: SemanticMemory) -> None:
        fact = make_fact()
        await memory.store(fact)
        assert fact.id in memory._index

    async def test_list_all_empty_initially(self, memory: SemanticMemory) -> None:
        assert await memory.list_all() == []


def make_stateful_memory() -> SemanticMemory:
    """Return a SemanticMemory backed by a simple in-memory dict store."""
    fact_store: dict[str, SemanticFact] = {}

    async def fake_save(entity: str, key: str, model_or_bytes: object, **_: object) -> None:
        if isinstance(model_or_bytes, SemanticFact):
            fact_store[key] = model_or_bytes

    async def fake_get(entity: str, key: str, cls: type) -> tuple:
        fact = fact_store.get(key)
        return fact, ("etag" if fact else "")

    async def fake_delete(entity: str, key: str, **_: object) -> None:
        fact_store.pop(key, None)

    store = AsyncMock()
    store.save = AsyncMock(side_effect=fake_save)
    store.get = AsyncMock(side_effect=fake_get)
    store.delete = AsyncMock(side_effect=fake_delete)
    return SemanticMemory(state_store=store, agent_id="agent-1")


class TestSemanticMemoryDeduplication:
    async def test_same_subject_predicate_does_not_create_duplicate(self) -> None:
        memory = make_stateful_memory()
        f1 = make_fact(subject="user", predicate="likes", object_value="Python", confidence=0.8)
        f2 = make_fact(subject="user", predicate="likes", object_value="Go", confidence=0.7)
        await memory.store(f1)
        await memory.store(f2)
        assert len(memory._index) == 1

    async def test_higher_confidence_incoming_replaces_object(self) -> None:
        memory = make_stateful_memory()
        f1 = make_fact(object_value="old value", confidence=0.5)
        f2 = make_fact(object_value="new value", confidence=0.9)
        await memory.store(f1)
        result = await memory.store(f2)
        assert result.object_value == "new value"
        assert result.id == f1.id  # kept original id

    async def test_lower_confidence_incoming_keeps_existing_object(self) -> None:
        memory = make_stateful_memory()
        f1 = make_fact(object_value="established value", confidence=0.9)
        f2 = make_fact(object_value="weaker value", confidence=0.3)
        await memory.store(f1)
        result = await memory.store(f2)
        assert result.object_value == "established value"

    async def test_dedup_merges_source_episode_ids(self) -> None:
        memory = make_stateful_memory()
        f1 = make_fact(source_ids=["ep-1"], confidence=0.7)
        f2 = make_fact(source_ids=["ep-2"], confidence=0.8)
        await memory.store(f1)
        result = await memory.store(f2)
        assert "ep-1" in result.source_episode_ids
        assert "ep-2" in result.source_episode_ids

    async def test_different_predicate_creates_new_fact(self) -> None:
        memory = make_stateful_memory()
        f1 = make_fact(subject="user", predicate="likes", object_value="Python")
        f2 = make_fact(subject="user", predicate="dislikes", object_value="Java")
        await memory.store(f1)
        await memory.store(f2)
        assert len(memory._index) == 2

    async def test_different_subject_creates_new_fact(self) -> None:
        memory = make_stateful_memory()
        f1 = make_fact(subject="alice", predicate="likes", object_value="Python")
        f2 = make_fact(subject="bob", predicate="likes", object_value="Python")
        await memory.store(f1)
        await memory.store(f2)
        assert len(memory._index) == 2


class TestSemanticMemoryGetDelete:
    async def test_get_returns_none_for_missing(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        mock_store.get.return_value = (None, "")
        result = await memory.get("nonexistent")
        assert result is None

    async def test_get_returns_fact_when_exists(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        fact = make_fact(fact_id="f-1")
        mock_store.get.return_value = (fact, "etag1")
        result = await memory.get("f-1")
        assert result is not None
        assert result.id == "f-1"

    async def test_delete_calls_store_delete(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        await memory.delete("f-1")
        mock_store.delete.assert_called()

    async def test_delete_removes_from_index(self, memory: SemanticMemory) -> None:
        fact = make_fact()
        await memory.store(fact)
        assert fact.id in memory._index
        await memory.delete(fact.id)
        assert fact.id not in memory._index


class TestSemanticMemoryFind:
    async def test_find_by_subject_returns_matching(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        f1 = make_fact(subject="alice", predicate="likes", fact_id="f1")
        f2 = make_fact(subject="alice", predicate="works_at", fact_id="f2")
        memory._index = ["f1", "f2"]

        async def fake_get(entity: str, key: str, cls: type) -> tuple:
            mapping = {"f1": f1, "f2": f2}
            return mapping.get(key), "etag"

        mock_store.get = AsyncMock(side_effect=fake_get)
        results = await memory.find_by_subject("alice")
        assert len(results) == 2
        assert all(r.subject == "alice" for r in results)

    async def test_find_by_subject_empty_for_no_match(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        f1 = make_fact(subject="alice", fact_id="f1")
        memory._index = ["f1"]
        mock_store.get = AsyncMock(return_value=(f1, "e"))
        results = await memory.find_by_subject("bob")
        assert results == []

    async def test_find_by_predicate_returns_matching(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        fact = make_fact(subject="user", predicate="prefers", fact_id="f1")
        memory._index = ["f1"]
        mock_store.get = AsyncMock(return_value=(fact, "e"))
        results = await memory.find_by_predicate("user", "prefers")
        assert len(results) == 1
        assert results[0].predicate == "prefers"

    async def test_find_by_predicate_empty_for_no_match(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        fact = make_fact(subject="user", predicate="likes", fact_id="f1")
        memory._index = ["f1"]
        mock_store.get = AsyncMock(return_value=(fact, "e"))
        results = await memory.find_by_predicate("user", "dislikes")
        assert results == []

    async def test_list_all_returns_all_stored_facts(
        self, memory: SemanticMemory, mock_store: AsyncMock
    ) -> None:
        f1 = make_fact(subject="a", predicate="p1", fact_id="f1")
        f2 = make_fact(subject="b", predicate="p2", fact_id="f2")
        memory._index = ["f1", "f2"]

        async def fake_get(entity: str, key: str, cls: type) -> tuple:
            mapping = {"f1": f1, "f2": f2}
            return mapping.get(key), "etag"

        mock_store.get = AsyncMock(side_effect=fake_get)
        results = await memory.list_all()
        assert len(results) == 2
