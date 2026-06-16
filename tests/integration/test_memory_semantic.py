"""Integration tests for SemanticMemory with FakeStateStore."""

from __future__ import annotations

import uuid

import pytest

from grampus.memory.types import SemanticFact
from tests.integration.conftest import FakeStateStore


def _make_fact(
    subject: str = "Alice",
    predicate: str = "likes",
    object_value: str = "Python",
    confidence: float = 0.9,
    source_ids: list[str] | None = None,
) -> SemanticFact:
    return SemanticFact(
        id=str(uuid.uuid4()),
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        confidence=confidence,
        source_episode_ids=source_ids or [],
    )


@pytest.mark.integration
class TestSemanticMemoryIntegration:
    async def test_store_and_get_fact(self, semantic_memory: object) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm: SemanticMemory = semantic_memory  # type: ignore[assignment]
        fact = _make_fact()
        stored = await sm.store(fact)
        fetched = await sm.get(stored.id)
        assert fetched is not None
        assert fetched.subject == "Alice"
        assert fetched.predicate == "likes"
        assert fetched.object_value == "Python"

    async def test_deduplication_merges_same_subject_predicate(
        self, semantic_memory: object
    ) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm: SemanticMemory = semantic_memory  # type: ignore[assignment]
        fact1 = _make_fact(object_value="JavaScript", confidence=0.7)
        fact2 = _make_fact(object_value="Python", confidence=0.9)

        await sm.store(fact1)
        await sm.store(fact2)

        all_facts = await sm.list_all()
        assert len(all_facts) == 1
        assert all_facts[0].object_value == "Python"

    async def test_higher_confidence_incoming_wins(self, semantic_memory: object) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm: SemanticMemory = semantic_memory  # type: ignore[assignment]
        low = _make_fact(object_value="low-conf", confidence=0.3)
        high = _make_fact(object_value="high-conf", confidence=0.95)

        await sm.store(low)
        await sm.store(high)

        all_facts = await sm.list_all()
        assert len(all_facts) == 1
        assert all_facts[0].object_value == "high-conf"
        assert all_facts[0].confidence == 0.95

    async def test_lower_confidence_incoming_does_not_override(
        self, semantic_memory: object
    ) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm: SemanticMemory = semantic_memory  # type: ignore[assignment]
        high = _make_fact(object_value="existing-high", confidence=0.9)
        low = _make_fact(object_value="incoming-low", confidence=0.3)

        await sm.store(high)
        await sm.store(low)

        all_facts = await sm.list_all()
        assert all_facts[0].object_value == "existing-high"

    async def test_source_episode_ids_merged_on_dedup(self, semantic_memory: object) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm: SemanticMemory = semantic_memory  # type: ignore[assignment]
        ep1 = str(uuid.uuid4())
        ep2 = str(uuid.uuid4())
        fact1 = _make_fact(confidence=0.7, source_ids=[ep1])
        fact2 = _make_fact(confidence=0.9, source_ids=[ep2])

        await sm.store(fact1)
        await sm.store(fact2)

        all_facts = await sm.list_all()
        merged_ids = all_facts[0].source_episode_ids
        assert ep1 in merged_ids
        assert ep2 in merged_ids

    async def test_find_by_subject_returns_correct_facts(self, semantic_memory: object) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm: SemanticMemory = semantic_memory  # type: ignore[assignment]
        await sm.store(_make_fact(subject="Alice", predicate="likes", object_value="Python"))
        await sm.store(_make_fact(subject="Bob", predicate="likes", object_value="Go"))
        await sm.store(
            SemanticFact(
                id=str(uuid.uuid4()),
                subject="Alice",
                predicate="knows",
                object_value="Dapr",
                confidence=0.8,
                source_episode_ids=[],
            )
        )

        alice_facts = await sm.find_by_subject("Alice")
        assert len(alice_facts) == 2
        assert all(f.subject == "Alice" for f in alice_facts)

    async def test_find_by_predicate_returns_correct_facts(self, semantic_memory: object) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm: SemanticMemory = semantic_memory  # type: ignore[assignment]
        await sm.store(_make_fact(subject="Alice", predicate="likes", object_value="Python"))
        await sm.store(
            SemanticFact(
                id=str(uuid.uuid4()),
                subject="Alice",
                predicate="knows",
                object_value="Django",
                confidence=0.8,
                source_episode_ids=[],
            )
        )

        results = await sm.find_by_predicate("Alice", "likes")
        assert len(results) == 1
        assert results[0].predicate == "likes"

    async def test_delete_removes_from_index(self, semantic_memory: object) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm: SemanticMemory = semantic_memory  # type: ignore[assignment]
        fact = _make_fact()
        stored = await sm.store(fact)
        await sm.delete(stored.id)

        all_facts = await sm.list_all()
        assert all(f.id != stored.id for f in all_facts)
        assert await sm.get(stored.id) is None

    async def test_facts_persist_across_instances(
        self,
        fake_state_store: FakeStateStore,
    ) -> None:
        from grampus.memory.semantic import SemanticMemory

        sm1 = SemanticMemory(fake_state_store, agent_id="persist-agent")
        fact = _make_fact()
        stored = await sm1.store(fact)

        sm2 = SemanticMemory(fake_state_store, agent_id="persist-agent")
        fetched = await sm2.get(stored.id)
        assert fetched is not None
        assert fetched.subject == "Alice"
