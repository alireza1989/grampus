"""Integration tests for EpisodicMemory with FakeStateStore."""

from __future__ import annotations

import pytest

from tests.integration.conftest import FakeEmbeddingService, FakeStateStore


@pytest.mark.integration
class TestEpisodicMemoryIntegration:
    async def test_store_and_retrieve_record(
        self,
        episodic_memory: object,
    ) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em: EpisodicMemory = episodic_memory  # type: ignore[assignment]
        record = await em.store("Alice likes Python.", session_id="s1")
        assert record.id != ""
        fetched = await em.get(record.id)
        assert fetched is not None
        assert fetched.content == "Alice likes Python."
        assert fetched.agent_id == "test-agent"
        assert fetched.session_id == "s1"

    async def test_records_persist_across_memory_instances(
        self,
        fake_state_store: FakeStateStore,
        fake_embedding_service: FakeEmbeddingService,
    ) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em1 = EpisodicMemory(fake_state_store, fake_embedding_service, agent_id="agent-x")
        record = await em1.store("User prefers dark mode.", session_id="s1")

        em2 = EpisodicMemory(fake_state_store, fake_embedding_service, agent_id="agent-x")
        fetched = await em2.get(record.id)
        assert fetched is not None
        assert fetched.content == "User prefers dark mode."

    async def test_update_metadata_marks_consolidated(self, episodic_memory: object) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em: EpisodicMemory = episodic_memory  # type: ignore[assignment]
        record = await em.store("Some content.", session_id="s1")
        await em.update_metadata(record.id, {"consolidated": True})

        updated = await em.get(record.id)
        assert updated is not None
        assert updated.metadata.get("consolidated") is True

    async def test_list_all_returns_all_stored(self, episodic_memory: object) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em: EpisodicMemory = episodic_memory  # type: ignore[assignment]
        for i in range(4):
            await em.store(f"Record {i}", session_id="s1")
        records = await em.list_all()
        assert len(records) == 4

    async def test_delete_removes_record(self, episodic_memory: object) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em: EpisodicMemory = episodic_memory  # type: ignore[assignment]
        record = await em.store("To be deleted.", session_id="s1")
        await em.delete(record.id)
        fetched = await em.get(record.id)
        assert fetched is None
        remaining = await em.list_all()
        assert all(r.id != record.id for r in remaining)

    async def test_importance_score_reflects_word_count(self, episodic_memory: object) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em: EpisodicMemory = episodic_memory  # type: ignore[assignment]
        short = await em.store("Hi.", session_id="s1")
        long_text = " ".join(["word"] * 200)
        long_ = await em.store(long_text, session_id="s1")
        assert short.importance_score < long_.importance_score

    async def test_embedding_stored_and_retrieved(
        self,
        episodic_memory: object,
    ) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em: EpisodicMemory = episodic_memory  # type: ignore[assignment]
        record = await em.store("Python async uses async/await.", session_id="s1")
        assert record.embedding is not None
        assert len(record.embedding) > 0

    async def test_access_count_increments_on_update_access(self, episodic_memory: object) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em: EpisodicMemory = episodic_memory  # type: ignore[assignment]
        record = await em.store("Accessed record.", session_id="s1")
        assert record.access_count == 0

        await em.update_access(record.id)
        updated = await em.get(record.id)
        assert updated is not None
        assert updated.access_count == 1

    async def test_store_with_provenance(self, episodic_memory: object) -> None:
        from nexus.memory.episodic import EpisodicMemory

        em: EpisodicMemory = episodic_memory  # type: ignore[assignment]
        record = await em.store(
            "Fact with provenance.",
            session_id="s1",
            provenance='{"source_type": "user_input"}',
        )
        assert record.provenance is not None
        assert "user_input" in record.provenance
