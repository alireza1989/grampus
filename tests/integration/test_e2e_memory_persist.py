"""E2E scenario: memory persists across session restarts."""

from __future__ import annotations

import json

import pytest

from tests.integration.conftest import (
    FakeEmbeddingService,
    FakeStateStore,
    MockModelClient,
)


@pytest.mark.integration
class TestMemoryPersistenceE2E:
    async def test_episodic_memory_survives_session_restart(
        self, fake_state_store: FakeStateStore
    ) -> None:
        """Session A stores a record; Session B retrieves it by ID."""
        from nexus.memory.episodic import EpisodicMemory

        emb = FakeEmbeddingService()
        agent_id = "persist-agent"

        em_session_a = EpisodicMemory(fake_state_store, emb, agent_id=agent_id)
        record = await em_session_a.store(
            "User prefers dark mode.", session_id="session-a"
        )

        em_session_b = EpisodicMemory(fake_state_store, emb, agent_id=agent_id)
        fetched = await em_session_b.get(record.id)
        assert fetched is not None
        assert fetched.content == "User prefers dark mode."

    async def test_semantic_facts_survive_consolidation_cycle(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.memory.consolidation import ConsolidationPipeline
        from nexus.memory.episodic import EpisodicMemory
        from nexus.memory.semantic import SemanticMemory

        emb = FakeEmbeddingService()
        agent_id = "consol-persist-agent"

        episodic = EpisodicMemory(fake_state_store, emb, agent_id=agent_id)
        semantic = SemanticMemory(fake_state_store, agent_id=agent_id)

        await episodic.store("Alice likes Python.", session_id="s1")

        mock_llm = MockModelClient()
        mock_llm._responses = [
            __import__("nexus.core.models.base", fromlist=["ModelResponse"]).ModelResponse(
                content=json.dumps(
                    [{"subject": "Alice", "predicate": "likes", "object_value": "Python", "confidence": 0.9}]
                ),
                tool_calls=[],
                token_usage=__import__("nexus.core.types", fromlist=["TokenUsage"]).TokenUsage(
                    input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="mock"
                ),
                model="mock",
                stop_reason="end_turn",
            )
        ]

        pipeline = ConsolidationPipeline(episodic, semantic, mock_llm, agent_id=agent_id)
        await pipeline.run()

        episodes = await episodic.list_all()
        assert all(e.metadata.get("consolidated") for e in episodes)

        semantic2 = SemanticMemory(fake_state_store, agent_id=agent_id)
        fact = await semantic2.get((await semantic.list_all())[0].id)
        assert fact is not None
        assert fact.subject == "Alice"

    async def test_provenance_preserved_across_sessions(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.memory.episodic import EpisodicMemory
        from nexus.memory.provenance import Provenance, ProvenanceTracker, SourceType

        emb = FakeEmbeddingService()
        agent_id = "prov-persist-agent"
        tracker = ProvenanceTracker()
        content = "Important fact with provenance."
        prov = tracker.create(content, SourceType.USER_INPUT, source_id="user-42")

        em1 = EpisodicMemory(fake_state_store, emb, agent_id=agent_id)
        record = await em1.store(content, session_id="s1", provenance=prov.model_dump_json())

        em2 = EpisodicMemory(fake_state_store, emb, agent_id=agent_id)
        fetched = await em2.get(record.id)
        assert fetched is not None
        assert fetched.provenance is not None

        loaded = Provenance(**json.loads(fetched.provenance))
        assert tracker.verify(content, loaded)

    async def test_multiple_records_all_retrievable_by_id(
        self, fake_state_store: FakeStateStore
    ) -> None:
        from nexus.memory.episodic import EpisodicMemory

        emb = FakeEmbeddingService()
        agent_id = "multi-persist-agent"

        em1 = EpisodicMemory(fake_state_store, emb, agent_id=agent_id)
        ids = []
        for i in range(5):
            r = await em1.store(f"Record {i}", session_id="s1")
            ids.append(r.id)

        em2 = EpisodicMemory(fake_state_store, emb, agent_id=agent_id)
        for i, rid in enumerate(ids):
            fetched = await em2.get(rid)
            assert fetched is not None
            assert fetched.content == f"Record {i}"
