"""Integration tests for ConsolidationPipeline."""

from __future__ import annotations

import json

import pytest

from nexus.core.models.base import ModelResponse
from nexus.core.types import TokenUsage
from tests.integration.conftest import FakeEmbeddingService, FakeStateStore, MockModelClient


def _llm_returning_facts(facts: list[dict]) -> MockModelClient:
    client = MockModelClient()
    response = ModelResponse(
        content=json.dumps(facts),
        tool_calls=[],
        token_usage=TokenUsage(
            input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="mock"
        ),
        model="mock",
        stop_reason="end_turn",
    )
    client._responses = [response]
    return client


@pytest.mark.integration
class TestConsolidationIntegration:
    async def test_pipeline_extracts_facts_from_episodes(
        self,
        fake_state_store: FakeStateStore,
        fake_embedding_service: FakeEmbeddingService,
    ) -> None:
        from nexus.memory.consolidation import ConsolidationPipeline
        from nexus.memory.episodic import EpisodicMemory
        from nexus.memory.semantic import SemanticMemory

        episodic = EpisodicMemory(fake_state_store, fake_embedding_service, agent_id="con-agent")
        semantic = SemanticMemory(fake_state_store, agent_id="con-agent")
        await episodic.store("Alice likes Python.", session_id="s1")

        llm = _llm_returning_facts(
            [{"subject": "Alice", "predicate": "likes", "object_value": "Python", "confidence": 0.9}]
        )
        pipeline = ConsolidationPipeline(episodic, semantic, llm, agent_id="con-agent")
        result = await pipeline.run()

        assert result.episodes_processed == 1
        assert result.facts_extracted == 1
        facts = await semantic.list_all()
        assert len(facts) == 1
        assert facts[0].subject == "Alice"

    async def test_pipeline_marks_episodes_consolidated(
        self,
        fake_state_store: FakeStateStore,
        fake_embedding_service: FakeEmbeddingService,
    ) -> None:
        from nexus.memory.consolidation import ConsolidationPipeline
        from nexus.memory.episodic import EpisodicMemory
        from nexus.memory.semantic import SemanticMemory

        episodic = EpisodicMemory(fake_state_store, fake_embedding_service, agent_id="mark-agent")
        semantic = SemanticMemory(fake_state_store, agent_id="mark-agent")
        record = await episodic.store("Bob knows Go.", session_id="s1")

        llm = _llm_returning_facts(
            [{"subject": "Bob", "predicate": "knows", "object_value": "Go", "confidence": 0.8}]
        )
        pipeline = ConsolidationPipeline(episodic, semantic, llm, agent_id="mark-agent")
        await pipeline.run()

        updated = await episodic.get(record.id)
        assert updated is not None
        assert updated.metadata.get("consolidated") is True

    async def test_pipeline_skips_already_consolidated(
        self,
        fake_state_store: FakeStateStore,
        fake_embedding_service: FakeEmbeddingService,
    ) -> None:
        from nexus.memory.consolidation import ConsolidationPipeline
        from nexus.memory.episodic import EpisodicMemory
        from nexus.memory.semantic import SemanticMemory

        episodic = EpisodicMemory(fake_state_store, fake_embedding_service, agent_id="skip-agent")
        semantic = SemanticMemory(fake_state_store, agent_id="skip-agent")
        record = await episodic.store("Already processed.", session_id="s1")
        await episodic.update_metadata(record.id, {"consolidated": True})

        llm = _llm_returning_facts([])
        pipeline = ConsolidationPipeline(episodic, semantic, llm, agent_id="skip-agent")
        result = await pipeline.run()

        assert result.episodes_processed == 0
        assert result.facts_extracted == 0

    async def test_pipeline_stores_facts_in_semantic_memory(
        self,
        fake_state_store: FakeStateStore,
        fake_embedding_service: FakeEmbeddingService,
    ) -> None:
        from nexus.memory.consolidation import ConsolidationPipeline
        from nexus.memory.episodic import EpisodicMemory
        from nexus.memory.semantic import SemanticMemory

        episodic = EpisodicMemory(fake_state_store, fake_embedding_service, agent_id="store-agent")
        semantic = SemanticMemory(fake_state_store, agent_id="store-agent")
        await episodic.store("Carol uses TypeScript.", session_id="s1")

        llm = _llm_returning_facts(
            [
                {"subject": "Carol", "predicate": "uses", "object_value": "TypeScript", "confidence": 0.85},
                {"subject": "Carol", "predicate": "language", "object_value": "TypeScript", "confidence": 0.7},
            ]
        )
        pipeline = ConsolidationPipeline(episodic, semantic, llm, agent_id="store-agent")
        result = await pipeline.run()

        assert result.facts_extracted == 2
        facts = await semantic.list_all()
        assert len(facts) == 2

    async def test_pipeline_handles_empty_memory_gracefully(
        self,
        fake_state_store: FakeStateStore,
        fake_embedding_service: FakeEmbeddingService,
    ) -> None:
        from nexus.memory.consolidation import ConsolidationPipeline
        from nexus.memory.episodic import EpisodicMemory
        from nexus.memory.semantic import SemanticMemory

        episodic = EpisodicMemory(fake_state_store, fake_embedding_service, agent_id="empty-agent")
        semantic = SemanticMemory(fake_state_store, agent_id="empty-agent")

        llm = MockModelClient()
        pipeline = ConsolidationPipeline(episodic, semantic, llm, agent_id="empty-agent")
        result = await pipeline.run()

        assert result.episodes_processed == 0
        assert result.facts_extracted == 0

    async def test_batch_size_limits_processing_per_run(
        self,
        fake_state_store: FakeStateStore,
        fake_embedding_service: FakeEmbeddingService,
    ) -> None:
        from nexus.memory.consolidation import ConsolidationPipeline
        from nexus.memory.episodic import EpisodicMemory
        from nexus.memory.semantic import SemanticMemory

        episodic = EpisodicMemory(fake_state_store, fake_embedding_service, agent_id="batch-agent")
        semantic = SemanticMemory(fake_state_store, agent_id="batch-agent")
        for i in range(6):
            await episodic.store(f"Episode {i}.", session_id="s1")

        llm = _llm_returning_facts([])
        pipeline = ConsolidationPipeline(
            episodic, semantic, llm, agent_id="batch-agent", batch_size=3
        )
        result = await pipeline.run()

        assert result.episodes_processed == 3
