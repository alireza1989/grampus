"""Memory stack integration tests using an in-memory duck-typed state store.

No real Dapr sidecar required — uses FakeStateStore from the integration
conftest and a local FakeEmbeddingService so no LLM API is called.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import pytest_asyncio

from grampus.memory.consolidation import ConsolidationPipeline
from grampus.memory.episodic import EpisodicMemory
from grampus.memory.retriever import EpisodicRetriever
from grampus.memory.semantic import SemanticMemory
from grampus.memory.types import SemanticFact

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Minimal fake embedding service (no external provider, no cache needed)
# ---------------------------------------------------------------------------


class _InlineEmbedService:
    """Returns hash-based deterministic vectors; no cache dependency."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, text: str, **_: Any) -> list[float]:
        padded = (text + "\0" * self._dim)[: self._dim]
        return [float(ord(c)) / 256.0 for c in padded]

    async def embed_batch(self, texts: list[str], **_: Any) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Shared fixtures (supplement parent integration/conftest.py fixtures)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def episodic_memory(fake_state_store: Any) -> EpisodicMemory:
    return EpisodicMemory(
        fake_state_store,
        _InlineEmbedService(),
        agent_id="test-infra-agent",
    )


@pytest_asyncio.fixture()
async def semantic_memory(fake_state_store: Any) -> SemanticMemory:
    return SemanticMemory(fake_state_store, agent_id="test-infra-agent")


# ---------------------------------------------------------------------------
# Test 1 — episodic write and retrieve
# ---------------------------------------------------------------------------


class TestEpisodicWriteAndRetrieve:
    async def test_episodic_write_and_retrieve(self, fake_state_store: Any) -> None:
        emb = _InlineEmbedService()
        memory = EpisodicMemory(fake_state_store, emb, agent_id="agent-ep")
        retriever = EpisodicRetriever(memory, emb)

        await memory.store("The sky is blue", session_id="s1")
        await memory.store("Water is wet", session_id="s1")
        await memory.store("Fire is hot", session_id="s1")

        results = await retriever.retrieve("what color is the sky?", top_k=3)

        assert len(results) >= 1, "Retriever should return at least one record"
        all_contents = [r.record.content for r in results]
        assert any("sky" in c.lower() or "blue" in c.lower() for c in all_contents), (
            f"Expected sky/blue in top results but got: {all_contents}"
        )


# ---------------------------------------------------------------------------
# Test 2 — semantic fact deduplication
# ---------------------------------------------------------------------------


class TestSemanticFactDeduplication:
    async def test_semantic_fact_deduplication(self, fake_state_store: Any) -> None:
        memory = SemanticMemory(fake_state_store, agent_id="agent-sem")

        fact = SemanticFact(
            id=str(uuid.uuid4()),
            subject="agent",
            predicate="knows",
            object_value="Python",
            confidence=0.9,
        )
        duplicate = SemanticFact(
            id=str(uuid.uuid4()),
            subject="agent",
            predicate="knows",
            object_value="Python",
            confidence=0.8,
        )

        await memory.store(fact)
        await memory.store(duplicate)

        all_facts = await memory.list_all()
        assert len(all_facts) == 1, (
            f"Deduplication should merge facts with same (subject, predicate); "
            f"got {len(all_facts)}: {all_facts}"
        )


# ---------------------------------------------------------------------------
# Test 3 — MemoryManager remember and recall
# ---------------------------------------------------------------------------


class TestMemoryManagerRememberAndRecall:
    async def test_memory_manager_remember_and_recall(self, memory_manager: Any) -> None:
        await memory_manager.remember(
            "Important fact: Grampus runs on Dapr",
            session_id="s1",
            memory_types=["episodic"],
        )

        result = await memory_manager.recall("Grampus", top_k=5)

        assert len(result.episodic) >= 1, "recall() should return the stored episodic record"
        contents = [r.record.content for r in result.episodic]
        assert any("Grampus" in c for c in contents), (
            f"Expected 'Grampus' in recalled content but got: {contents}"
        )


# ---------------------------------------------------------------------------
# Test 4 — consolidation extracts semantic facts from episodic records
# ---------------------------------------------------------------------------


class TestConsolidationExtractsFacts:
    async def test_consolidation_extracts_facts(self, fake_state_store: Any) -> None:
        from tests.integration.conftest import MockModelClient

        emb = _InlineEmbedService()
        episodic = EpisodicMemory(fake_state_store, emb, agent_id="agent-cons")
        semantic = SemanticMemory(fake_state_store, agent_id="agent-cons")

        # MockModelClient returns JSON fact array that ConsolidationPipeline parses
        fact_json = (
            '[{"subject": "grampus", "predicate": "runs_on", '
            '"object_value": "dapr", "confidence": 0.95}]'
        )
        mock_llm = MockModelClient(default_text=fact_json)

        pipeline = ConsolidationPipeline(
            episodic,
            semantic,
            mock_llm,
            agent_id="agent-cons",
        )

        await episodic.store("Grampus is built on the Dapr runtime.", session_id="s1")
        await episodic.store("Dapr provides state, pub/sub, and workflows.", session_id="s1")

        result = await pipeline.run()

        assert result.episodes_processed >= 1, "Pipeline should process stored episodes"
        assert result.facts_extracted >= 1, (
            f"Pipeline should extract at least one fact; got facts_extracted={result.facts_extracted}"
        )

        extracted_facts = await semantic.list_all()
        assert len(extracted_facts) >= 1, (
            "SemanticMemory should contain extracted facts after consolidation"
        )
