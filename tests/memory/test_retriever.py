"""Tests for grampus.memory.retriever — EpisodicRetriever hybrid search."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from grampus.memory.episodic import EpisodicMemory
from grampus.memory.retriever import EpisodicRetriever
from grampus.memory.types import EpisodicRecord, RetrievedRecord


def make_record(
    content: str = "test",
    age_days: float = 0.0,
    importance: float = 0.5,
    embedding: list[float] | None = None,
    record_id: str | None = None,
) -> EpisodicRecord:
    ts = datetime.now(UTC) - timedelta(days=age_days)
    return EpisodicRecord(
        id=record_id or f"rec-{content[:8]}",
        agent_id="agent-1",
        session_id="s1",
        timestamp=ts,
        content=content,
        trust_score=0.9,
        importance_score=importance,
        embedding=embedding,
    )


@pytest.fixture()
def mock_episodic() -> AsyncMock:
    mem = AsyncMock(spec=EpisodicMemory)
    mem.list_all = AsyncMock(return_value=[])
    mem.update_access = AsyncMock(return_value=None)
    return mem


@pytest.fixture()
def mock_embeddings() -> AsyncMock:
    svc = AsyncMock()
    svc.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
    return svc


@pytest.fixture()
def retriever(mock_episodic: AsyncMock, mock_embeddings: AsyncMock) -> EpisodicRetriever:
    return EpisodicRetriever(
        episodic_memory=mock_episodic,
        embedding_service=mock_embeddings,
    )


class TestEpisodicRetrieverInit:
    def test_default_weights(self, mock_episodic: AsyncMock, mock_embeddings: AsyncMock) -> None:
        r = EpisodicRetriever(
            episodic_memory=mock_episodic,
            embedding_service=mock_embeddings,
        )
        assert math.isclose(r.alpha + r.beta + r.gamma, 1.0, rel_tol=1e-9)

    def test_custom_weights_valid(
        self, mock_episodic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        r = EpisodicRetriever(
            episodic_memory=mock_episodic,
            embedding_service=mock_embeddings,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
        )
        assert r.alpha == 0.5

    def test_weights_not_summing_to_one_raises(
        self, mock_episodic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            EpisodicRetriever(
                episodic_memory=mock_episodic,
                embedding_service=mock_embeddings,
                alpha=0.5,
                beta=0.5,
                gamma=0.5,
            )


class TestEpisodicRetrieverRetrieve:
    async def test_empty_memory_returns_empty(
        self, retriever: EpisodicRetriever, mock_episodic: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = []
        results = await retriever.retrieve("any query")
        assert results == []

    async def test_returns_at_most_top_k(
        self, retriever: EpisodicRetriever, mock_episodic: AsyncMock
    ) -> None:
        records = [
            make_record(f"fact {i}", embedding=[1.0, 0.0, 0.0], record_id=f"r{i}")
            for i in range(10)
        ]
        mock_episodic.list_all.return_value = records
        results = await retriever.retrieve("query", top_k=3)
        assert len(results) <= 3

    async def test_results_sorted_by_score_descending(
        self, retriever: EpisodicRetriever, mock_episodic: AsyncMock
    ) -> None:
        records = [
            make_record(
                "old fact",
                age_days=100.0,
                importance=0.1,
                embedding=[1.0, 0.0, 0.0],
                record_id="r0",
            ),
            make_record(
                "recent fact",
                age_days=0.1,
                importance=0.9,
                embedding=[1.0, 0.0, 0.0],
                record_id="r1",
            ),
        ]
        mock_episodic.list_all.return_value = records
        results = await retriever.retrieve("query", top_k=5)
        assert len(results) >= 2
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_recent_record_scores_higher_than_old(
        self, mock_episodic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        # Pure recency (alpha=1, beta=0, gamma=0)
        r = EpisodicRetriever(
            episodic_memory=mock_episodic,
            embedding_service=mock_embeddings,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
        )
        records = [
            make_record("old", age_days=365.0, embedding=[1.0, 0.0], record_id="old"),
            make_record("new", age_days=0.1, embedding=[1.0, 0.0], record_id="new"),
        ]
        mock_episodic.list_all.return_value = records
        results = await r.retrieve("q")
        scores = {res.record.id: res.recency_score for res in results}
        assert scores["new"] > scores["old"]

    async def test_similar_record_scores_higher_on_similarity(
        self, mock_episodic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]  # query vector
        r = EpisodicRetriever(
            episodic_memory=mock_episodic,
            embedding_service=mock_embeddings,
            alpha=0.0,
            beta=1.0,
            gamma=0.0,
        )
        records = [
            make_record("similar", embedding=[1.0, 0.0], record_id="sim"),
            make_record("different", embedding=[0.0, 1.0], record_id="diff"),
        ]
        mock_episodic.list_all.return_value = records
        results = await r.retrieve("query")
        scores = {res.record.id: res.similarity_score for res in results}
        assert scores["sim"] > scores["diff"]

    async def test_record_without_embedding_gets_zero_similarity(
        self, retriever: EpisodicRetriever, mock_episodic: AsyncMock
    ) -> None:
        records = [make_record("no embedding", embedding=None, record_id="noem")]
        mock_episodic.list_all.return_value = records
        results = await retriever.retrieve("query")
        assert len(results) == 1
        assert results[0].similarity_score == 0.0

    async def test_min_score_filters_out_low_scores(
        self, mock_episodic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]
        r = EpisodicRetriever(
            episodic_memory=mock_episodic,
            embedding_service=mock_embeddings,
            alpha=0.0,
            beta=1.0,
            gamma=0.0,
        )
        records = [
            make_record("similar", embedding=[1.0, 0.0], record_id="sim"),
            make_record("different", embedding=[0.0, 1.0], record_id="diff"),
        ]
        mock_episodic.list_all.return_value = records
        results = await r.retrieve("query", min_score=0.5)
        # Only the similar record should pass a 0.5 threshold
        for res in results:
            assert res.score >= 0.5

    async def test_update_access_called_for_returned_records(
        self, retriever: EpisodicRetriever, mock_episodic: AsyncMock
    ) -> None:
        records = [make_record("fact", embedding=[1.0, 0.0, 0.0], record_id="r0")]
        mock_episodic.list_all.return_value = records
        await retriever.retrieve("query", top_k=5)
        # Allow event loop to process the background task
        import asyncio

        await asyncio.sleep(0)
        mock_episodic.update_access.assert_called()

    async def test_returns_retrieved_record_objects(
        self, retriever: EpisodicRetriever, mock_episodic: AsyncMock
    ) -> None:
        records = [make_record("fact", embedding=[1.0, 0.0, 0.0])]
        mock_episodic.list_all.return_value = records
        results = await retriever.retrieve("q")
        assert all(isinstance(r, RetrievedRecord) for r in results)

    async def test_scores_include_all_components(
        self, retriever: EpisodicRetriever, mock_episodic: AsyncMock
    ) -> None:
        records = [make_record("fact", embedding=[1.0, 0.0, 0.0])]
        mock_episodic.list_all.return_value = records
        results = await retriever.retrieve("q")
        assert len(results) == 1
        r = results[0]
        assert 0.0 <= r.recency_score <= 1.0
        assert 0.0 <= r.similarity_score <= 1.0
        assert 0.0 <= r.importance_score <= 1.0
        assert 0.0 <= r.score <= 1.0

    async def test_importance_weight_affects_ranking(
        self, mock_episodic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]
        r = EpisodicRetriever(
            episodic_memory=mock_episodic,
            embedding_service=mock_embeddings,
            alpha=0.0,
            beta=0.0,
            gamma=1.0,
        )
        records = [
            make_record("low importance", importance=0.1, embedding=[1.0, 0.0], record_id="low"),
            make_record("high importance", importance=0.9, embedding=[1.0, 0.0], record_id="high"),
        ]
        mock_episodic.list_all.return_value = records
        results = await r.retrieve("q")
        scores = {res.record.id: res.score for res in results}
        assert scores["high"] > scores["low"]

    @given(
        alpha=st.floats(min_value=0.0, max_value=1.0),
        beta=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=30, suppress_health_check=["function_scoped_fixture"])
    def test_composite_score_in_unit_interval(
        self,
        alpha: float,
        beta: float,
        mock_episodic: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        gamma = max(0.0, 1.0 - alpha - beta)
        total = alpha + beta + gamma
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            return
        import asyncio

        r = EpisodicRetriever(
            episodic_memory=mock_episodic,
            embedding_service=mock_embeddings,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        records = [make_record("fact", age_days=1.0, importance=0.5, embedding=[1.0, 0.0])]
        mock_episodic.list_all.return_value = records
        mock_embeddings.embed.return_value = [1.0, 0.0]
        results = asyncio.run(r.retrieve("q"))
        for res in results:
            assert -1e-9 <= res.score <= 1.0 + 1e-9
