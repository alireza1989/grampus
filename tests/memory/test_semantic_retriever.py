"""Tests for nexus.memory.semantic_retriever — SemanticRetriever."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from nexus.memory.semantic_retriever import SemanticRetriever
from nexus.memory.types import SemanticFact


def make_fact(
    subject: str = "user",
    predicate: str = "prefers",
    object_value: str = "dark mode",
    embedding: list[float] | None = None,
    fact_id: str | None = None,
) -> SemanticFact:
    return SemanticFact(
        id=fact_id or str(uuid.uuid4()),
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        embedding=embedding,
    )


@pytest.fixture()
def mock_semantic() -> AsyncMock:
    mem = AsyncMock()
    mem.list_all = AsyncMock(return_value=[])
    mem.find_by_subject = AsyncMock(return_value=[])
    mem.find_by_predicate = AsyncMock(return_value=[])
    return mem


@pytest.fixture()
def mock_embeddings() -> AsyncMock:
    svc = AsyncMock()
    svc.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
    return svc


@pytest.fixture()
def retriever(mock_semantic: AsyncMock, mock_embeddings: AsyncMock) -> SemanticRetriever:
    return SemanticRetriever(
        semantic_memory=mock_semantic,
        embedding_service=mock_embeddings,
    )


class TestSemanticRetrieverBySubject:
    async def test_returns_facts_matching_subject(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        facts = [make_fact(subject="alice"), make_fact(subject="alice")]
        mock_semantic.find_by_subject.return_value = facts
        results = await retriever.retrieve_by_subject("alice")
        assert len(results) == 2
        mock_semantic.find_by_subject.assert_called_once_with("alice")

    async def test_returns_empty_for_unknown_subject(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        mock_semantic.find_by_subject.return_value = []
        results = await retriever.retrieve_by_subject("nobody")
        assert results == []

    async def test_delegates_to_semantic_memory(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        await retriever.retrieve_by_subject("test-subject")
        mock_semantic.find_by_subject.assert_called_once_with("test-subject")


class TestSemanticRetrieverByPredicate:
    async def test_returns_facts_matching_predicate(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        fact = make_fact(subject="user", predicate="likes")
        mock_semantic.find_by_predicate.return_value = [fact]
        results = await retriever.retrieve_by_predicate("user", "likes")
        assert len(results) == 1
        mock_semantic.find_by_predicate.assert_called_once_with("user", "likes")

    async def test_returns_empty_for_no_match(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        mock_semantic.find_by_predicate.return_value = []
        results = await retriever.retrieve_by_predicate("user", "hates")
        assert results == []


class TestSemanticRetrieverSimilar:
    async def test_empty_memory_returns_empty(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        mock_semantic.list_all.return_value = []
        results = await retriever.retrieve_similar("any query")
        assert results == []

    async def test_facts_without_embedding_are_skipped(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        mock_semantic.list_all.return_value = [make_fact(embedding=None)]
        results = await retriever.retrieve_similar("query")
        assert results == []

    async def test_returns_at_most_top_k(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        facts = [make_fact(embedding=[1.0, 0.0, 0.0]) for _ in range(10)]
        mock_semantic.list_all.return_value = facts
        results = await retriever.retrieve_similar("query", top_k=3)
        assert len(results) <= 3

    async def test_sorted_by_similarity_descending(
        self, mock_semantic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]
        retriever = SemanticRetriever(
            semantic_memory=mock_semantic, embedding_service=mock_embeddings
        )
        facts = [
            make_fact(object_value="similar", embedding=[1.0, 0.0], fact_id="sim"),
            make_fact(object_value="different", embedding=[0.0, 1.0], fact_id="diff"),
        ]
        mock_semantic.list_all.return_value = facts
        results = await retriever.retrieve_similar("query")
        assert results[0].id == "sim"
        assert results[1].id == "diff"

    async def test_similar_fact_scores_higher_than_dissimilar(
        self, mock_semantic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]
        retriever = SemanticRetriever(
            semantic_memory=mock_semantic, embedding_service=mock_embeddings
        )
        similar = make_fact(embedding=[1.0, 0.0], fact_id="sim")
        dissimilar = make_fact(embedding=[0.0, 1.0], fact_id="diff")
        mock_semantic.list_all.return_value = [similar, dissimilar]
        results = await retriever.retrieve_similar("query")
        scores = {r.id: r.score for r in results}
        assert scores["sim"] > scores["diff"]

    async def test_returns_scored_facts(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock
    ) -> None:
        mock_semantic.list_all.return_value = [make_fact(embedding=[1.0, 0.0, 0.0])]
        results = await retriever.retrieve_similar("query")
        assert len(results) == 1
        assert hasattr(results[0], "score")
        assert 0.0 <= results[0].score <= 1.0

    async def test_embeds_query_once(
        self, retriever: SemanticRetriever, mock_semantic: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        mock_semantic.list_all.return_value = [
            make_fact(embedding=[1.0, 0.0, 0.0]),
            make_fact(embedding=[0.5, 0.5, 0.0]),
        ]
        await retriever.retrieve_similar("my query")
        mock_embeddings.embed.assert_called_once_with("my query")
