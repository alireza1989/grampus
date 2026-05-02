"""Semantic fact retrieval: subject/predicate lookup and similarity search."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from nexus.core.logging import get_logger
from nexus.memory.embeddings import cosine_similarity
from nexus.memory.types import SemanticFact

_log = get_logger(__name__)


class ScoredFact(BaseModel):
    """A semantic fact annotated with a similarity score."""

    id: str
    subject: str
    predicate: str
    object_value: str
    score: float
    fact: SemanticFact


class SemanticRetriever:
    """Retrieve semantic facts by subject, predicate, or semantic similarity.

    Args:
        semantic_memory: Source of SemanticFact records.
        embedding_service: Used to embed free-text queries for similarity search.
    """

    def __init__(self, semantic_memory: Any, embedding_service: Any) -> None:
        self._memory = semantic_memory
        self._embeddings = embedding_service

    async def retrieve_by_subject(self, subject: str) -> list[SemanticFact]:
        """Return all facts whose subject matches exactly."""
        return await self._memory.find_by_subject(subject)  # type: ignore[no-any-return]

    async def retrieve_by_predicate(self, subject: str, predicate: str) -> list[SemanticFact]:
        """Return all facts matching (subject, predicate) exactly."""
        return await self._memory.find_by_predicate(subject, predicate)  # type: ignore[no-any-return]

    async def retrieve_similar(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[ScoredFact]:
        """Return the top-*k* facts most semantically similar to *query*.

        Facts without an embedding are excluded from similarity ranking.
        """
        facts: list[SemanticFact] = await self._memory.list_all()
        facts_with_embeddings = [f for f in facts if f.embedding is not None]
        if not facts_with_embeddings:
            return []

        query_embedding = await self._embeddings.embed(query)

        scored: list[ScoredFact] = []
        for fact in facts_with_embeddings:
            score = cosine_similarity(query_embedding, fact.embedding)  # type: ignore[arg-type]
            scored.append(
                ScoredFact(
                    id=fact.id,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object_value=fact.object_value,
                    score=max(0.0, min(1.0, score)),
                    fact=fact,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        result = scored[:top_k]
        _log.debug(
            "semantic_retrieved",
            query_len=len(query),
            total=len(facts_with_embeddings),
            returned=len(result),
        )
        return result
