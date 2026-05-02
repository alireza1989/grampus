"""Hybrid episodic memory retriever: recency × similarity × importance."""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.embeddings import cosine_similarity
from nexus.memory.types import EpisodicRecord, RetrievedRecord

_log = get_logger(__name__)

_DEFAULT_ALPHA = 0.4  # recency weight
_DEFAULT_BETA = 0.4  # similarity weight
_DEFAULT_GAMMA = 0.2  # importance weight
_DEFAULT_DECAY_RATE = 0.01  # λ for exp(-λ * age_in_days)
_WEIGHT_TOLERANCE = 1e-6


class EpisodicRetriever:
    """Rank episodic records by a weighted combination of recency, semantic
    similarity, and stored importance score.

    Score formula::

        score = α·recency + β·similarity + γ·importance
        recency = exp(-decay_rate * age_in_days)

    Args:
        episodic_memory: Source of records to search.
        embedding_service: Used to embed the query string.
        alpha: Recency weight (default 0.4).
        beta: Similarity weight (default 0.4).
        gamma: Importance weight (default 0.2).
        decay_rate: Exponential decay constant (default 0.01).

    Raises:
        ValueError: If ``alpha + beta + gamma`` does not sum to 1.0.
    """

    def __init__(
        self,
        episodic_memory: Any,
        embedding_service: Any,
        *,
        alpha: float = _DEFAULT_ALPHA,
        beta: float = _DEFAULT_BETA,
        gamma: float = _DEFAULT_GAMMA,
        decay_rate: float = _DEFAULT_DECAY_RATE,
    ) -> None:
        if not math.isclose(alpha + beta + gamma, 1.0, rel_tol=_WEIGHT_TOLERANCE):
            raise ValueError(f"Retriever weights must sum to 1.0, got {alpha + beta + gamma:.4f}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._decay = decay_rate
        self._memory = episodic_memory
        self._embeddings = embedding_service

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedRecord]:
        """Return the top-*k* records most relevant to *query*.

        Records with no embedding receive ``similarity_score=0.0`` and are
        still eligible to rank via recency and importance.

        Side-effect: ``update_access`` is scheduled as a background task for
        all returned records so it doesn't add latency to the caller.
        """
        records: list[EpisodicRecord] = await self._memory.list_all()
        if not records:
            return []

        query_embedding = await self._embeddings.embed(query)
        now = datetime.now(UTC)

        scored: list[RetrievedRecord] = []
        for rec in records:
            recency = _recency_score(rec.timestamp, now, self._decay)
            similarity = (
                _clip(cosine_similarity(query_embedding, rec.embedding))
                if rec.embedding is not None
                else 0.0
            )
            importance = rec.importance_score
            score = self.alpha * recency + self.beta * similarity + self.gamma * importance
            scored.append(
                RetrievedRecord(
                    record=rec,
                    score=_clip(score),
                    recency_score=_clip(recency),
                    similarity_score=_clip(similarity),
                    importance_score=_clip(importance),
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        results = [r for r in scored[:top_k] if r.score >= min_score]

        # Fire-and-forget access tracking
        for result in results:
            asyncio.create_task(self._memory.update_access(result.record.id))

        _log.debug(
            "episodic_retrieved",
            query_len=len(query),
            total_records=len(records),
            returned=len(results),
        )
        return results


def _recency_score(timestamp: datetime, now: datetime, decay_rate: float) -> float:
    age_days = max((now - timestamp).total_seconds() / 86_400, 0.0)
    return math.exp(-decay_rate * age_days)


def _clip(v: float) -> float:
    """Clamp to [0, 1] to absorb floating-point rounding errors."""
    return max(0.0, min(1.0, v))
