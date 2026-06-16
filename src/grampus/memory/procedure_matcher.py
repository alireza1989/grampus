"""Procedure matcher: find relevant stored procedures via cosine similarity."""

from __future__ import annotations

from typing import Any

from grampus.core.logging import get_logger
from grampus.memory.embeddings import cosine_similarity
from grampus.memory.types import Procedure

_log = get_logger(__name__)


class ProcedureMatcher:
    """Retrieve stored procedures most relevant to a task description.

    Procedures without an embedding are silently skipped. Results are
    sorted by cosine similarity (descending).

    Args:
        procedural_memory: Source of Procedure records.
        embedding_service: Used to embed the query for similarity search.
    """

    def __init__(self, procedural_memory: Any, embedding_service: Any) -> None:
        self._memory = procedural_memory
        self._embeddings = embedding_service

    async def find_matches(
        self,
        task_description: str,
        *,
        top_k: int = 5,
    ) -> list[tuple[Procedure, float]]:
        """Return the top-*k* procedures most relevant to *task_description*.

        Returns an empty list if no procedures have embeddings. Each entry
        is a ``(Procedure, score)`` tuple where ``score`` is the cosine
        similarity clamped to ``[0.0, 1.0]``.
        """
        procedures: list[Procedure] = await self._memory.list_all()
        candidates = [p for p in procedures if p.embedding is not None]
        if not candidates:
            return []

        query_embedding = await self._embeddings.embed(task_description)

        scored: list[tuple[Procedure, float]] = []
        for procedure in candidates:
            score = cosine_similarity(query_embedding, procedure.embedding)  # type: ignore[arg-type]
            scored.append((procedure, max(0.0, min(1.0, score))))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        result = scored[:top_k]
        _log.debug(
            "procedure_matched",
            query_len=len(task_description),
            total_with_embedding=len(candidates),
            returned=len(result),
        )
        return result
