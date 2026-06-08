"""GraphBuilder — session-scoped event-progression-graph (GAM, arXiv 2604.12285)."""

from __future__ import annotations

import contextlib
import math
import uuid
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.graph.types import EventGraph, EventNode, SemanticShiftEvent

_log = get_logger(__name__)

_SEMANTIC_SHIFT_THRESHOLD = 0.30
_MIN_EVENTS_FOR_SHIFT_CHECK = 3
_CONTENT_SUMMARY_MAX_CHARS = 200


class GraphBuilder:
    """Maintains a session-scoped event-progression-graph (GAM, arXiv 2604.12285).

    Each significant event is appended to an in-memory EventGraph. When the
    embedding of the most recent events drifts from the last consolidated state
    by more than the shift threshold, a SemanticShiftEvent is emitted and the
    caller should trigger consolidation.

    The EventGraph is NOT persisted to Dapr — it is session-local and discarded
    after consolidation.

    Args:
        embedding_service: For embedding event content summaries.
        shift_threshold: Cosine distance threshold for semantic shift detection.
    """

    def __init__(
        self,
        embedding_service: Any,
        shift_threshold: float = _SEMANTIC_SHIFT_THRESHOLD,
    ) -> None:
        self._embeddings = embedding_service
        self._shift_threshold = shift_threshold
        self._graphs: dict[str, EventGraph] = {}

    def init_session(self, session_id: str, agent_id: str) -> EventGraph:
        """Create and register a new EventGraph for this session."""
        graph = EventGraph(session_id=session_id, agent_id=agent_id)
        self._graphs[session_id] = graph
        return graph

    async def append_event(
        self,
        session_id: str,
        event_type: str,
        content: str,
        agent_id: str,
    ) -> SemanticShiftEvent | None:
        """Add an event to the session graph and check for semantic shift.

        Returns SemanticShiftEvent if shift detected, None otherwise.
        Always returns None (silently) if session_id not found or embedding fails.
        """
        graph = self._graphs.get(session_id)
        if graph is None:
            return None

        summary = content[:_CONTENT_SUMMARY_MAX_CHARS]
        event_id = str(uuid.uuid4())

        embedding: list[float] | None = None
        with contextlib.suppress(Exception):
            embedding = await self._embeddings.embed(summary)

        node = EventNode(
            event_id=event_id,
            event_type=event_type,
            content_summary=summary,
            embedding=embedding,
            session_id=session_id,
            agent_id=agent_id,
        )
        graph.events.append(node)

        if embedding is None or len(graph.events) < _MIN_EVENTS_FOR_SHIFT_CHECK:
            return None

        if graph.last_embedding is None:
            graph.last_embedding = embedding
            return None

        distance = self._cosine_distance(embedding, graph.last_embedding)
        if distance > self._shift_threshold:
            graph.last_embedding = embedding
            _log.debug(
                "semantic_shift_detected",
                session_id=session_id,
                distance=distance,
                event_id=event_id,
            )
            return SemanticShiftEvent(
                session_id=session_id,
                agent_id=agent_id,
                shift_distance=distance,
                trigger_event_id=event_id,
            )

        return None

    def get_graph(self, session_id: str) -> EventGraph | None:
        """Return the current EventGraph for this session."""
        return self._graphs.get(session_id)

    def end_session(self, session_id: str) -> EventGraph | None:
        """Remove and return the EventGraph. Called after consolidation."""
        return self._graphs.pop(session_id, None)

    def _cosine_distance(self, a: list[float], b: list[float]) -> float:
        """1 - cosine_similarity. Returns 1.0 on zero vectors."""
        return 1.0 - _cosine_similarity(a, b)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on zero vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
