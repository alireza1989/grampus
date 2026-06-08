"""AdaptiveRetriever — routes queries to the right memory structure (FluxMem, arXiv 2602.14038)."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from nexus.core.logging import get_logger
from nexus.memory.lifecycle.types import MemoryType, QueryClassification

if TYPE_CHECKING:
    from nexus.memory.graph.retriever import GraphRetriever
    from nexus.memory.lifecycle.tier_manager import LifecycleTierManager
    from nexus.memory.manager import MemoryRecallResult

_log = get_logger(__name__)

_GRAPH_QUERY_MIN_LEN = 80
_SEQUENTIAL_KEYWORDS = frozenset(["last time", "previously", "earlier", "before", "recent"])
_GRAPH_KEYWORDS = frozenset(["how", "why", "relationship", "explain", "cause", "effect"])


class AdaptiveRetriever:
    """Routes memory queries to the right retrieval structure (FluxMem, arXiv 2602.14038).

    Three routing targets:
    - GRAPH: GraphRetriever — for knowledge-dense, multi-concept queries
    - FLAT: EpisodicRetriever + SemanticRetriever — for simple factual lookups
    - SEQUENTIAL: most recent N EpisodicRecords — for recency/continuity queries

    Args:
        episodic_retriever: EpisodicRetriever (existing Phase 3 component).
        semantic_retriever: SemanticRetriever (existing Phase 4 component).
        graph_retriever: GraphRetriever (new F3 component). Optional.
        episodic_memory: EpisodicMemory for SEQUENTIAL retrieval.
        tier_manager: LifecycleTierManager for recording accesses. Optional.
    """

    def __init__(
        self,
        episodic_retriever: Any,
        semantic_retriever: Any,
        *,
        graph_retriever: GraphRetriever | None = None,
        episodic_memory: Any | None = None,
        tier_manager: LifecycleTierManager | None = None,
    ) -> None:
        self._ep_retriever = episodic_retriever
        self._sem_retriever = semantic_retriever
        self._graph_retriever = graph_retriever
        self._episodic_memory = episodic_memory
        self._tier_manager = tier_manager

    async def retrieve(
        self,
        agent_id: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> MemoryRecallResult:
        """Classify query and route to appropriate retrieval method.

        Returns MemoryRecallResult. Never raises — falls back to empty result.
        """
        from nexus.memory.manager import MemoryRecallResult  # noqa: PLC0415

        classification = self.classify(query)
        _log.debug("adaptive_route", classification=classification.value, query_len=len(query))

        with contextlib.suppress(Exception):
            if classification == QueryClassification.GRAPH:
                result = await self._retrieve_graph(agent_id, query, top_k)
                if result.episodic or result.semantic:
                    return result
                return await self._retrieve_flat(query, top_k)
            elif classification == QueryClassification.SEQUENTIAL:
                return await self._retrieve_sequential(agent_id, top_k)
            else:
                return await self._retrieve_flat(query, top_k)

        return MemoryRecallResult(episodic=[], semantic=[], query=query)

    def classify(self, query: str) -> QueryClassification:
        """Classify query type. Pure function — no I/O."""
        query_lower = query.lower()
        if any(kw in query_lower for kw in _SEQUENTIAL_KEYWORDS):
            return QueryClassification.SEQUENTIAL
        if (
            len(query) > _GRAPH_QUERY_MIN_LEN or any(kw in query_lower for kw in _GRAPH_KEYWORDS)
        ) and self._graph_retriever is not None:
            return QueryClassification.GRAPH
        return QueryClassification.FLAT

    async def _retrieve_graph(self, agent_id: str, query: str, top_k: int) -> MemoryRecallResult:
        """Graph traversal retrieval. Falls back to FLAT on empty graph."""
        assert self._graph_retriever is not None  # noqa: S101
        result = await self._graph_retriever.query(agent_id, query, top_k=top_k)
        recall = self._graph_result_to_recall(result, query)

        if self._tier_manager:
            for node in result.nodes:
                with contextlib.suppress(Exception):
                    await self._tier_manager.record_access(node.node_id, MemoryType.SEMANTIC)

        return recall

    async def _retrieve_flat(self, query: str, top_k: int) -> MemoryRecallResult:
        """Standard episodic + semantic retrieval (existing Phase 3/4 path)."""
        from nexus.memory.manager import MemoryRecallResult  # noqa: PLC0415
        from nexus.memory.types import RetrievedRecord  # noqa: PLC0415

        episodic: list[RetrievedRecord] = []
        with contextlib.suppress(Exception):
            episodic = await self._ep_retriever.retrieve(query, top_k=top_k)

        from nexus.memory.types import SemanticFact  # noqa: PLC0415

        semantic: list[SemanticFact] = []
        with contextlib.suppress(Exception):
            scored = await self._sem_retriever.retrieve_similar(query, top_k=top_k)
            semantic = [sf.fact for sf in scored]

        if self._tier_manager:
            for rec in episodic:
                with contextlib.suppress(Exception):
                    await self._tier_manager.record_access(rec.record.id, MemoryType.EPISODIC)

        return MemoryRecallResult(episodic=episodic, semantic=semantic, query=query)

    async def _retrieve_sequential(self, agent_id: str, top_k: int) -> MemoryRecallResult:
        """Load the most recent top_k episodic records by timestamp."""
        from nexus.memory.manager import MemoryRecallResult  # noqa: PLC0415
        from nexus.memory.types import EpisodicRecord, RetrievedRecord  # noqa: PLC0415

        if self._episodic_memory is None:
            return MemoryRecallResult(episodic=[], semantic=[], query="")

        all_records: list[EpisodicRecord] = await self._episodic_memory.list_all()
        all_records.sort(key=lambda r: r.timestamp, reverse=True)
        recent = all_records[:top_k]

        episodic = [
            RetrievedRecord(
                record=rec,
                score=1.0,
                recency_score=1.0,
                similarity_score=0.0,
                importance_score=rec.importance_score,
            )
            for rec in recent
        ]

        if self._tier_manager:
            for rec in recent:
                with contextlib.suppress(Exception):
                    await self._tier_manager.record_access(rec.id, MemoryType.EPISODIC)

        return MemoryRecallResult(episodic=episodic, semantic=[], query="")

    def _graph_result_to_recall(self, result: Any, query: str) -> MemoryRecallResult:
        """Convert GraphQueryResult to MemoryRecallResult for uniform interface."""
        from nexus.memory.manager import MemoryRecallResult  # noqa: PLC0415
        from nexus.memory.types import SemanticFact  # noqa: PLC0415

        schematic: list[SemanticFact] = []
        regular: list[SemanticFact] = []

        for node in result.nodes:
            fact = SemanticFact(
                id=node.node_id,
                subject=node.label,
                predicate="is",
                object_value=node.description,
                confidence=node.confidence,
                source_episode_ids=node.source_episode_ids,
            )
            if node.metadata.get("category") == "schematic":
                schematic.append(fact)
            else:
                regular.append(fact)

        return MemoryRecallResult(
            episodic=[],
            semantic=schematic + regular,
            query=query,
        )
