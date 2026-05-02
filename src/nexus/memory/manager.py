"""Unified memory interface — single facade over all four memory types."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from nexus.core.logging import get_logger
from nexus.core.types import Message
from nexus.memory.consolidation import ConsolidationPipeline, ConsolidationResult
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.procedural import ProceduralMemory
from nexus.memory.retriever import EpisodicRetriever
from nexus.memory.semantic import SemanticMemory
from nexus.memory.semantic_retriever import SemanticRetriever
from nexus.memory.types import RetrievedRecord, SemanticFact
from nexus.memory.working import WorkingMemory

_log = get_logger(__name__)


class MemoryRecallResult(BaseModel):
    """Combined results from a cross-memory recall query."""

    episodic: list[RetrievedRecord] = Field(default_factory=list)
    semantic: list[SemanticFact] = Field(default_factory=list)
    query: str


class MemoryManager:
    """Single facade over working, episodic, semantic, and procedural memory.

    All dependencies are injected — no instantiation occurs internally.
    This is what the orchestration layer talks to.

    Args:
        working_memory: In-session message buffer with auto-summarization.
        episodic_memory: Cross-session episodic record store.
        semantic_memory: Structured fact store with deduplication.
        procedural_memory: Learned workflow store.
        episodic_retriever: Hybrid retriever for episodic records.
        semantic_retriever: Similarity-based retriever for semantic facts.
        consolidation_pipeline: LLM-based fact extractor from episodic records.
        agent_id: Scopes operations to this agent.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        procedural_memory: ProceduralMemory,
        episodic_retriever: EpisodicRetriever,
        semantic_retriever: SemanticRetriever,
        consolidation_pipeline: ConsolidationPipeline,
        *,
        agent_id: str,
    ) -> None:
        self._working = working_memory
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._procedural = procedural_memory
        self._ep_retriever = episodic_retriever
        self._sem_retriever = semantic_retriever
        self._consolidation = consolidation_pipeline
        self._agent_id = agent_id

    async def remember(
        self,
        content: str,
        *,
        session_id: str,
        memory_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Persist *content* into the requested memory types.

        Args:
            content: Text content to store.
            session_id: Current session identifier (used for episodic records).
            memory_types: Which stores to write to (``"episodic"``, ``"semantic"``).
                Unknown values are silently ignored. Defaults to ``["episodic"]``.
            **kwargs: Forwarded to the underlying store methods.
        """
        types = memory_types if memory_types is not None else ["episodic"]

        for memory_type in types:
            if memory_type == "episodic":
                await self._episodic.store(content, session_id=session_id, **kwargs)
                _log.debug("memory_remembered_episodic", agent=self._agent_id)
            elif memory_type == "semantic":
                fact = SemanticFact(
                    id=str(uuid.uuid4()),
                    subject=self._agent_id,
                    predicate="knows",
                    object_value=content,
                )
                await self._semantic.store(fact)
                _log.debug("memory_remembered_semantic", agent=self._agent_id)
            else:
                _log.debug("memory_remember_unknown_type", memory_type=memory_type)

    async def recall(
        self,
        query: str,
        *,
        memory_types: list[str] | None = None,
        top_k: int = 5,
    ) -> MemoryRecallResult:
        """Query memory and return a combined result.

        Args:
            query: Free-text query string.
            memory_types: Which stores to search (``"episodic"``, ``"semantic"``).
                Unknown values are silently ignored.
                Defaults to ``["episodic", "semantic"]``.
            top_k: Maximum records to return from each retriever.

        Returns:
            A :class:`MemoryRecallResult` with results from each queried store.
        """
        types = memory_types if memory_types is not None else ["episodic", "semantic"]

        episodic_results: list[RetrievedRecord] = []
        semantic_results: list[SemanticFact] = []

        for memory_type in types:
            if memory_type == "episodic":
                episodic_results = await self._ep_retriever.retrieve(query, top_k=top_k)
                _log.debug(
                    "memory_recalled_episodic",
                    agent=self._agent_id,
                    returned=len(episodic_results),
                )
            elif memory_type == "semantic":
                scored = await self._sem_retriever.retrieve_similar(query, top_k=top_k)
                semantic_results = [sf.fact for sf in scored]
                _log.debug(
                    "memory_recalled_semantic",
                    agent=self._agent_id,
                    returned=len(semantic_results),
                )
            else:
                _log.debug("memory_recall_unknown_type", memory_type=memory_type)

        return MemoryRecallResult(
            episodic=episodic_results,
            semantic=semantic_results,
            query=query,
        )

    async def forget(self, record_id: str, *, memory_type: str) -> None:
        """Delete a record from the specified memory store.

        Args:
            record_id: ID of the record to remove.
            memory_type: Which store to delete from (``"episodic"`` or ``"semantic"``).

        Raises:
            ValueError: If *memory_type* is not ``"episodic"`` or ``"semantic"``.
        """
        if memory_type == "episodic":
            await self._episodic.delete(record_id)
            _log.debug("memory_forgotten_episodic", agent=self._agent_id, record_id=record_id)
        elif memory_type == "semantic":
            await self._semantic.delete(record_id)
            _log.debug("memory_forgotten_semantic", agent=self._agent_id, record_id=record_id)
        else:
            raise ValueError(
                f"Unknown memory_type {memory_type!r}. Must be 'episodic' or 'semantic'."
            )

    async def consolidate(self) -> ConsolidationResult:
        """Run one consolidation pass: extract semantic facts from episodic records.

        Returns:
            A :class:`ConsolidationResult` with extraction statistics.
        """
        result = await self._consolidation.run()
        _log.debug(
            "memory_consolidated",
            agent=self._agent_id,
            facts_extracted=result.facts_extracted,
            episodes_processed=result.episodes_processed,
        )
        return result

    async def add_message(self, message: Message) -> None:
        """Add *message* to the working memory window.

        Delegates directly to :meth:`WorkingMemory.add`.
        """
        await self._working.add(message)

    async def get_messages(self) -> list[Message]:
        """Return the current working memory window.

        Delegates directly to :meth:`WorkingMemory.get_messages`.
        """
        return await self._working.get_messages()
