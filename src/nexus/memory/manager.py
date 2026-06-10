"""Unified memory interface — single facade over all four memory types."""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from nexus.core.errors import MemorySecurityError
from nexus.core.logging import get_logger
from nexus.core.types import Message
from nexus.memory.consolidation import ConsolidationPipeline, ConsolidationResult
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.procedural import ProceduralMemory
from nexus.memory.provenance import Provenance, ProvenanceTracker, SourceType
from nexus.memory.retriever import EpisodicRetriever
from nexus.memory.semantic import SemanticMemory
from nexus.memory.semantic_retriever import SemanticRetriever
from nexus.memory.types import EpisodicRecord, Procedure, RetrievedRecord, SemanticFact
from nexus.memory.validator import MemoryValidator
from nexus.memory.working import WorkingMemory

if TYPE_CHECKING:
    from nexus.memory.graph.consolidator import SemanticConsolidator
    from nexus.memory.lifecycle.adaptive_router import AdaptiveRetriever
    from nexus.memory.lifecycle.tier_manager import LifecycleTierManager
    from nexus.plugins.manager import PluginManager

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
        provenance_tracker: Optional tracker that annotates every write with provenance.
            When ``None``, provenance metadata is not attached.
        memory_validator: Optional validator that gates writes through injection
            detection, size limits, and rate limiting.
            When ``None``, validation is skipped.
        graph_consolidator: Optional F3 SemanticConsolidator. When set, the agent's
            knowledge graph is available for graph-based retrieval.
        lifecycle_manager: Optional F3 LifecycleTierManager for access tracking.
        adaptive_router: Optional F3 AdaptiveRetriever. When set, ``recall()``
            routes queries through it instead of calling retrievers directly.
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
        provenance_tracker: ProvenanceTracker | None = None,
        memory_validator: MemoryValidator | None = None,
        graph_consolidator: SemanticConsolidator | None = None,
        lifecycle_manager: LifecycleTierManager | None = None,
        adaptive_router: AdaptiveRetriever | None = None,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        self._working = working_memory
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._procedural = procedural_memory
        self._ep_retriever = episodic_retriever
        self._sem_retriever = semantic_retriever
        self._consolidation = consolidation_pipeline
        self._agent_id = agent_id
        self._tracker = provenance_tracker
        self._validator = memory_validator
        self._graph_consolidator = graph_consolidator
        self._lifecycle_manager = lifecycle_manager
        self._adaptive_router = adaptive_router
        self._plugins = plugin_manager

    async def remember(
        self,
        content: str,
        *,
        session_id: str,
        memory_types: list[str] | None = None,
        source_type: SourceType = SourceType.LLM_GENERATED,
        source_id: str = "unknown",
        **kwargs: Any,
    ) -> None:
        """Persist *content* into the requested memory types.

        The write path is:
        1. Create :class:`Provenance` (if tracker is configured).
        2. Validate via :class:`MemoryValidator` (if configured) — raises
           :class:`MemorySecurityError` on rejection.
        3. Store into each requested memory type with provenance JSON attached.

        Args:
            content: Text content to store.
            session_id: Current session identifier (used for episodic records).
            memory_types: Which stores to write to (``"episodic"``, ``"semantic"``).
                Unknown values are silently ignored. Defaults to ``["episodic"]``.
            source_type: Provenance category of the write origin.
            source_id: Identifier of the specific source.
            **kwargs: Forwarded to the underlying store methods.

        Raises:
            MemorySecurityError: When the validator blocks the write.
        """
        types = memory_types if memory_types is not None else ["episodic"]

        provenance: Provenance | None = None
        if self._tracker is not None:
            provenance = self._tracker.create(content, source_type, source_id=source_id)

        if self._validator is not None:
            result = self._validator.validate(content, source_id=source_id)
            if not result.allowed:
                raise MemorySecurityError(
                    f"Memory write blocked: {'; '.join(result.reasons)}",
                    code="MEMORY_WRITE_BLOCKED",
                    details={"reasons": result.reasons, "source_id": source_id},
                    hint="The content was flagged as a potential memory injection. Review the source and trust level of this write.",
                )

        _final_content = content
        _mem_ctx: Any = None
        if self._plugins:
            from nexus.plugins.types import HookBlockedError, MemoryWriteContext

            _mem_ctx = MemoryWriteContext(
                agent_id=self._agent_id,
                session_id=session_id,
                memory_type=",".join(types),
                source_id=source_id,
            )
            try:
                _final_content = await self._plugins.call_pre_memory_write(_mem_ctx, content)
            except HookBlockedError as exc:
                raise MemorySecurityError(
                    str(exc),
                    code="PLUGIN_BLOCKED",
                    details={"hook": "pre_memory_write", "source_id": source_id},
                ) from exc

        provenance_json: str | None = provenance.model_dump_json() if provenance else None

        for memory_type in types:
            if memory_type == "episodic":
                await self._episodic.store(
                    _final_content, session_id=session_id, provenance=provenance_json, **kwargs
                )
                _log.debug("memory_remembered_episodic", agent=self._agent_id)
            elif memory_type == "semantic":
                fact = SemanticFact(
                    id=str(uuid.uuid4()),
                    subject=self._agent_id,
                    predicate="knows",
                    object_value=_final_content,
                )
                await self._semantic.store(fact)
                _log.debug("memory_remembered_semantic", agent=self._agent_id)
            else:
                _log.debug("memory_remember_unknown_type", memory_type=memory_type)

        if self._plugins and _mem_ctx is not None:
            with contextlib.suppress(Exception):
                await self._plugins.call_post_memory_write(_mem_ctx, None)

    async def recall(
        self,
        query: str,
        *,
        memory_types: list[str] | None = None,
        top_k: int = 5,
    ) -> MemoryRecallResult:
        """Query memory and return a combined result.

        When an AdaptiveRetriever is configured (F3), routes the query through
        it for optimal structure selection. Falls back to the existing path on
        any error to guarantee zero behavioral change when router is absent.

        Args:
            query: Free-text query string.
            memory_types: Which stores to search (``"episodic"``, ``"semantic"``).
                Unknown values are silently ignored.
                Defaults to ``["episodic", "semantic"]``.
            top_k: Maximum records to return from each retriever.

        Returns:
            A :class:`MemoryRecallResult` with results from each queried store.
        """
        # F3: route through AdaptiveRetriever when available
        if self._adaptive_router is not None:
            with contextlib.suppress(Exception):
                return await self._adaptive_router.retrieve(self._agent_id, query, top_k=top_k)

        # fallback: existing path (unchanged)
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

    async def list_records(
        self,
        *,
        agent_id: str | None = None,
        memory_type: str | None = None,
        query: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List memory records for UI inspection.

        Gathers records from each backend (filtered by *memory_type* if given),
        normalises them to a common dict schema, applies filters, sorts by
        ``created_at`` descending, and returns a paginated slice.

        Args:
            agent_id: When set, only return records belonging to this agent.
            memory_type: Restrict to one backend: ``"episodic"``, ``"semantic"``,
                ``"working"``, or ``"procedural"``.  ``None`` means all backends.
            query: Case-insensitive substring filter on the ``content`` field.
            min_trust: Exclude records whose ``trust_score`` is below this value.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).

        Returns:
            A list of dicts, each with keys: ``id``, ``agent_id``,
            ``memory_type``, ``content``, ``trust_score``, ``created_at``,
            ``last_accessed``, ``metadata``, ``provenance``.
        """
        types_to_fetch = (
            ["episodic", "semantic", "working", "procedural"]
            if memory_type is None
            else [memory_type]
        )

        all_records: list[dict[str, Any]] = []
        for mtype in types_to_fetch:
            try:
                records = await self._fetch_records_of_type(mtype)
                all_records.extend(records)
            except Exception:  # noqa: BLE001
                _log.warning("list_records_backend_error", memory_type=mtype)

        if agent_id is not None:
            all_records = [r for r in all_records if r.get("agent_id") == agent_id]

        all_records = [
            r
            for r in all_records
            if (r.get("trust_score") is None or (r.get("trust_score") or 0.0) >= min_trust)
        ]

        if query:
            q = query.lower()
            all_records = [r for r in all_records if q in (r.get("content") or "").lower()]

        all_records.sort(
            key=lambda r: r.get("created_at") or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )

        return all_records[offset : offset + limit]

    async def _fetch_records_of_type(self, memory_type: str) -> list[dict[str, Any]]:
        """Load and normalise records from a single backend."""
        if memory_type == "episodic":
            return [_norm_episodic(r) for r in await self._episodic.list_all()]
        if memory_type == "semantic":
            return [_norm_semantic(f) for f in await self._semantic.list_all()]
        if memory_type == "working":
            return [_norm_message(m) for m in await self._working.get_messages()]
        if memory_type == "procedural":
            return [_norm_procedure(p) for p in await self._procedural.list_all()]
        return []


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _norm_episodic(rec: EpisodicRecord) -> dict[str, Any]:
    prov: dict[str, Any] | None = None
    if rec.provenance:
        try:
            import json

            prov = json.loads(rec.provenance)
        except Exception:  # noqa: BLE001
            prov = {"raw": rec.provenance}
    return {
        "id": rec.id,
        "agent_id": rec.agent_id,
        "memory_type": "episodic",
        "content": rec.content,
        "trust_score": rec.trust_score,
        "created_at": rec.timestamp,
        "last_accessed": rec.last_accessed,
        "metadata": rec.metadata,
        "provenance": prov,
    }


def _norm_semantic(fact: SemanticFact) -> dict[str, Any]:
    content = f"{fact.subject} {fact.predicate} {fact.object_value}"
    return {
        "id": fact.id,
        "agent_id": fact.subject,
        "memory_type": "semantic",
        "content": content,
        "trust_score": fact.confidence,
        "created_at": fact.created_at,
        "last_accessed": None,
        "metadata": {"source_episode_ids": fact.source_episode_ids},
        "provenance": None,
    }


def _norm_message(msg: Message) -> dict[str, Any]:
    content = str(msg.content) if msg.content is not None else ""
    return {
        "id": str(uuid.uuid4()),
        "agent_id": None,
        "memory_type": "working",
        "content": content,
        "trust_score": None,
        "created_at": msg.timestamp if hasattr(msg, "timestamp") else None,
        "last_accessed": None,
        "metadata": {},
        "provenance": None,
    }


def _norm_procedure(proc: Procedure) -> dict[str, Any]:
    steps_summary = " ".join(s.action for s in proc.steps[:3])
    content = f"{proc.name}: {proc.description}. {steps_summary}".strip(". ")
    return {
        "id": proc.id,
        "agent_id": proc.agent_id,
        "memory_type": "procedural",
        "content": content,
        "trust_score": None,
        "created_at": proc.last_used,
        "last_accessed": proc.last_used,
        "metadata": {"success_count": proc.success_count, "failure_count": proc.failure_count},
        "provenance": None,
    }
