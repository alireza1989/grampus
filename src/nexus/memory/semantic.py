"""Semantic memory: structured fact store with deduplication."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.types import SemanticFact

_log = get_logger(__name__)

_ENTITY = "semantic"
_INDEX_KEY = "_index"


class SemanticMemory:
    """CRUD store for semantic facts backed by a DaprStateStore.

    Key layout (within the agent's namespace):
    - ``semantic:{fact_id}`` — individual fact
    - ``semantic:_index`` — JSON list of fact IDs for this agent

    Deduplication: storing a fact whose (subject, predicate) matches an
    existing fact merges them rather than creating a duplicate. The higher-
    confidence fact's ``object_value`` wins; ``source_episode_ids`` are
    always unioned.

    Args:
        state_store: A DaprStateStore (or duck-typed equivalent).
        agent_id: Scopes all keys to this agent.
    """

    def __init__(self, state_store: Any, *, agent_id: str) -> None:
        self._store = state_store
        self._agent_id = agent_id
        self._index: list[str] = []

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def store(self, fact: SemanticFact) -> SemanticFact:
        """Persist *fact*, merging with any existing (subject, predicate) match.

        Returns the stored fact (may be the merged version of an existing one).
        """
        existing_facts = await self.list_all()
        for existing in existing_facts:
            if existing.subject == fact.subject and existing.predicate == fact.predicate:
                merged = _merge_facts(existing, fact)
                await self._save_fact(merged)
                _log.debug("semantic_fact_merged", subject=fact.subject, predicate=fact.predicate)
                return merged

        await self._save_fact(fact)
        self._index.append(fact.id)
        await self._save_index()
        _log.debug("semantic_fact_stored", fact_id=fact.id, agent=self._agent_id)
        return fact

    async def get(self, fact_id: str) -> SemanticFact | None:
        """Load a single fact by ID. Returns None if not found."""
        result, _ = await self._store.get(_ENTITY, fact_id, SemanticFact)
        return result  # type: ignore[no-any-return]

    async def delete(self, fact_id: str) -> None:
        """Remove a fact and its entry from the index."""
        await self._store.delete(_ENTITY, fact_id)
        if fact_id in self._index:
            self._index.remove(fact_id)
            await self._save_index()
        _log.debug("semantic_fact_deleted", fact_id=fact_id, agent=self._agent_id)

    async def list_all(self) -> list[SemanticFact]:
        """Return all facts for this agent."""
        if not self._index:
            return []
        facts: list[SemanticFact] = []
        for fid in list(self._index):
            fact = await self.get(fid)
            if fact is not None:
                facts.append(fact)
        return facts

    # ------------------------------------------------------------------
    # Filtered queries
    # ------------------------------------------------------------------

    async def find_by_subject(self, subject: str) -> list[SemanticFact]:
        """Return all facts whose subject matches exactly."""
        return [f for f in await self.list_all() if f.subject == subject]

    async def find_by_predicate(self, subject: str, predicate: str) -> list[SemanticFact]:
        """Return all facts matching (subject, predicate) exactly."""
        return [
            f for f in await self.list_all() if f.subject == subject and f.predicate == predicate
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _save_fact(self, fact: SemanticFact) -> None:
        await self._store.save(_ENTITY, fact.id, fact)

    async def _save_index(self) -> None:
        data = json.dumps(self._index).encode()
        await self._store.save(_ENTITY, _INDEX_KEY, data)


def _merge_facts(existing: SemanticFact, incoming: SemanticFact) -> SemanticFact:
    """Return a merged fact, keeping the higher-confidence object_value."""
    merged_ids = list({*existing.source_episode_ids, *incoming.source_episode_ids})
    now = datetime.now(UTC)

    if incoming.confidence > existing.confidence:
        return existing.model_copy(
            update={
                "object_value": incoming.object_value,
                "confidence": incoming.confidence,
                "source_episode_ids": merged_ids,
                "updated_at": now,
                "embedding": incoming.embedding or existing.embedding,
            }
        )
    return existing.model_copy(
        update={
            "source_episode_ids": merged_ids,
            "updated_at": now,
        }
    )
