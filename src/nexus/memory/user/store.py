"""Dapr-backed persistence for UserFact records and UserProfile."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.user.types import (
    UserFact,
    UserFactCategory,
    UserProfile,
    _FactIndex,
)

_log = get_logger(__name__)

_FACT_ENTITY = "user_fact"
_PROFILE_ENTITY = "user_profile"
_INDEX_SUFFIX = "_index"


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Returns 0.0 for zero-magnitude vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class UserMemoryStore:
    """Dapr-backed store for UserFacts and UserProfiles.

    Key layout:
    - ``user_fact:{user_id}:{fact_id}`` — individual fact
    - ``user_fact:{user_id}:_index`` — JSON list of fact IDs for this user
    - ``user_profile:{user_id}`` — single UserProfile per user

    Args:
        state_store: DaprStateStore (or duck-typed equivalent).
        embedding_service: For embedding facts during store_fact().
    """

    def __init__(self, state_store: Any, embedding_service: Any) -> None:
        self._store = state_store
        self._embeddings = embedding_service

    # ------------------------------------------------------------------
    # UserFact CRUD
    # ------------------------------------------------------------------

    async def store_fact(self, fact: UserFact) -> UserFact:
        """Persist a UserFact. Generates embedding if not already set.

        Registers in per-user index. Returns the stored fact.
        """
        if fact.embedding is None:
            try:
                fact = fact.model_copy(
                    update={"embedding": await self._embeddings.embed(fact.content)}
                )
            except Exception:
                _log.warning("user_fact_embedding_failed", fact_id=fact.id)

        await self._store.save(_FACT_ENTITY, self._fact_key(fact.user_id, fact.id), fact)

        index = await self._load_index(fact.user_id)
        if fact.id not in index.ids:
            index.ids.append(fact.id)
            await self._store.save(_FACT_ENTITY, self._index_key(fact.user_id), index)

        _log.debug("user_fact_stored", user_id=fact.user_id, fact_id=fact.id)
        return fact

    async def get_fact(self, user_id: str, fact_id: str) -> UserFact | None:
        """Load a single fact by ID. Returns None if not found."""
        result, _ = await self._store.get(_FACT_ENTITY, self._fact_key(user_id, fact_id), UserFact)
        return result  # type: ignore[no-any-return]

    async def update_fact(self, fact: UserFact) -> UserFact:
        """Overwrite an existing fact (used for confidence updates + expiry)."""
        await self._store.save(_FACT_ENTITY, self._fact_key(fact.user_id, fact.id), fact)
        _log.debug("user_fact_updated", user_id=fact.user_id, fact_id=fact.id)
        return fact

    async def expire_fact(self, user_id: str, fact_id: str) -> None:
        """Set valid_until = now on a specific fact."""
        fact = await self.get_fact(user_id, fact_id)
        if fact is None:
            return
        expired = fact.model_copy(update={"valid_until": datetime.now(UTC)})
        await self.update_fact(expired)
        _log.debug("user_fact_expired", user_id=user_id, fact_id=fact_id)

    async def get_valid_facts(
        self,
        user_id: str,
        *,
        category: UserFactCategory | None = None,
    ) -> list[UserFact]:
        """Return all non-expired facts for this user, optionally filtered by category."""
        all_facts = await self.list_all_facts(user_id)
        results = [f for f in all_facts if f.is_valid]
        if category is not None:
            results = [f for f in results if f.category == category]
        return results

    async def find_similar_facts(
        self,
        user_id: str,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        valid_only: bool = True,
    ) -> list[UserFact]:
        """Cosine-similarity search over embedded facts.

        Falls back to returning all valid facts if none have embeddings.
        Pure Python cosine — no numpy required.
        """
        facts = (
            await self.get_valid_facts(user_id)
            if valid_only
            else await self.list_all_facts(user_id)
        )
        embedded = [f for f in facts if f.embedding is not None]

        if not embedded:
            return facts[:top_k]

        scored = sorted(
            ((f, _cosine(query_embedding, f.embedding)) for f in embedded),  # type: ignore[arg-type]
            key=lambda x: x[1],
            reverse=True,
        )
        return [f for f, _ in scored[:top_k]]

    async def increment_access(self, user_id: str, fact_id: str) -> None:
        """Increment access_count and update last_accessed on a fact."""
        fact = await self.get_fact(user_id, fact_id)
        if fact is None:
            return
        updated = fact.model_copy(
            update={
                "access_count": fact.access_count + 1,
                "last_accessed": datetime.now(UTC),
            }
        )
        await self.update_fact(updated)

    async def get_profile(self, user_id: str) -> UserProfile | None:
        """Load the UserProfile for this user, or None if not found."""
        result, _ = await self._store.get(_PROFILE_ENTITY, user_id, UserProfile)
        return result  # type: ignore[no-any-return]

    async def store_profile(self, profile: UserProfile) -> UserProfile:
        """Overwrite the UserProfile for this user."""
        await self._store.save(_PROFILE_ENTITY, profile.user_id, profile)
        _log.debug("user_profile_stored", user_id=profile.user_id, version=profile.version)
        return profile

    async def list_all_facts(self, user_id: str) -> list[UserFact]:
        """Return all facts (including expired) for this user."""
        index = await self._load_index(user_id)
        facts: list[UserFact] = []
        for fid in index.ids:
            fact = await self.get_fact(user_id, fid)
            if fact is not None:
                facts.append(fact)
        return facts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fact_key(self, user_id: str, fact_id: str) -> str:
        return f"{user_id}:{fact_id}"

    def _index_key(self, user_id: str) -> str:
        return f"{user_id}:{_INDEX_SUFFIX}"

    async def _load_index(self, user_id: str) -> _FactIndex:
        result, _ = await self._store.get(_FACT_ENTITY, self._index_key(user_id), _FactIndex)
        if result is None:
            return _FactIndex()
        return result  # type: ignore[no-any-return]


def _new_fact_id() -> str:
    return str(uuid.uuid4())
