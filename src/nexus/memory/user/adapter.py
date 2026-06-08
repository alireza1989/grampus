"""UserMemoryAdapter — session-level integration between UserMemoryStore and AgentRunner."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from nexus.core.logging import get_logger
from nexus.memory.user.types import (
    FactExtractionResult,
    UserFact,
    UserMemoryContext,
    UserProfile,
)

if TYPE_CHECKING:
    from nexus.memory.user.extractor import FactExtractor
    from nexus.memory.user.store import UserMemoryStore
    from nexus.memory.user.synthesizer import ProfileSynthesizer

_log = get_logger(__name__)

_MAX_RELEVANT_FACTS_IN_CONTEXT = 5


class UserMemoryAdapter:
    """Session-level integration between UserMemoryStore and AgentRunner.

    Two responsibilities:
    1. get_context(): called before each LLM call to inject user-aware context
    2. observe_session_end(): called after run() to trigger fact extraction +
       conditional profile synthesis

    Args:
        store: UserMemoryStore.
        extractor: FactExtractor for post-session extraction.
        synthesizer: ProfileSynthesizer for profile updates.
        embedding_service: For embedding the task query in get_context().
    """

    def __init__(
        self,
        store: UserMemoryStore,
        extractor: FactExtractor,
        synthesizer: ProfileSynthesizer,
        embedding_service: Any,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._synthesizer = synthesizer
        self._embeddings = embedding_service

    async def get_context(
        self,
        user_id: str,
        task: str,
        model_client: Any,
    ) -> UserMemoryContext:
        """Build context for injection into AgentRunner.

        Steps:
        1. Load UserProfile (or None if first interaction)
        2. Embed task query
        3. find_similar_facts(user_id, task_embedding, top_k=5, valid_only=True)
        4. Increment access_count for each returned fact
        5. format_context() → UserMemoryContext

        Never raises — returns empty UserMemoryContext on error.
        """
        with contextlib.suppress(Exception):
            return await self._do_get_context(user_id, task)
        return UserMemoryContext(user_id=user_id)

    async def _do_get_context(self, user_id: str, task: str) -> UserMemoryContext:
        profile = await self._store.get_profile(user_id)

        task_embedding: list[float] | None = None
        with contextlib.suppress(Exception):
            task_embedding = await self._embeddings.embed(task)

        relevant: list[UserFact] = []
        if task_embedding is not None:
            relevant = await self._store.find_similar_facts(
                user_id, task_embedding, top_k=_MAX_RELEVANT_FACTS_IN_CONTEXT, valid_only=True
            )
        elif profile is not None:
            relevant = (await self._store.get_valid_facts(user_id))[:_MAX_RELEVANT_FACTS_IN_CONTEXT]

        for fact in relevant:
            with contextlib.suppress(Exception):
                await self._store.increment_access(user_id, fact.id)

        formatted = self.format_context(user_id, profile, relevant)
        return UserMemoryContext(
            user_id=user_id,
            profile=profile,
            relevant_facts=relevant,
            formatted_context=formatted,
        )

    async def observe_session_end(
        self,
        user_id: str,
        session_id: str,
        model_client: Any,
    ) -> FactExtractionResult:
        """Post-session hook.

        Steps:
        1. extractor.extract_from_session(user_id, session_id, model_client)
        2. synthesizer.synthesize(user_id, model_client) — respects synthesis_interval
        3. Return FactExtractionResult

        Never raises — returns FactExtractionResult(facts_extracted=0) on error.
        """
        empty = FactExtractionResult(
            user_id=user_id,
            session_id=session_id,
            facts_extracted=0,
            facts_updated=0,
            facts_expired=0,
        )
        with contextlib.suppress(Exception):
            result = await self._extractor.extract_from_session(user_id, session_id, model_client)
            with contextlib.suppress(Exception):
                await self._synthesizer.synthesize(user_id, model_client)
            return result
        return empty

    def format_context(
        self,
        user_id: str,
        profile: UserProfile | None,
        relevant_facts: list[UserFact],
    ) -> str:
        """Build the injected string from profile + facts.

        Returns empty string if both profile is None and relevant_facts is empty.
        Profile is always at the top; relevant facts follow as bullets.
        Facts grouped by category. Max _MAX_RELEVANT_FACTS_IN_CONTEXT facts total.
        """
        if profile is None and not relevant_facts:
            return ""

        parts: list[str] = []

        if profile is not None:
            domains = (
                ", ".join(profile.expertise_domains) if profile.expertise_domains else "general"
            )
            goals_line = self._format_goals_line(profile.active_goals)
            constraints_line = self._format_constraints_line(profile.active_constraints)

            header = (
                f"User profile for {user_id}:\n"
                f"- Expertise: level {profile.expertise_level}/5 in {domains}\n"
                f"- Communication style: {profile.communication_style}, "
                f"preferred depth: {profile.preferred_depth}\n"
                f"{goals_line}"
                f"{constraints_line}"
            ).rstrip()
            parts.append(header)

        capped = relevant_facts[:_MAX_RELEVANT_FACTS_IN_CONTEXT]
        if capped:
            by_cat: dict[str, list[UserFact]] = {}
            for f in capped:
                by_cat.setdefault(f.category.value, []).append(f)

            fact_lines: list[str] = ["Relevant facts about this user:"]
            for cat, cat_facts in sorted(by_cat.items()):
                for f in cat_facts:
                    fact_lines.append(f"  [{cat}] {f.content}")
            parts.append("\n".join(fact_lines))

        return "\n\n".join(parts)

    def _format_goals_line(self, goals: list[str]) -> str:
        if not goals:
            return ""
        return f"- Active goals: {', '.join(goals[:3])}\n"

    def _format_constraints_line(self, constraints: list[str]) -> str:
        if not constraints:
            return ""
        return f"- Constraints: {', '.join(constraints[:3])}\n"
