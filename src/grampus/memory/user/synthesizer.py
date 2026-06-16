"""ProfileSynthesizer — reflective agent: UserFacts → UserProfile (Bi-Mem + HMO)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from grampus.core.logging import get_logger
from grampus.memory.user.types import ProfileSynthesisResult, UserFact, UserProfile

if TYPE_CHECKING:
    from grampus.memory.user.store import UserMemoryStore

_log = get_logger(__name__)

_SYNTHESIZE_SYSTEM_PROMPT = (
    "You are a user profile synthesizer. "
    "Given a list of facts about a user, produce a structured profile. "
    "Be conservative — only assert what the facts clearly support. "
    "Reply with only valid JSON matching this schema: "
    '{"expertise_level": 1-5, "expertise_domains": ["..."], '
    '"communication_style": "concise|balanced|detailed", '
    '"preferred_depth": "overview|balanced|deep-dive", '
    '"active_goals": ["..."], "past_key_decisions": ["..."], '
    '"active_constraints": ["..."]}. '
    "expertise_level: 1=complete novice, 3=competent practitioner, 5=leading expert."
)

_SYNTHESIZE_USER_TEMPLATE = "User facts ({count} total):\n{facts_text}\n\nSynthesize the profile:"

_DEFAULT_SYNTHESIS_INTERVAL = 10


class ProfileSynthesizer:
    """Reflective agent: synthesizes UserProfile from accumulated UserFacts.

    Triggered when new_facts_since_last_synthesis >= synthesis_interval.
    Replaces the existing UserProfile entirely — version is incremented.

    Args:
        store: UserMemoryStore for loading facts + persisting profile.
        synthesis_interval: Number of new facts between re-syntheses. Default 10.
        max_facts_in_prompt: Maximum facts to pass to LLM (oldest pruned). Default 30.
    """

    def __init__(
        self,
        store: UserMemoryStore,
        *,
        synthesis_interval: int = _DEFAULT_SYNTHESIS_INTERVAL,
        max_facts_in_prompt: int = 30,
    ) -> None:
        self._store = store
        self._interval = synthesis_interval
        self._max_facts = max_facts_in_prompt

    async def synthesize(
        self,
        user_id: str,
        model_client: Any,
        *,
        force: bool = False,
    ) -> ProfileSynthesisResult:
        """Synthesize or update UserProfile from current valid facts.

        Steps:
        1. Load existing profile (for version + synthesis_fact_count)
        2. Count valid facts; if count - synthesis_fact_count < synthesis_interval
           AND force=False → return ProfileSynthesisResult(triggered=False)
        3. Load all valid facts; sort by access_count desc; take top max_facts_in_prompt
        4. LLM call with SYNTHESIZE_USER_TEMPLATE
        5. Parse response; build UserProfile with version = previous + 1
        6. store_profile() → return ProfileSynthesisResult(triggered=True, ...)

        Never raises — returns triggered=False on any error.
        """
        try:
            return await self._do_synthesize(user_id, model_client, force=force)
        except Exception:
            _log.warning("profile_synthesis_failed", user_id=user_id)
            return ProfileSynthesisResult(user_id=user_id, triggered=False)

    async def _do_synthesize(
        self,
        user_id: str,
        model_client: Any,
        *,
        force: bool,
    ) -> ProfileSynthesisResult:
        existing_profile = await self._store.get_profile(user_id)
        prev_version = existing_profile.version if existing_profile else 0
        prev_fact_count = existing_profile.synthesis_fact_count if existing_profile else 0

        valid_facts = await self._store.get_valid_facts(user_id)
        current_count = len(valid_facts)

        if not force and (current_count - prev_fact_count) < self._interval:
            return ProfileSynthesisResult(user_id=user_id, triggered=False)

        if not valid_facts:
            return ProfileSynthesisResult(user_id=user_id, triggered=False)

        top_facts = sorted(valid_facts, key=lambda f: f.access_count, reverse=True)[
            : self._max_facts
        ]

        facts_text = self._format_facts_for_prompt(top_facts)
        from grampus.core.types import Message, Role  # noqa: PLC0415

        response = await model_client.complete(
            messages=[
                Message(role=Role.SYSTEM, content=_SYNTHESIZE_SYSTEM_PROMPT),
                Message(
                    role=Role.USER,
                    content=_SYNTHESIZE_USER_TEMPLATE.format(
                        count=len(top_facts), facts_text=facts_text
                    ),
                ),
            ],
            model="claude-haiku-4-5-20251001",
            temperature=0.2,
        )

        new_version = prev_version + 1
        profile = self._parse_synthesis_response(
            response.content or "", user_id, new_version, current_count
        )
        await self._store.store_profile(profile)

        _log.debug(
            "profile_synthesized",
            user_id=user_id,
            version=new_version,
            facts_used=len(top_facts),
        )
        return ProfileSynthesisResult(
            user_id=user_id,
            triggered=True,
            previous_version=prev_version,
            new_version=new_version,
            facts_used=len(top_facts),
        )

    def _format_facts_for_prompt(self, facts: list[UserFact]) -> str:
        """Format facts as a numbered list grouped by category."""
        by_category: dict[str, list[UserFact]] = {}
        for f in facts:
            by_category.setdefault(f.category.value, []).append(f)

        lines: list[str] = []
        i = 1
        for cat, cat_facts in sorted(by_category.items()):
            lines.append(f"[{cat.upper()}]")
            for f in cat_facts:
                lines.append(f"  {i}. {f.content}")
                i += 1
        return "\n".join(lines)

    def _parse_synthesis_response(
        self, response_text: str, user_id: str, version: int, fact_count: int
    ) -> UserProfile:
        """Parse JSON response into UserProfile. Returns safe defaults on failure."""
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("no JSON object found")
            data = json.loads(response_text[start:end])

            level = int(data.get("expertise_level", 3))
            level = max(1, min(5, level))

            return UserProfile(
                user_id=user_id,
                expertise_level=level,
                expertise_domains=list(data.get("expertise_domains", [])),
                communication_style=str(data.get("communication_style", "balanced")),
                preferred_depth=str(data.get("preferred_depth", "balanced")),
                active_goals=list(data.get("active_goals", [])),
                past_key_decisions=list(data.get("past_key_decisions", [])),
                active_constraints=list(data.get("active_constraints", [])),
                last_synthesized=datetime.now(UTC),
                synthesis_fact_count=fact_count,
                version=version,
            )
        except Exception:
            _log.warning("synthesis_parse_failed", user_id=user_id)
            return UserProfile(
                user_id=user_id,
                last_synthesized=datetime.now(UTC),
                synthesis_fact_count=fact_count,
                version=version,
            )
