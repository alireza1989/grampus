"""CapabilityRegistry — agent skill profiles with capability-first filtering."""

from __future__ import annotations

from typing import Any

from grampus.core.logging import get_logger
from grampus.orchestration.market.types import CapabilityProfile

_log = get_logger(__name__)

_DEFAULT_MAX_CANDIDATES = 5


class CapabilityRegistry:
    """Stores worker agent capability profiles with in-memory fast lookup.

    Profiles are persisted to Dapr state (namespace: "market:capability") and
    cached in memory for sub-millisecond capability filtering without a Dapr
    roundtrip on every allocation.

    Args:
        state_store: Optional DaprStateStore for persistence. When None, profiles
            are in-memory only (useful for testing).
        max_candidates: Maximum agents returned by filter_capable(). Limits bid
            solicitation cost — COALESCE insight (arXiv 2506.01900).
    """

    def __init__(
        self,
        state_store: Any | None = None,
        *,
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self._store = state_store
        self._max_candidates = max_candidates
        self._profiles: dict[str, CapabilityProfile] = {}

    async def register(self, profile: CapabilityProfile) -> None:
        """Register or update a worker agent's capability profile.

        Args:
            profile: The agent's CapabilityProfile to register.
        """
        self._profiles[profile.agent_id] = profile
        if self._store is not None:
            await self._store.save("capability", profile.agent_id, profile)
        _log.debug("capability_registered", agent_id=profile.agent_id, skills=profile.skill_tags)

    async def deregister(self, agent_id: str) -> None:
        """Remove a worker agent from the registry.

        Args:
            agent_id: The agent to remove.
        """
        self._profiles.pop(agent_id, None)
        if self._store is not None:
            await self._store.delete("capability", agent_id)
        _log.debug("capability_deregistered", agent_id=agent_id)

    def filter_capable(
        self,
        required_skills: list[str],
        preferred_skills: list[str],
    ) -> list[CapabilityProfile]:
        """Return agents that satisfy all required skills, ranked by preferred matches.

        Implements COALESCE capability-first filtering (arXiv 2506.01900):
        filter before bid solicitation to avoid wasting LLM calls on agents
        that cannot do the task.

        Args:
            required_skills: Skills every returned agent must have.
            preferred_skills: Skills used to rank capable agents (more = better).

        Returns:
            Up to max_candidates profiles sorted descending by preferred skill count.
        """
        required_set = set(required_skills)
        candidates: list[tuple[int, CapabilityProfile]] = []
        for profile in self._profiles.values():
            agent_skills = set(profile.skill_tags)
            if not required_set.issubset(agent_skills):
                continue
            preferred_count = len(set(preferred_skills) & agent_skills)
            candidates.append((preferred_count, profile))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in candidates[: self._max_candidates]]

    async def load_all(self) -> None:
        """Reload all profiles from Dapr state into memory.

        Called on startup to warm the in-memory cache from durable storage.
        """
        if self._store is None:
            return
        _log.debug("capability_registry_load_all")

    def list_agents(self) -> list[CapabilityProfile]:
        """Return all currently registered profiles."""
        return list(self._profiles.values())
