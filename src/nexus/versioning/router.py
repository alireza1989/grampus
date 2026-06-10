"""VersionRouter — resolves which AgentDefinition to serve for a given request."""

from __future__ import annotations

import contextlib
import hashlib
from typing import TYPE_CHECKING

from nexus.core.logging import get_logger
from nexus.core.types import AgentDefinition
from nexus.versioning.store import VersionStore

if TYPE_CHECKING:
    from nexus.versioning.ab_testing import ABTestManager

_log = get_logger(__name__)


class VersionRouter:
    """Resolves which AgentDefinition to use for a given (agent_id, user_id) pair.

    When an active A/B test exists and user_id is provided, uses deterministic
    sticky hash routing to assign users to control or treatment.
    Without an active test, returns the production version.
    Returns None when no version is deployed yet.
    """

    def __init__(
        self,
        store: VersionStore,
        ab_manager: ABTestManager | None = None,
    ) -> None:
        self._store = store
        self._ab = ab_manager

    async def resolve(self, agent_id: str, user_id: str | None = None) -> AgentDefinition | None:
        """Return the definition to use, or None if no version is deployed."""
        with contextlib.suppress(Exception):
            return await self._resolve_inner(agent_id, user_id)
        # Fallback: production version
        return await self._resolve_production(agent_id)

    async def _resolve_inner(self, agent_id: str, user_id: str | None) -> AgentDefinition | None:
        if self._ab is not None and user_id is not None:
            test = await self._ab.get_active_test(agent_id)
            if test is not None:
                bucket = (
                    int(
                        hashlib.sha256(f"{test.experiment_id}:{user_id}".encode()).hexdigest(),
                        16,
                    )
                    % 100
                )
                threshold = int(test.traffic_split * 100)
                version_id = (
                    test.treatment_version_id if bucket < threshold else test.control_version_id
                )
                version = await self._store.get_version(version_id)
                if version is not None:
                    _log.debug(
                        "version_ab_resolved",
                        agent_id=agent_id,
                        experiment_id=test.experiment_id,
                        version_id=version_id,
                        user_id=user_id,
                        bucket=bucket,
                    )
                    return version.definition

        return await self._resolve_production(agent_id)

    async def _resolve_production(self, agent_id: str) -> AgentDefinition | None:
        record = await self._store.get_deployment(agent_id)
        if record is None:
            _log.debug("version_no_deployment", agent_id=agent_id)
            return None

        version = await self._store.get_version(record.version_id)
        if version is None:
            _log.warning(
                "version_deployment_dangling",
                agent_id=agent_id,
                version_id=record.version_id,
            )
            return None

        _log.debug(
            "version_resolved",
            agent_id=agent_id,
            version_id=record.version_id,
            user_id=None,
        )
        return version.definition
