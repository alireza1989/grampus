"""SectionLockManager — TODO-claim protocol using Dapr distributed lock.

Implements the CodeCRDT TODO-claim protocol (arXiv 2510.18893) to guarantee
at-most-one-winner section claiming across concurrent agents.
"""

from __future__ import annotations

import contextlib
from typing import Any

from grampus.core.errors import LockAcquisitionError
from grampus.core.logging import get_logger
from grampus.orchestration.artifact.store import ArtifactStore
from grampus.orchestration.artifact.types import SectionOwnershipState

_log = get_logger(__name__)

_DEFAULT_LOCK_TIMEOUT_SECS = 300  # 5 minutes


class SectionLockManager:
    """TODO-claim protocol (CodeCRDT, arXiv 2510.18893).

    Guarantees at-most-one-winner: only one agent can hold CLAIMED on a
    section at a time. Uses the Dapr distributed lock from Phase 2 as the
    atomic primitive.

    Lock key: "artifact-lock:{artifact_id}:{section_id}"

    Args:
        dapr_lock_factory: Callable that returns a DaprLock for a given resource_id and owner.
        store: ArtifactStore for updating section ownership state after acquiring the lock.
    """

    def __init__(
        self,
        dapr_lock_factory: Any,
        store: ArtifactStore,
    ) -> None:
        self._lock_factory = dapr_lock_factory
        self._store = store

    async def claim(
        self,
        artifact_id: str,
        section_id: str,
        agent_id: str,
        timeout_secs: int = _DEFAULT_LOCK_TIMEOUT_SECS,
    ) -> bool:
        """Attempt to claim a section for exclusive write.

        Idempotent if the same agent already holds the claim.

        Args:
            artifact_id: Target artifact.
            section_id: Target section.
            agent_id: Agent requesting the claim.
            timeout_secs: Lock expiry in seconds.

        Returns:
            True when the claim succeeds; False when already claimed by another agent.
        """
        section = await self._store._load_section(artifact_id, section_id)
        if section is not None and section.ownership_state == SectionOwnershipState.CLAIMED:
            return section.owner_agent_id == agent_id

        resource_id = _lock_key(artifact_id, section_id)
        lock = self._lock_factory(
            resource_id=resource_id, lock_owner=agent_id, expiry_seconds=timeout_secs
        )

        try:
            await lock.__aenter__()
        except LockAcquisitionError:
            _log.debug(
                "section_claim_failed",
                artifact_id=artifact_id,
                section_id=section_id,
                agent_id=agent_id,
            )
            return False

        await self._store.update_section_state(
            artifact_id, section_id, SectionOwnershipState.CLAIMED, agent_id=agent_id
        )
        _log.debug(
            "section_claimed",
            artifact_id=artifact_id,
            section_id=section_id,
            agent_id=agent_id,
        )
        return True

    async def release(
        self,
        artifact_id: str,
        section_id: str,
        agent_id: str,
        mark_complete: bool = True,
    ) -> None:
        """Release a section after writing.

        Args:
            artifact_id: Target artifact.
            section_id: Target section.
            agent_id: Agent releasing the claim.
            mark_complete: True → state becomes MERGED; False → UNOWNED.
        """
        section = await self._store._load_section(artifact_id, section_id)
        if section is None:
            return
        if section.owner_agent_id != agent_id:
            return

        new_state = SectionOwnershipState.MERGED if mark_complete else SectionOwnershipState.UNOWNED
        await self._store.update_section_state(artifact_id, section_id, new_state, agent_id=None)

        resource_id = _lock_key(artifact_id, section_id)
        lock = self._lock_factory(resource_id=resource_id, lock_owner=agent_id, expiry_seconds=60)
        with contextlib.suppress(Exception):
            await lock.__aexit__(None, None, None)

        _log.debug(
            "section_released",
            artifact_id=artifact_id,
            section_id=section_id,
            agent_id=agent_id,
            mark_complete=mark_complete,
        )

    async def is_claimed_by(self, artifact_id: str, section_id: str, agent_id: str) -> bool:
        """Return True if this agent currently holds the claim.

        Args:
            artifact_id: Target artifact.
            section_id: Target section.
            agent_id: Agent to check.
        """
        section = await self._store._load_section(artifact_id, section_id)
        if section is None:
            return False
        return (
            section.ownership_state == SectionOwnershipState.CLAIMED
            and section.owner_agent_id == agent_id
        )

    async def get_owner(self, artifact_id: str, section_id: str) -> str | None:
        """Return current owner agent_id or None if unclaimed.

        Args:
            artifact_id: Target artifact.
            section_id: Target section.
        """
        section = await self._store._load_section(artifact_id, section_id)
        if section is None:
            return None
        if section.ownership_state == SectionOwnershipState.CLAIMED:
            return section.owner_agent_id
        return None


def _lock_key(artifact_id: str, section_id: str) -> str:
    return f"artifact-lock:{artifact_id}:{section_id}"
