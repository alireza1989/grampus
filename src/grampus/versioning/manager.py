"""VersionManager — facade for the agent version lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime

from grampus.core.errors import VersioningError
from grampus.core.logging import get_logger
from grampus.core.types import AgentDefinition
from grampus.versioning.store import VersionStore
from grampus.versioning.types import (
    AgentVersion,
    DeploymentRecord,
    VersionDiff,
    VersionStatus,
    compute_version_id,
    diff_versions,
)

_log = get_logger(__name__)


class VersionManager:
    """Facade for agent version lifecycle: create, deploy, rollback, diff."""

    def __init__(self, store: VersionStore, *, agent_id: str) -> None:
        self._store = store
        self._agent_id = agent_id

    async def create_version(
        self,
        definition: AgentDefinition,
        *,
        version_tag: str,
        author: str = "unknown",
        description: str = "",
        parent_version_id: str | None = None,
        tags: list[str] | None = None,
    ) -> AgentVersion:
        """Create and persist a new version snapshot.

        If an identical definition already exists (same version_id hash),
        the existing version is returned without creating a duplicate.
        """
        version_id = compute_version_id(definition)
        existing = await self._store.get_version(version_id)
        if existing is not None:
            _log.debug("version_already_exists", version_id=version_id)
            return existing

        version = AgentVersion(
            version_id=version_id,
            agent_id=self._agent_id,
            version_tag=version_tag,
            definition=definition,
            author=author,
            description=description,
            parent_version_id=parent_version_id,
            tags=tags or [],
        )
        await self._store.save_version(version)
        _log.debug("version_created", version_id=version_id, tag=version_tag)
        return version

    async def deploy(self, version_id: str, *, deployed_by: str = "system") -> DeploymentRecord:
        """Promote a version to active deployment.

        Raises:
            VersioningError: code="VERSION_NOT_FOUND" when version_id is unknown.
        """
        version = await self._store.get_version(version_id)
        if version is None:
            raise VersioningError(
                f"Version '{version_id}' not found",
                code="VERSION_NOT_FOUND",
                details={"version_id": version_id},
            )

        current = await self._store.get_deployment(self._agent_id)
        previous_version_id = current.version_id if current is not None else None

        record = DeploymentRecord(
            agent_id=self._agent_id,
            version_id=version_id,
            deployed_at=datetime.now(UTC),
            deployed_by=deployed_by,
            previous_version_id=previous_version_id,
        )
        await self._store.save_deployment(record)
        await self._store.update_version_status(version_id, VersionStatus.PRODUCTION)

        if previous_version_id and previous_version_id != version_id:
            prev = await self._store.get_version(previous_version_id)
            if prev is not None and prev.status == VersionStatus.PRODUCTION:
                await self._store.update_version_status(previous_version_id, VersionStatus.RETIRED)

        _log.debug(
            "version_deployed",
            version_id=version_id,
            previous=previous_version_id,
        )
        return record

    async def rollback(self) -> DeploymentRecord:
        """Revert to the previous deployment.

        Raises:
            VersioningError: code="NO_PRIOR_VERSION" when there is no prior deployment.
        """
        current = await self._store.get_deployment(self._agent_id)
        if current is None:
            raise VersioningError(
                f"No active deployment found for agent '{self._agent_id}'",
                code="NO_PRIOR_VERSION",
                details={"agent_id": self._agent_id},
            )
        if current.previous_version_id is None:
            raise VersioningError(
                f"No prior deployment to roll back to for agent '{self._agent_id}'",
                code="NO_PRIOR_VERSION",
                details={"agent_id": self._agent_id},
                hint="This is the first deployment; there is nothing earlier to restore.",
            )
        return await self.deploy(current.previous_version_id)

    async def get_active_version(self) -> AgentVersion | None:
        """Return the currently deployed version, or None if none is deployed."""
        record = await self._store.get_deployment(self._agent_id)
        if record is None:
            return None
        return await self._store.get_version(record.version_id)

    async def retire(self, version_id: str) -> None:
        """Mark a version as retired."""
        await self._store.update_version_status(version_id, VersionStatus.RETIRED)

    async def list_versions(self) -> list[AgentVersion]:
        """Return all versions for this agent, sorted newest-first."""
        return await self._store.list_versions(self._agent_id)

    async def get_version(self, version_id: str) -> AgentVersion | None:
        """Load a specific version by ID."""
        return await self._store.get_version(version_id)

    async def diff(self, version_id_a: str, version_id_b: str) -> VersionDiff:
        """Compute a structured diff between two versions.

        Raises:
            VersioningError: code="VERSION_NOT_FOUND" when either ID is unknown.
        """
        a = await self._store.get_version(version_id_a)
        if a is None:
            raise VersioningError(
                f"Version '{version_id_a}' not found",
                code="VERSION_NOT_FOUND",
                details={"version_id": version_id_a},
            )
        b = await self._store.get_version(version_id_b)
        if b is None:
            raise VersioningError(
                f"Version '{version_id_b}' not found",
                code="VERSION_NOT_FOUND",
                details={"version_id": version_id_b},
            )
        return diff_versions(a, b)
