"""Dapr-backed persistence for agent versions and deployments."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from nexus.core.logging import get_logger
from nexus.versioning.types import AgentVersion, DeploymentRecord, VersionStatus

_log = get_logger(__name__)

_MAX_HISTORY = 50


class _VersionIndex(BaseModel):
    """Serializable index mapping agent_id → list of version IDs."""

    ids: list[str] = []


class _DeploymentHistory(BaseModel):
    """Serializable append-only list of deployment records, newest-first."""

    records: list[DeploymentRecord] = []


class VersionStore:
    """Dapr-backed persistence for agent versions and deployments."""

    _ENTITY_VERSIONS = "agent_versions"
    _ENTITY_DEPLOYMENT = "agent_deployment"
    _ENTITY_DEPLOYMENT_HISTORY = "agent_deployment_history"

    def __init__(self, state_store: Any) -> None:
        self._state = state_store

    async def save_version(self, version: AgentVersion) -> None:
        """Persist a version and update the agent's version index."""
        await self._state.save(self._ENTITY_VERSIONS, version.version_id, version)

        index = await self._load_index(version.agent_id)
        if version.version_id not in index.ids:
            index.ids.append(version.version_id)
            await self._state.save(self._ENTITY_VERSIONS, f"index:{version.agent_id}", index)
        _log.debug("version_saved", version_id=version.version_id, agent_id=version.agent_id)

    async def get_version(self, version_id: str) -> AgentVersion | None:
        """Load a single version by ID, or None if not found."""
        result, _ = await self._state.get(self._ENTITY_VERSIONS, version_id, AgentVersion)
        return cast("AgentVersion | None", result)

    async def list_versions(self, agent_id: str) -> list[AgentVersion]:
        """Return all versions for an agent, sorted newest-first."""
        index = await self._load_index(agent_id)
        if not index.ids:
            return []

        versions: list[AgentVersion] = []
        for vid in index.ids:
            try:
                version, _ = await self._state.get(self._ENTITY_VERSIONS, vid, AgentVersion)
                if version is not None:
                    versions.append(version)
            except Exception:
                _log.warning("version_load_failed", version_id=vid, agent_id=agent_id)

        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    async def save_deployment(self, record: DeploymentRecord) -> None:
        """Persist the current deployment and prepend to history."""
        await self._state.save(self._ENTITY_DEPLOYMENT, record.agent_id, record)

        history = await self._load_deployment_history(record.agent_id)
        updated = [record, *history.records]
        if len(updated) > _MAX_HISTORY:
            updated = updated[:_MAX_HISTORY]
        new_history = _DeploymentHistory(records=updated)
        await self._state.save(self._ENTITY_DEPLOYMENT_HISTORY, record.agent_id, new_history)
        _log.debug(
            "deployment_saved",
            agent_id=record.agent_id,
            version_id=record.version_id,
        )

    async def get_deployment(self, agent_id: str) -> DeploymentRecord | None:
        """Load the current active deployment for an agent."""
        result, _ = await self._state.get(self._ENTITY_DEPLOYMENT, agent_id, DeploymentRecord)
        return cast("DeploymentRecord | None", result)

    async def get_deployment_history(self, agent_id: str) -> list[DeploymentRecord]:
        """Return all deployments for an agent, newest-first."""
        history = await self._load_deployment_history(agent_id)
        return history.records

    async def update_version_status(self, version_id: str, status: VersionStatus) -> None:
        """Update the status field of an existing version in-place."""
        version, _ = await self._state.get(self._ENTITY_VERSIONS, version_id, AgentVersion)
        if version is None:
            _log.warning("version_status_update_not_found", version_id=version_id)
            return
        updated = version.model_copy(update={"status": status})
        await self._state.save(self._ENTITY_VERSIONS, version_id, updated)
        _log.debug("version_status_updated", version_id=version_id, status=status)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _load_index(self, agent_id: str) -> _VersionIndex:
        result, _ = await self._state.get(self._ENTITY_VERSIONS, f"index:{agent_id}", _VersionIndex)
        return result if result is not None else _VersionIndex()

    async def _load_deployment_history(self, agent_id: str) -> _DeploymentHistory:
        result, _ = await self._state.get(
            self._ENTITY_DEPLOYMENT_HISTORY, agent_id, _DeploymentHistory
        )
        return result if result is not None else _DeploymentHistory()
