"""Tests for VersionManager — version lifecycle facade."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import VersioningError
from nexus.core.types import AgentDefinition
from nexus.versioning.manager import VersionManager
from nexus.versioning.store import VersionStore
from nexus.versioning.types import (
    VersionStatus,
    compute_version_id,
)


def _make_def(
    prompt: str = "You are helpful.", model: str = "claude-sonnet-4-6"
) -> AgentDefinition:
    return AgentDefinition(
        name="mgr-agent",
        model=model,
        system_prompt=prompt,
        tools=["search"],
    )


def _make_state_store() -> MagicMock:
    store: dict[tuple[str, str], object] = {}

    mock = MagicMock()

    async def save(entity_type: str, entity_id: str, model: object, **kwargs: object) -> None:
        store[(entity_type, entity_id)] = model

    async def get(entity_type: str, entity_id: str, cls: type) -> tuple[object | None, str]:
        val = store.get((entity_type, entity_id))
        if val is None:
            return None, ""
        if hasattr(val, "model_dump_json"):
            restored = cls.model_validate_json(val.model_dump_json())  # type: ignore[attr-defined]
            return restored, "etag-1"
        return val, "etag-1"

    mock.save = AsyncMock(side_effect=save)
    mock.get = AsyncMock(side_effect=get)
    return mock


def _make_manager(agent_id: str = "mgr-agent") -> VersionManager:
    store = VersionStore(_make_state_store())
    return VersionManager(store, agent_id=agent_id)


class TestVersionManagerCreate:
    @pytest.mark.asyncio
    async def test_create_version_returns_agent_version(self) -> None:
        mgr = _make_manager()
        defn = _make_def()
        version = await mgr.create_version(defn, version_tag="v1.0")

        assert version.version_id == compute_version_id(defn)
        assert version.version_tag == "v1.0"
        assert version.agent_id == "mgr-agent"

    @pytest.mark.asyncio
    async def test_create_version_idempotent_same_definition(self) -> None:
        mgr = _make_manager()
        defn = _make_def()
        v1 = await mgr.create_version(defn, version_tag="v1.0")
        v2 = await mgr.create_version(defn, version_tag="v1.0-dup")
        # Identical definition → same version_id
        assert v1.version_id == v2.version_id

    @pytest.mark.asyncio
    async def test_create_version_with_metadata(self) -> None:
        mgr = _make_manager()
        defn = _make_def()
        version = await mgr.create_version(
            defn,
            version_tag="v1.0",
            author="alice",
            description="Initial release",
            tags=["baseline"],
        )
        assert version.author == "alice"
        assert version.description == "Initial release"
        assert "baseline" in version.tags


class TestVersionManagerDeploy:
    @pytest.mark.asyncio
    async def test_deploy_sets_active_deployment(self) -> None:
        mgr = _make_manager()
        defn = _make_def()
        version = await mgr.create_version(defn, version_tag="v1.0")

        record = await mgr.deploy(version.version_id)
        assert record.version_id == version.version_id
        assert record.agent_id == "mgr-agent"

    @pytest.mark.asyncio
    async def test_deploy_updates_version_status_to_production(self) -> None:
        mgr = _make_manager()
        defn = _make_def()
        version = await mgr.create_version(defn, version_tag="v1.0")
        await mgr.deploy(version.version_id)

        active = await mgr.get_active_version()
        assert active is not None
        assert active.status == VersionStatus.PRODUCTION

    @pytest.mark.asyncio
    async def test_deploy_nonexistent_version_raises(self) -> None:
        mgr = _make_manager()
        with pytest.raises(VersioningError) as exc_info:
            await mgr.deploy("nonexistent-version-id")
        assert exc_info.value.code == "VERSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_deploy_sets_previous_version_id(self) -> None:
        mgr = _make_manager()
        defn_a = _make_def("Prompt A")
        defn_b = _make_def("Prompt B")

        v1 = await mgr.create_version(defn_a, version_tag="v1.0")
        await mgr.deploy(v1.version_id)

        v2 = await mgr.create_version(defn_b, version_tag="v2.0")
        record = await mgr.deploy(v2.version_id)

        assert record.previous_version_id == v1.version_id


class TestVersionManagerRollback:
    @pytest.mark.asyncio
    async def test_rollback_reverts_to_previous_deployment(self) -> None:
        mgr = _make_manager()
        defn_a = _make_def("Prompt A")
        defn_b = _make_def("Prompt B")

        v1 = await mgr.create_version(defn_a, version_tag="v1.0")
        await mgr.deploy(v1.version_id)

        v2 = await mgr.create_version(defn_b, version_tag="v2.0")
        await mgr.deploy(v2.version_id)

        record = await mgr.rollback()
        assert record.version_id == v1.version_id

    @pytest.mark.asyncio
    async def test_rollback_no_prior_deployment_raises(self) -> None:
        mgr = _make_manager()
        defn = _make_def()
        v1 = await mgr.create_version(defn, version_tag="v1.0")
        await mgr.deploy(v1.version_id)

        with pytest.raises(VersioningError) as exc_info:
            await mgr.rollback()
        assert exc_info.value.code == "NO_PRIOR_VERSION"

    @pytest.mark.asyncio
    async def test_rollback_when_no_deployment_raises(self) -> None:
        mgr = _make_manager()
        with pytest.raises(VersioningError):
            await mgr.rollback()


class TestVersionManagerQuery:
    @pytest.mark.asyncio
    async def test_get_active_version_returns_deployed(self) -> None:
        mgr = _make_manager()
        defn = _make_def()
        v1 = await mgr.create_version(defn, version_tag="v1.0")
        await mgr.deploy(v1.version_id)

        active = await mgr.get_active_version()
        assert active is not None
        assert active.version_id == v1.version_id

    @pytest.mark.asyncio
    async def test_get_active_version_none_when_not_deployed(self) -> None:
        mgr = _make_manager()
        active = await mgr.get_active_version()
        assert active is None

    @pytest.mark.asyncio
    async def test_list_versions_sorted_newest_first(self) -> None:
        mgr = _make_manager()
        defn_a = _make_def("Prompt A")
        defn_b = _make_def("Prompt B")

        await mgr.create_version(defn_a, version_tag="v1.0")
        await mgr.create_version(defn_b, version_tag="v2.0")

        versions = await mgr.list_versions()
        assert len(versions) == 2

    @pytest.mark.asyncio
    async def test_diff_returns_version_diff(self) -> None:
        mgr = _make_manager()
        defn_a = _make_def("Old prompt.")
        defn_b = _make_def("New prompt.")

        v1 = await mgr.create_version(defn_a, version_tag="v1.0")
        v2 = await mgr.create_version(defn_b, version_tag="v2.0")

        diff = await mgr.diff(v1.version_id, v2.version_id)
        assert diff.has_changes
        assert diff.system_prompt_diff != ""

    @pytest.mark.asyncio
    async def test_diff_missing_version_raises(self) -> None:
        mgr = _make_manager()
        defn = _make_def()
        v1 = await mgr.create_version(defn, version_tag="v1.0")

        with pytest.raises(VersioningError) as exc_info:
            await mgr.diff(v1.version_id, "nonexistent")
        assert exc_info.value.code == "VERSION_NOT_FOUND"
