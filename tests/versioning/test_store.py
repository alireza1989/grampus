"""Tests for VersionStore — Dapr-backed persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.types import AgentDefinition
from grampus.versioning.store import VersionStore
from grampus.versioning.types import (
    AgentVersion,
    DeploymentRecord,
    VersionStatus,
    compute_version_id,
)


def _make_def(name: str = "test-agent") -> AgentDefinition:
    return AgentDefinition(
        name=name,
        model="claude-sonnet-4-6",
        system_prompt="You are helpful.",
        tools=["search"],
    )


def _make_version(defn: AgentDefinition, tag: str = "v1.0") -> AgentVersion:
    return AgentVersion(
        version_id=compute_version_id(defn),
        agent_id=defn.name,
        version_tag=tag,
        definition=defn,
    )


def _make_state_store() -> MagicMock:
    """Create a mock DaprStateStore that stores data in memory."""
    store: dict[tuple[str, str], object] = {}

    mock = MagicMock()

    async def save(entity_type: str, entity_id: str, model: object, **kwargs: object) -> None:
        store[(entity_type, entity_id)] = model

    async def get(entity_type: str, entity_id: str, cls: type) -> tuple[object | None, str]:
        val = store.get((entity_type, entity_id))
        if val is None:
            return None, ""
        # Round-trip through JSON for realistic deserialization
        if hasattr(val, "model_dump_json"):
            restored = cls.model_validate_json(val.model_dump_json())  # type: ignore[attr-defined]
            return restored, "etag-1"
        return val, "etag-1"

    mock.save = AsyncMock(side_effect=save)
    mock.get = AsyncMock(side_effect=get)
    return mock


class TestVersionStoreSaveGet:
    @pytest.mark.asyncio
    async def test_save_and_get_version(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)
        defn = _make_def()
        version = _make_version(defn)

        await vs.save_version(version)
        result = await vs.get_version(version.version_id)

        assert result is not None
        assert result.version_id == version.version_id
        assert result.version_tag == version.version_tag

    @pytest.mark.asyncio
    async def test_get_nonexistent_version_returns_none(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)

        result = await vs.get_version("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_versions_returns_all_for_agent(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)
        defn_a = AgentDefinition(name="agent-x", model="m", system_prompt="A", tools=[])
        defn_b = AgentDefinition(name="agent-x", model="m", system_prompt="B", tools=[])

        v1 = _make_version(defn_a, "v1.0")
        v1 = v1.model_copy(update={"agent_id": "agent-x", "version_id": compute_version_id(defn_a)})
        v2 = AgentVersion(
            version_id=compute_version_id(defn_b),
            agent_id="agent-x",
            version_tag="v2.0",
            definition=defn_b,
        )

        await vs.save_version(v1)
        await vs.save_version(v2)

        versions = await vs.list_versions("agent-x")
        assert len(versions) == 2
        ids = {v.version_id for v in versions}
        assert v1.version_id in ids
        assert v2.version_id in ids

    @pytest.mark.asyncio
    async def test_list_versions_empty_returns_empty_list(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)
        versions = await vs.list_versions("nonexistent-agent")
        assert versions == []

    @pytest.mark.asyncio
    async def test_list_versions_sorted_newest_first(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)
        defn_a = AgentDefinition(name="sorted-agent", model="m", system_prompt="A", tools=[])
        defn_b = AgentDefinition(name="sorted-agent", model="m", system_prompt="B", tools=[])

        earlier = datetime(2024, 1, 1, tzinfo=UTC)
        later = datetime(2025, 1, 1, tzinfo=UTC)

        v1 = AgentVersion(
            version_id=compute_version_id(defn_a),
            agent_id="sorted-agent",
            version_tag="v1",
            definition=defn_a,
            created_at=earlier,
        )
        v2 = AgentVersion(
            version_id=compute_version_id(defn_b),
            agent_id="sorted-agent",
            version_tag="v2",
            definition=defn_b,
            created_at=later,
        )

        await vs.save_version(v1)
        await vs.save_version(v2)

        versions = await vs.list_versions("sorted-agent")
        assert versions[0].version_tag == "v2"


class TestVersionStoreDeployment:
    @pytest.mark.asyncio
    async def test_save_and_get_deployment(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)
        rec = DeploymentRecord(
            agent_id="agent-y",
            version_id="v-abc",
            deployed_at=datetime.now(UTC),
        )
        await vs.save_deployment(rec)
        result = await vs.get_deployment("agent-y")
        assert result is not None
        assert result.version_id == "v-abc"

    @pytest.mark.asyncio
    async def test_get_deployment_nonexistent_returns_none(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)
        result = await vs.get_deployment("no-such-agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_deployment_history_newest_first(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)

        rec1 = DeploymentRecord(
            agent_id="hist-agent",
            version_id="v1",
            deployed_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        rec2 = DeploymentRecord(
            agent_id="hist-agent",
            version_id="v2",
            deployed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await vs.save_deployment(rec1)
        await vs.save_deployment(rec2)

        history = await vs.get_deployment_history("hist-agent")
        assert len(history) >= 2
        assert history[0].version_id == "v2"

    @pytest.mark.asyncio
    async def test_deployment_history_empty_returns_empty(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)
        history = await vs.get_deployment_history("empty-agent")
        assert history == []

    @pytest.mark.asyncio
    async def test_update_version_status(self) -> None:
        state_store = _make_state_store()
        vs = VersionStore(state_store)
        defn = _make_def()
        version = _make_version(defn)
        await vs.save_version(version)

        await vs.update_version_status(version.version_id, VersionStatus.PRODUCTION)

        updated = await vs.get_version(version.version_id)
        assert updated is not None
        assert updated.status == VersionStatus.PRODUCTION
