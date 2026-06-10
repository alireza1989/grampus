"""Tests for versioning types and pure functions."""

from __future__ import annotations

from datetime import UTC, datetime

from nexus.core.types import AgentDefinition
from nexus.versioning.types import (
    AgentVersion,
    DeploymentRecord,
    VersionDiff,
    VersionStatus,
    compute_version_id,
    diff_versions,
)


def _make_def(**kwargs: object) -> AgentDefinition:
    defaults: dict[str, object] = {
        "name": "test-agent",
        "model": "claude-sonnet-4-6",
        "system_prompt": "You are helpful.",
        "tools": ["search", "calculator"],
        "temperature": 0.0,
    }
    defaults.update(kwargs)
    return AgentDefinition(**defaults)


def _make_version(definition: AgentDefinition, **kwargs: object) -> AgentVersion:
    defaults: dict[str, object] = {
        "version_id": compute_version_id(definition),
        "agent_id": "test-agent",
        "version_tag": "v1.0",
        "definition": definition,
    }
    defaults.update(kwargs)
    return AgentVersion(**defaults)


class TestVersionStatus:
    def test_enum_values(self) -> None:
        assert VersionStatus.DRAFT == "draft"
        assert VersionStatus.CANARY == "canary"
        assert VersionStatus.PRODUCTION == "production"
        assert VersionStatus.RETIRED == "retired"

    def test_all_four_values_exist(self) -> None:
        values = {s.value for s in VersionStatus}
        assert values == {"draft", "canary", "production", "retired"}


class TestAgentVersionModel:
    def test_round_trip_serialization(self) -> None:
        defn = _make_def()
        v = _make_version(defn)
        as_json = v.model_dump_json()
        restored = AgentVersion.model_validate_json(as_json)
        assert restored.version_id == v.version_id
        assert restored.definition.model == v.definition.model
        assert restored.status == VersionStatus.DRAFT

    def test_default_status_is_draft(self) -> None:
        defn = _make_def()
        v = _make_version(defn)
        assert v.status == VersionStatus.DRAFT

    def test_created_at_defaults_to_utc_now(self) -> None:
        defn = _make_def()
        v = _make_version(defn)
        assert v.created_at.tzinfo is not None

    def test_frozen_model_cannot_be_mutated(self) -> None:
        defn = _make_def()
        v = _make_version(defn)
        try:
            v.version_tag = "v2.0"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AssertionError:
            raise
        except Exception:
            pass


class TestDeploymentRecordModel:
    def test_round_trip_serialization(self) -> None:
        rec = DeploymentRecord(
            agent_id="test-agent",
            version_id="abc123",
            deployed_at=datetime.now(UTC),
        )
        as_json = rec.model_dump_json()
        restored = DeploymentRecord.model_validate_json(as_json)
        assert restored.agent_id == rec.agent_id
        assert restored.version_id == rec.version_id
        assert restored.deployed_by == "system"
        assert restored.previous_version_id is None


class TestComputeVersionId:
    def test_deterministic_same_definition(self) -> None:
        defn1 = _make_def()
        defn2 = _make_def()
        assert compute_version_id(defn1) == compute_version_id(defn2)

    def test_deterministic_across_calls(self) -> None:
        defn = _make_def()
        id1 = compute_version_id(defn)
        id2 = compute_version_id(defn)
        assert id1 == id2

    def test_different_system_prompt_gives_different_id(self) -> None:
        defn_a = _make_def(system_prompt="You are helpful.")
        defn_b = _make_def(system_prompt="You are concise.")
        assert compute_version_id(defn_a) != compute_version_id(defn_b)

    def test_different_tools_gives_different_id(self) -> None:
        defn_a = _make_def(tools=["search"])
        defn_b = _make_def(tools=["search", "calculator"])
        assert compute_version_id(defn_a) != compute_version_id(defn_b)

    def test_different_temperature_gives_different_id(self) -> None:
        defn_a = _make_def(temperature=0.0)
        defn_b = _make_def(temperature=0.7)
        assert compute_version_id(defn_a) != compute_version_id(defn_b)

    def test_returns_sha256_hex_string(self) -> None:
        defn = _make_def()
        vid = compute_version_id(defn)
        assert len(vid) == 64
        assert all(c in "0123456789abcdef" for c in vid)

    def test_tool_order_independent(self) -> None:
        defn_a = _make_def(tools=["search", "calculator"])
        defn_b = _make_def(tools=["calculator", "search"])
        # Tools list is sorted before hashing — same set should produce same ID
        assert compute_version_id(defn_a) == compute_version_id(defn_b)


class TestDiffVersions:
    def test_identical_versions_has_no_changes(self) -> None:
        defn = _make_def()
        v1 = _make_version(defn, version_tag="v1.0")
        v2 = _make_version(defn, version_tag="v1.0")
        diff = diff_versions(v1, v2)
        assert not diff.has_changes
        assert diff.system_prompt_diff == ""
        assert diff.tools_added == []
        assert diff.tools_removed == []
        assert diff.config_changes == {}

    def test_detects_system_prompt_change(self) -> None:
        defn_a = _make_def(system_prompt="Old prompt.")
        defn_b = _make_def(system_prompt="New prompt.")
        v1 = _make_version(defn_a, version_id=compute_version_id(defn_a), version_tag="v1")
        v2 = _make_version(defn_b, version_id=compute_version_id(defn_b), version_tag="v2")
        diff = diff_versions(v1, v2)
        assert diff.has_changes
        assert diff.system_prompt_diff != ""

    def test_detects_tools_added(self) -> None:
        defn_a = _make_def(tools=["search"])
        defn_b = _make_def(tools=["search", "calculator"])
        v1 = _make_version(defn_a, version_id=compute_version_id(defn_a), version_tag="v1")
        v2 = _make_version(defn_b, version_id=compute_version_id(defn_b), version_tag="v2")
        diff = diff_versions(v1, v2)
        assert "calculator" in diff.tools_added
        assert diff.tools_removed == []
        assert diff.has_changes

    def test_detects_tools_removed(self) -> None:
        defn_a = _make_def(tools=["search", "calculator"])
        defn_b = _make_def(tools=["search"])
        v1 = _make_version(defn_a, version_id=compute_version_id(defn_a), version_tag="v1")
        v2 = _make_version(defn_b, version_id=compute_version_id(defn_b), version_tag="v2")
        diff = diff_versions(v1, v2)
        assert "calculator" in diff.tools_removed
        assert diff.tools_added == []
        assert diff.has_changes

    def test_detects_temperature_change(self) -> None:
        defn_a = _make_def(temperature=0.0)
        defn_b = _make_def(temperature=0.7)
        v1 = _make_version(defn_a, version_id=compute_version_id(defn_a), version_tag="v1")
        v2 = _make_version(defn_b, version_id=compute_version_id(defn_b), version_tag="v2")
        diff = diff_versions(v1, v2)
        assert "temperature" in diff.config_changes
        assert diff.config_changes["temperature"] == (0.0, 0.7)
        assert diff.has_changes

    def test_version_diff_round_trip(self) -> None:
        defn_a = _make_def()
        defn_b = _make_def(system_prompt="Changed.")
        v1 = _make_version(defn_a, version_id=compute_version_id(defn_a), version_tag="v1")
        v2 = _make_version(defn_b, version_id=compute_version_id(defn_b), version_tag="v2")
        diff = diff_versions(v1, v2)
        as_json = diff.model_dump_json()
        restored = VersionDiff.model_validate_json(as_json)
        assert restored.has_changes == diff.has_changes
