"""Tests for Phase E36 — Artifact-Centric Collaboration.

All tests use asyncio_mode = "auto". No real Dapr or LLM calls —
mocked via AsyncMock and in-memory fakes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.errors import (
    ArtifactConflictError,
    ArtifactSectionNotFoundError,
    GrampusError,
)
from grampus.core.types import AgentStatus, ExecutionResult, TokenUsage
from grampus.orchestration.artifact.collaborator import ArtifactCollaborator
from grampus.orchestration.artifact.conflict_detector import ConflictDetector
from grampus.orchestration.artifact.crew import ArtifactCrew
from grampus.orchestration.artifact.lock_manager import SectionLockManager
from grampus.orchestration.artifact.schema import SchemaValidator
from grampus.orchestration.artifact.store import ArtifactStore
from grampus.orchestration.artifact.types import (
    Artifact,
    ArtifactContentType,
    ArtifactSchema,
    ArtifactSection,
    ConflictType,
    EditOperation,
    SectionOwnershipState,
    SectionSchema,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_schema(
    sections: list[SectionSchema] | None = None,
) -> ArtifactSchema:
    if sections is None:
        sections = [
            SectionSchema(
                section_id="intro",
                description="Introduction section",
                content_type=ArtifactContentType.MARKDOWN,
            ),
            SectionSchema(
                section_id="body",
                description="Body section",
                content_type=ArtifactContentType.MARKDOWN,
                dependencies=["intro"],
            ),
        ]
    return ArtifactSchema(
        artifact_type="document",
        description="Test document",
        sections=sections,
    )


def _make_section(
    section_id: str = "intro",
    version: int = 0,
    content: Any = None,
    state: SectionOwnershipState = SectionOwnershipState.UNOWNED,
    owner: str | None = None,
) -> ArtifactSection:
    return ArtifactSection(
        section_id=section_id,
        schema_ref=section_id,
        content=content,
        version=version,
        ownership_state=state,
        owner_agent_id=owner,
    )


def _make_artifact(schema: ArtifactSchema | None = None, artifact_id: str = "art-1") -> Artifact:
    s = schema or _make_schema()
    sections = {sec.section_id: _make_section(section_id=sec.section_id) for sec in s.sections}
    return Artifact(
        artifact_id=artifact_id,
        artifact_type=s.artifact_type,
        artifact_schema=s,
        sections=sections,
    )


def _make_store() -> tuple[ArtifactStore, dict[str, Any]]:
    """Return (store, backing_dict) using an in-memory fake DaprStateStore."""
    backing: dict[str, Any] = {}

    async def _save(entity_type: str, entity_id: str, model: Any, **kwargs: Any) -> None:
        backing[f"{entity_type}:{entity_id}"] = model.model_copy(deep=True)

    async def _get(entity_type: str, entity_id: str, cls: type) -> tuple[Any, str]:
        key = f"{entity_type}:{entity_id}"
        return backing.get(key), ""

    state_store = MagicMock()
    state_store.save = AsyncMock(side_effect=_save)
    state_store.get = AsyncMock(side_effect=_get)

    validator = SchemaValidator()
    store = ArtifactStore(state_store=state_store, validator=validator)
    return store, backing


def _execution_result(output: str = "section content") -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model="fake"
        ),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


# ---------------------------------------------------------------------------
# Types tests
# ---------------------------------------------------------------------------


def test_artifact_schema_get_section_found() -> None:
    schema = _make_schema()
    section = schema.get_section("intro")
    assert section is not None
    assert section.section_id == "intro"


def test_artifact_schema_get_section_not_found() -> None:
    schema = _make_schema()
    assert schema.get_section("nonexistent") is None


def test_artifact_schema_dependency_ids() -> None:
    schema = _make_schema()
    dep_ids = schema.dependency_ids()
    assert dep_ids["intro"] == []
    assert dep_ids["body"] == ["intro"]


# ---------------------------------------------------------------------------
# SchemaValidator tests
# ---------------------------------------------------------------------------


def test_validate_text_content_passes() -> None:
    validator = SchemaValidator()
    schema = SectionSchema(
        section_id="s1",
        description="test",
        content_type=ArtifactContentType.TEXT,
    )
    section = _make_section(content="hello world")
    assert validator.validate(section, schema) is None


def test_validate_json_missing_required_fields_fails() -> None:
    validator = SchemaValidator()
    schema = SectionSchema(
        section_id="s1",
        description="test",
        content_type=ArtifactContentType.JSON,
        required_fields=["title", "body"],
    )
    section = _make_section(content={"title": "hello"})
    conflict = validator.validate(section, schema)
    assert conflict is not None
    assert conflict.conflict_type == ConflictType.SCHEMA_VALIDATION
    assert "body" in conflict.description
    assert conflict.resolution == "reject"


def test_validate_json_correct_required_fields_passes() -> None:
    validator = SchemaValidator()
    schema = SectionSchema(
        section_id="s1",
        description="test",
        content_type=ArtifactContentType.JSON,
        required_fields=["title", "body"],
    )
    section = _make_section(content={"title": "hello", "body": "world"})
    assert validator.validate(section, schema) is None


def test_validate_exceeds_max_tokens_fails() -> None:
    validator = SchemaValidator()
    schema = SectionSchema(
        section_id="s1",
        description="test",
        content_type=ArtifactContentType.TEXT,
        max_tokens=2,
    )
    section = _make_section(content="a" * 100)
    conflict = validator.validate(section, schema)
    assert conflict is not None
    assert "max_tokens" in conflict.description


def test_validate_wrong_content_type_fails() -> None:
    validator = SchemaValidator()
    schema = SectionSchema(
        section_id="s1",
        description="test",
        content_type=ArtifactContentType.TEXT,
    )
    section = _make_section(content={"key": "value"})
    conflict = validator.validate(section, schema)
    assert conflict is not None
    assert conflict.conflict_type == ConflictType.SCHEMA_VALIDATION


# ---------------------------------------------------------------------------
# ArtifactStore tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_initializes_all_sections_unowned() -> None:
    store, backing = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    assert artifact.artifact_id is not None
    for sec in artifact.sections.values():
        assert sec.ownership_state == SectionOwnershipState.UNOWNED
        assert sec.version == 0
        assert sec.content is None


@pytest.mark.asyncio
async def test_write_section_increments_version() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)

    await store.update_section_state(
        artifact.artifact_id, "intro", SectionOwnershipState.CLAIMED, "agent-1"
    )

    op = EditOperation(
        op_type="write",
        artifact_id=artifact.artifact_id,
        section_id="intro",
        agent_id="agent-1",
        content="Hello world",
    )
    result = await store.write_section(op)
    assert result.success is True
    assert result.new_version == 1


@pytest.mark.asyncio
async def test_write_section_requires_claimed_ownership() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)

    op = EditOperation(
        op_type="write",
        artifact_id=artifact.artifact_id,
        section_id="intro",
        agent_id="agent-1",
        content="Hello",
    )
    result = await store.write_section(op)
    assert result.success is False
    assert result.conflict is not None
    assert result.conflict.conflict_type == ConflictType.OWNERSHIP


@pytest.mark.asyncio
async def test_write_section_version_mismatch_returns_conflict() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    await store.update_section_state(aid, "intro", SectionOwnershipState.CLAIMED, "agent-1")
    op1 = EditOperation(
        op_type="write", artifact_id=aid, section_id="intro", agent_id="agent-1", content="v1"
    )
    await store.write_section(op1)

    await store.update_section_state(aid, "intro", SectionOwnershipState.CLAIMED, "agent-1")
    op2 = EditOperation(
        op_type="write",
        artifact_id=aid,
        section_id="intro",
        agent_id="agent-1",
        content="v2",
        expected_version=0,
    )
    result = await store.write_section(op2)
    assert result.success is False
    assert result.conflict is not None
    assert result.conflict.conflict_type == ConflictType.VERSION_MISMATCH


@pytest.mark.asyncio
async def test_write_section_schema_validation_failure() -> None:
    store, _ = _make_store()
    schema = ArtifactSchema(
        artifact_type="report",
        description="test",
        sections=[
            SectionSchema(
                section_id="data",
                description="data section",
                content_type=ArtifactContentType.JSON,
                required_fields=["results"],
            )
        ],
    )
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    await store.update_section_state(aid, "data", SectionOwnershipState.CLAIMED, "agent-1")
    op = EditOperation(
        op_type="write",
        artifact_id=aid,
        section_id="data",
        agent_id="agent-1",
        content={"wrong_field": "value"},
    )
    result = await store.write_section(op)
    assert result.success is False
    assert result.conflict is not None
    assert result.conflict.conflict_type == ConflictType.SCHEMA_VALIDATION


@pytest.mark.asyncio
async def test_read_section_returns_content() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    await store.update_section_state(aid, "intro", SectionOwnershipState.CLAIMED, "agent-1")
    op = EditOperation(
        op_type="write", artifact_id=aid, section_id="intro", agent_id="agent-1", content="content"
    )
    await store.write_section(op)
    await store.update_section_state(aid, "intro", SectionOwnershipState.MERGED)

    section = await store.read_section(aid, "intro", "reader-1")
    assert section.content == "content"


@pytest.mark.asyncio
async def test_complete_artifact_fails_if_sections_not_merged() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)

    with pytest.raises(ArtifactConflictError) as exc_info:
        await store.complete_artifact(artifact.artifact_id)
    assert "SECTIONS_NOT_MERGED" in exc_info.value.code


@pytest.mark.asyncio
async def test_complete_artifact_succeeds_when_all_merged() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    for sec in schema.sections:
        await store.update_section_state(aid, sec.section_id, SectionOwnershipState.CLAIMED, "a1")
        op = EditOperation(
            op_type="write",
            artifact_id=aid,
            section_id=sec.section_id,
            agent_id="a1",
            content="done",
        )
        await store.write_section(op)
        await store.update_section_state(aid, sec.section_id, SectionOwnershipState.MERGED)

    completed = await store.complete_artifact(aid)
    assert completed.completed_at is not None


@pytest.mark.asyncio
async def test_load_returns_persisted_artifact() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    original = await store.create(schema, artifact_id="known-id")
    loaded = await store.load("known-id")
    assert loaded.artifact_id == "known-id"
    assert loaded.artifact_type == original.artifact_type


# ---------------------------------------------------------------------------
# SectionLockManager tests
# ---------------------------------------------------------------------------


def _make_lock_factory(succeed: bool = True) -> Any:
    """Return a lock factory whose __aenter__ succeeds or raises."""
    from grampus.core.errors import LockAcquisitionError

    class FakeLock:
        async def __aenter__(self) -> FakeLock:
            if not succeed:
                raise LockAcquisitionError("lock busy", code="LOCK_ACQUISITION_ERROR")
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

    def factory(**kwargs: Any) -> FakeLock:
        return FakeLock()

    return factory


@pytest.mark.asyncio
async def test_claim_succeeds_on_unowned_section() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)

    manager = SectionLockManager(_make_lock_factory(succeed=True), store)
    success = await manager.claim(artifact.artifact_id, "intro", "agent-1")
    assert success is True

    section = await store._load_section(artifact.artifact_id, "intro")
    assert section is not None
    assert section.ownership_state == SectionOwnershipState.CLAIMED
    assert section.owner_agent_id == "agent-1"


@pytest.mark.asyncio
async def test_claim_returns_false_when_already_claimed_by_other() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)

    await store.update_section_state(
        artifact.artifact_id, "intro", SectionOwnershipState.CLAIMED, "agent-1"
    )

    manager = SectionLockManager(_make_lock_factory(succeed=True), store)
    success = await manager.claim(artifact.artifact_id, "intro", "agent-2")
    assert success is False


@pytest.mark.asyncio
async def test_claim_idempotent_for_same_agent() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)

    await store.update_section_state(
        artifact.artifact_id, "intro", SectionOwnershipState.CLAIMED, "agent-1"
    )

    manager = SectionLockManager(_make_lock_factory(succeed=True), store)
    success = await manager.claim(artifact.artifact_id, "intro", "agent-1")
    assert success is True


@pytest.mark.asyncio
async def test_release_sets_state_to_merged() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    await store.update_section_state(aid, "intro", SectionOwnershipState.CLAIMED, "agent-1")
    manager = SectionLockManager(_make_lock_factory(), store)
    await manager.release(aid, "intro", "agent-1", mark_complete=True)

    section = await store._load_section(aid, "intro")
    assert section is not None
    assert section.ownership_state == SectionOwnershipState.MERGED


@pytest.mark.asyncio
async def test_release_sets_state_to_unowned_when_incomplete() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    await store.update_section_state(aid, "intro", SectionOwnershipState.CLAIMED, "agent-1")
    manager = SectionLockManager(_make_lock_factory(), store)
    await manager.release(aid, "intro", "agent-1", mark_complete=False)

    section = await store._load_section(aid, "intro")
    assert section is not None
    assert section.ownership_state == SectionOwnershipState.UNOWNED


@pytest.mark.asyncio
async def test_get_owner_returns_agent_id() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)

    await store.update_section_state(
        artifact.artifact_id, "intro", SectionOwnershipState.CLAIMED, "agent-42"
    )
    manager = SectionLockManager(_make_lock_factory(), store)
    owner = await manager.get_owner(artifact.artifact_id, "intro")
    assert owner == "agent-42"


# ---------------------------------------------------------------------------
# ConflictDetector tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dependency_version_check_passes_when_deps_completed() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    await store.update_section_state(aid, "intro", SectionOwnershipState.CLAIMED, "a1")
    op = EditOperation(
        op_type="write", artifact_id=aid, section_id="intro", agent_id="a1", content="done"
    )
    await store.write_section(op)
    await store.update_section_state(aid, "intro", SectionOwnershipState.MERGED)

    detector = ConflictDetector(store)
    body_schema = schema.get_section("body")
    assert body_schema is not None
    conflicts = await detector.check(aid, "body", body_schema, "body content")
    dep_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY_VERSION]
    assert len(dep_conflicts) == 0


@pytest.mark.asyncio
async def test_dependency_version_check_fails_when_dep_unwritten() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    detector = ConflictDetector(store)
    body_schema = schema.get_section("body")
    assert body_schema is not None
    conflicts = await detector.check(aid, "body", body_schema, "body content")
    dep_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY_VERSION]
    assert len(dep_conflicts) == 1
    assert "intro" in dep_conflicts[0].description


@pytest.mark.asyncio
async def test_structural_check_returns_advisory_not_rejection() -> None:
    store, _ = _make_store()
    schema = ArtifactSchema(
        artifact_type="report",
        description="test",
        sections=[
            SectionSchema(
                section_id="data",
                description="data",
                content_type=ArtifactContentType.JSON,
                required_fields=["key1"],
            )
        ],
    )
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    data_schema = schema.get_section("data")
    assert data_schema is not None
    detector = ConflictDetector(store)
    conflicts = await detector.check(aid, "data", data_schema, {"wrong": "value"})
    structural = [c for c in conflicts if c.conflict_type == ConflictType.SCHEMA_VALIDATION]
    assert any(c.resolution == "human_review" for c in structural)


# ---------------------------------------------------------------------------
# ArtifactCollaborator tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_scoped_context_excludes_incomplete_dependencies() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    manager = SectionLockManager(_make_lock_factory(), store)
    detector = ConflictDetector(store)
    collaborator = ArtifactCollaborator("agent-1", store, manager, detector)

    scoped = await collaborator.get_scoped_context(aid, "body")
    assert "intro" not in scoped.completed_dependencies


@pytest.mark.asyncio
async def test_get_scoped_context_includes_one_line_summaries() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    await store.update_section_state(aid, "intro", SectionOwnershipState.CLAIMED, "agent-1")
    op = EditOperation(
        op_type="write",
        artifact_id=aid,
        section_id="intro",
        agent_id="agent-1",
        content="Intro content here",
    )
    await store.write_section(op)
    await store.update_section_state(aid, "intro", SectionOwnershipState.MERGED)

    manager = SectionLockManager(_make_lock_factory(), store)
    detector = ConflictDetector(store)
    collaborator = ArtifactCollaborator("agent-1", store, manager, detector)

    scoped = await collaborator.get_scoped_context(aid, "body")
    assert "intro" in scoped.completed_dependencies
    assert len(scoped.completed_dependencies["intro"]) <= 200


@pytest.mark.asyncio
async def test_write_with_conflict_detector_rejection_does_not_persist() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    manager = SectionLockManager(_make_lock_factory(), store)
    detector = ConflictDetector(store)
    collaborator = ArtifactCollaborator("agent-1", store, manager, detector)

    await store.update_section_state(aid, "body", SectionOwnershipState.CLAIMED, "agent-1")

    result = await collaborator.write_section(aid, "body", "body content")
    assert result.success is False
    conflict = result.conflict
    assert conflict is not None
    assert conflict.conflict_type == ConflictType.DEPENDENCY_VERSION

    section = await store._load_section(aid, "body")
    assert section is not None
    assert section.version == 0


@pytest.mark.asyncio
async def test_full_claim_write_release_lifecycle() -> None:
    store, _ = _make_store()
    schema = _make_schema()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    manager = SectionLockManager(_make_lock_factory(), store)
    detector = ConflictDetector(store)
    collaborator = ArtifactCollaborator("agent-1", store, manager, detector)

    claimed = await collaborator.claim_section(aid, "intro")
    assert claimed is True

    result = await collaborator.write_section(aid, "intro", "My introduction text")
    assert result.success is True
    assert result.new_version == 1

    await collaborator.release_section(aid, "intro", mark_complete=True)
    section = await store._load_section(aid, "intro")
    assert section is not None
    assert section.ownership_state == SectionOwnershipState.MERGED


# ---------------------------------------------------------------------------
# ArtifactCrew tests
# ---------------------------------------------------------------------------


def _make_crew_components(
    schema: ArtifactSchema,
    num_agents: int = 2,
    agent_output: str = "written content",
) -> tuple[ArtifactStore, ArtifactCrew]:
    store, _ = _make_store()

    fake_runner = MagicMock()
    fake_runner.run = AsyncMock(return_value=_execution_result(agent_output))

    agents = [fake_runner] * num_agents

    collaborators = []
    for i in range(num_agents):
        lock_mgr = SectionLockManager(_make_lock_factory(), store)
        detector = ConflictDetector(store)
        collab = ArtifactCollaborator(f"agent-{i}", store, lock_mgr, detector)
        collaborators.append(collab)

    crew = ArtifactCrew(agents=agents, collaborators=collaborators, store=store, max_retries=0)
    return store, crew


@pytest.mark.asyncio
async def test_build_waves_simple_chain() -> None:
    schema = ArtifactSchema(
        artifact_type="doc",
        description="test",
        sections=[
            SectionSchema(section_id="A", description="A", content_type=ArtifactContentType.TEXT),
            SectionSchema(
                section_id="B",
                description="B",
                content_type=ArtifactContentType.TEXT,
                dependencies=["A"],
            ),
            SectionSchema(
                section_id="C",
                description="C",
                content_type=ArtifactContentType.TEXT,
                dependencies=["B"],
            ),
        ],
    )
    store, _ = _make_store()
    lock_mgr = SectionLockManager(_make_lock_factory(), store)
    detector = ConflictDetector(store)
    collab = ArtifactCollaborator("a1", store, lock_mgr, detector)
    fake_runner = MagicMock()
    fake_runner.run = AsyncMock(return_value=_execution_result())
    crew = ArtifactCrew(agents=[fake_runner], collaborators=[collab], store=store)

    waves = await crew._build_waves(schema)
    assert len(waves) == 3
    assert waves[0] == ["A"]
    assert waves[1] == ["B"]
    assert waves[2] == ["C"]


@pytest.mark.asyncio
async def test_build_waves_parallel() -> None:
    schema = ArtifactSchema(
        artifact_type="doc",
        description="test",
        sections=[
            SectionSchema(section_id="A", description="A", content_type=ArtifactContentType.TEXT),
            SectionSchema(section_id="B", description="B", content_type=ArtifactContentType.TEXT),
        ],
    )
    store, _ = _make_store()
    lock_mgr = SectionLockManager(_make_lock_factory(), store)
    detector = ConflictDetector(store)
    collab = ArtifactCollaborator("a1", store, lock_mgr, detector)
    fake_runner = MagicMock()
    fake_runner.run = AsyncMock(return_value=_execution_result())
    crew = ArtifactCrew(agents=[fake_runner], collaborators=[collab], store=store)

    waves = await crew._build_waves(schema)
    assert len(waves) == 1
    assert set(waves[0]) == {"A", "B"}


@pytest.mark.asyncio
async def test_build_waves_circular_raises() -> None:
    schema = ArtifactSchema(
        artifact_type="doc",
        description="test",
        sections=[
            SectionSchema(
                section_id="A",
                description="A",
                content_type=ArtifactContentType.TEXT,
                dependencies=["B"],
            ),
            SectionSchema(
                section_id="B",
                description="B",
                content_type=ArtifactContentType.TEXT,
                dependencies=["A"],
            ),
        ],
    )
    store, _ = _make_store()
    lock_mgr = SectionLockManager(_make_lock_factory(), store)
    detector = ConflictDetector(store)
    collab = ArtifactCollaborator("a1", store, lock_mgr, detector)
    fake_runner = MagicMock()
    crew = ArtifactCrew(agents=[fake_runner], collaborators=[collab], store=store)

    with pytest.raises(ArtifactConflictError) as exc_info:
        await crew._build_waves(schema)
    assert exc_info.value.code == "CIRCULAR_DEPENDENCY"


@pytest.mark.asyncio
async def test_run_two_section_artifact() -> None:
    schema = ArtifactSchema(
        artifact_type="doc",
        description="Two section doc",
        sections=[
            SectionSchema(
                section_id="sec1", description="First", content_type=ArtifactContentType.TEXT
            ),
            SectionSchema(
                section_id="sec2", description="Second", content_type=ArtifactContentType.TEXT
            ),
        ],
    )
    store, crew = _make_crew_components(schema, num_agents=2)
    artifact = await store.create(schema)

    completed = await crew.run(artifact.artifact_id, "Write the document")

    assert completed.completed_at is not None
    for sec in completed.sections.values():
        assert sec.ownership_state == SectionOwnershipState.MERGED
        assert sec.version >= 1


@pytest.mark.asyncio
async def test_integration_check_catches_schema_violation_after_wave() -> None:
    schema = ArtifactSchema(
        artifact_type="doc",
        description="test",
        sections=[
            SectionSchema(
                section_id="data",
                description="JSON data",
                content_type=ArtifactContentType.JSON,
                required_fields=["key"],
            )
        ],
    )
    store, _ = _make_store()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    await store.update_section_state(aid, "data", SectionOwnershipState.CLAIMED, "a1")
    await store._store.save(
        "artifact_section",
        store._build_section_key(aid, "data"),
        ArtifactSection(
            section_id="data",
            schema_ref="data",
            content={"wrong": "value"},
            version=1,
            ownership_state=SectionOwnershipState.MERGED,
        ),
    )

    lock_mgr = SectionLockManager(_make_lock_factory(), store)
    detector = ConflictDetector(store)
    collab = ArtifactCollaborator("a1", store, lock_mgr, detector)
    fake_runner = MagicMock()
    fake_runner.run = AsyncMock(return_value=_execution_result())
    crew = ArtifactCrew(agents=[fake_runner], collaborators=[collab], store=store)

    conflicts = await crew._integration_check(aid, ["data"])
    assert len(conflicts) > 0
    assert conflicts[0].conflict_type == ConflictType.SCHEMA_VALIDATION


# ---------------------------------------------------------------------------
# Error class tests
# ---------------------------------------------------------------------------


def test_artifact_conflict_error_is_grampus_error() -> None:
    err = ArtifactConflictError("test conflict", code="TEST_CODE")
    assert isinstance(err, GrampusError)
    assert err.code == "TEST_CODE"


def test_artifact_section_not_found_error_is_grampus_error() -> None:
    err = ArtifactSectionNotFoundError("not found", code="SECTION_NOT_FOUND")
    assert isinstance(err, GrampusError)


# ---------------------------------------------------------------------------
# Integration test (mocked Dapr + AgentRunner)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_artifact_crew_pipeline() -> None:
    """Full 3-section pipeline: verify all sections MERGED, versions > 0, artifact completed."""
    schema = ArtifactSchema(
        artifact_type="research_report",
        description="A 3-section research report",
        sections=[
            SectionSchema(
                section_id="background",
                description="Background section",
                content_type=ArtifactContentType.MARKDOWN,
            ),
            SectionSchema(
                section_id="analysis",
                description="Analysis section",
                content_type=ArtifactContentType.MARKDOWN,
                dependencies=["background"],
            ),
            SectionSchema(
                section_id="conclusion",
                description="Conclusion section",
                content_type=ArtifactContentType.MARKDOWN,
                dependencies=["background", "analysis"],
            ),
        ],
    )

    store, _ = _make_store()
    artifact = await store.create(schema)
    aid = artifact.artifact_id

    agents = []
    collaborators = []
    for i in range(2):
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=_execution_result(f"Content from agent-{i}"))
        agents.append(fake_runner)
        lock_mgr = SectionLockManager(_make_lock_factory(), store)
        detector = ConflictDetector(store)
        collab = ArtifactCollaborator(f"agent-{i}", store, lock_mgr, detector)
        collaborators.append(collab)

    crew = ArtifactCrew(agents=agents, collaborators=collaborators, store=store, max_retries=1)
    completed = await crew.run(aid, "Write a research report")

    assert completed.completed_at is not None
    assert len(completed.sections) == 3
    for sec in completed.sections.values():
        assert sec.ownership_state == SectionOwnershipState.MERGED
        assert sec.version > 0
    assert completed.global_version > 0
