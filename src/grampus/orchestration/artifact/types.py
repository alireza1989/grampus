"""Pydantic v2 types for artifact-centric collaboration (E36).

All models are pure data — no logic. Logic lives in store.py, schema.py, etc.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SectionOwnershipState(StrEnum):
    """MESI-inspired ownership states (Token Coherence, arXiv 2603.15183).

    Monotonic versioning + ownership states reduce synchronization cost from
    O(n×S×|D|) to O((n+W)×|D|) by preventing any silent writes.
    """

    UNOWNED = "unowned"  # Available for any agent to claim
    CLAIMED = "claimed"  # One agent holds exclusive write
    REVIEWING = "reviewing"  # One or more agents reading; no writes
    MERGED = "merged"  # Complete; immutable


class ArtifactContentType(StrEnum):
    """Supported content types for artifact sections."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    CODE = "code"


class SectionSchema(BaseModel):
    """Explicit section specification — the Specification Gap insight.

    Implicit specs cause 25–39 percentage-point coordination failure between
    agents (arXiv 2603.24284). Every field here is mandatory; there is no
    implicit "just write something".

    Args:
        section_id: Unique identifier for this section within the artifact.
        description: What this section must contain (be explicit).
        content_type: Expected data format for the content.
        dependencies: Other section_ids that must be completed first.
        required_fields: For JSON content type — keys that must be present.
        max_tokens: Approximate token budget (None = unlimited).
        validation_rules: Free-text constraints for LLM self-check.
    """

    section_id: str
    description: str
    content_type: ArtifactContentType
    dependencies: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    max_tokens: int | None = None
    validation_rules: list[str] = Field(default_factory=list)


class ArtifactSchema(BaseModel):
    """Shared specification for the entire artifact.

    Args:
        artifact_type: Category label ("document", "codebase", "schema", "report").
        description: Overall artifact goal.
        sections: Ordered list of section specifications.
        global_constraints: Cross-section consistency rules.
    """

    artifact_type: str
    description: str
    sections: list[SectionSchema]
    global_constraints: list[str] = Field(default_factory=list)

    def get_section(self, section_id: str) -> SectionSchema | None:
        """Return the SectionSchema for *section_id*, or None if not found."""
        return next((s for s in self.sections if s.section_id == section_id), None)

    def dependency_ids(self) -> dict[str, list[str]]:
        """Return {section_id: [dep_section_ids]} for topological sort."""
        return {s.section_id: s.dependencies for s in self.sections}


class ArtifactSection(BaseModel):
    """Mutable runtime state for a single section.

    Args:
        section_id: Matches SectionSchema.section_id.
        content: Written content (None until first write).
        schema_ref: Points to SectionSchema.section_id.
        version: Monotonic write counter (Token Coherence invariant).
        ownership_state: Current MESI-inspired ownership state.
        owner_agent_id: Agent holding CLAIMED state (None otherwise).
        last_modified: Timestamp of last write.
        modification_count: Total writes to this section.
    """

    section_id: str
    content: str | dict[str, Any] | list[Any] | None = None
    schema_ref: str
    version: int = 0
    ownership_state: SectionOwnershipState = SectionOwnershipState.UNOWNED
    owner_agent_id: str | None = None
    last_modified: datetime | None = None
    modification_count: int = 0


class Artifact(BaseModel):
    """Full artifact with schema and all section states.

    Args:
        artifact_id: Unique identifier.
        artifact_type: Matches ArtifactSchema.artifact_type.
        schema: Immutable schema (set at creation, never changed).
        sections: Mutable section states keyed by section_id.
        global_version: Increments on any section write.
        created_at: Creation timestamp.
        completed_at: Set when all sections reach MERGED state.
        metadata: Arbitrary key/value annotations.
    """

    artifact_id: str
    artifact_type: str
    artifact_schema: ArtifactSchema
    sections: dict[str, ArtifactSection]
    global_version: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EditOperation(BaseModel):
    """Describes a single edit operation on a section.

    Args:
        op_type: Operation category.
        artifact_id: Target artifact.
        section_id: Target section.
        agent_id: Agent performing the operation.
        content: New content (only for "write" ops).
        expected_version: If set, write fails on version mismatch.
        mark_complete: For "release" — True → MERGED, False → UNOWNED.
    """

    op_type: Literal["claim", "write", "release", "read"]
    artifact_id: str
    section_id: str
    agent_id: str
    content: str | dict[str, Any] | list[Any] | None = None
    expected_version: int | None = None
    mark_complete: bool = True


class ConflictType(StrEnum):
    """Category of conflict detected during a write operation."""

    SCHEMA_VALIDATION = "schema_validation"
    DEPENDENCY_VERSION = "dependency_version"
    OWNERSHIP = "ownership"
    VERSION_MISMATCH = "version_mismatch"


class SectionConflict(BaseModel):
    """Describes a write-time conflict on a section.

    Args:
        section_id: Section where the conflict occurred.
        conflict_type: Category of the conflict.
        description: Human-readable explanation.
        resolution: Recommended action for the caller.
    """

    section_id: str
    conflict_type: ConflictType
    description: str
    resolution: Literal["reject", "retry", "human_review"]


class ArtifactEditResult(BaseModel):
    """Result of an edit operation attempt.

    Args:
        success: True when the operation completed without conflict.
        op_type: Operation that was attempted.
        section_id: Section that was targeted.
        new_version: Section version after write (None on failure).
        conflict: Populated when success=False.
        agent_id: Agent that performed the operation.
        timestamp: When the result was produced.
    """

    success: bool
    op_type: str
    section_id: str
    new_version: int | None = None
    conflict: SectionConflict | None = None
    agent_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScopedContext(BaseModel):
    """Scoped context provided to an agent for one section (CAID, arXiv 2603.21489).

    Each agent receives only: artifact description + its section schema +
    one-line summaries of completed dependency sections. Full artifact history
    is never passed — this confines error propagation to the active section.

    Args:
        artifact_description: Overall goal of the artifact.
        section_schema: Full spec for the section this agent must write.
        completed_dependencies: {section_id: one-line summary} for deps that are MERGED.
        global_constraints: Cross-section consistency rules from ArtifactSchema.
        approach_hint: Optional hint from a planner (empty string = not provided).
    """

    artifact_description: str
    section_schema: SectionSchema
    completed_dependencies: dict[str, str]
    global_constraints: list[str]
    approach_hint: str = ""
