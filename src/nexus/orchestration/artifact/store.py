"""ArtifactStore — Dapr-backed, versioned, MESI-ownership-enforced persistence."""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from nexus.core.errors import ArtifactConflictError, ArtifactSectionNotFoundError
from nexus.core.logging import get_logger
from nexus.orchestration.artifact.schema import SchemaValidator
from nexus.orchestration.artifact.types import (
    Artifact,
    ArtifactEditResult,
    ArtifactSchema,
    ArtifactSection,
    ConflictType,
    EditOperation,
    SectionConflict,
    SectionOwnershipState,
)

if TYPE_CHECKING:
    from nexus.dapr.state import DaprStateStore

_log = get_logger(__name__)

_ENTITY_ARTIFACT = "artifact"
_ENTITY_SECTION = "artifact_section"


class ArtifactStore:
    """Dapr-backed storage for artifacts with MESI ownership and monotonic versioning.

    Dapr key namespace:
    - Artifact:  namespace:artifact:{artifact_id}
    - Section:   namespace:artifact_section:{artifact_id}:{section_id}

    Every write_section call enforces in order:
    1. Agent is current CLAIMED owner
    2. expected_version matches (if provided)
    3. SchemaValidator passes
    Then increments section.version + artifact.global_version and persists.

    Args:
        state_store: Dapr state store (namespace already embedded in store).
        validator: SchemaValidator for write-time content checks.
        tracer: Optional NexusTracer for OTEL spans.
    """

    def __init__(
        self,
        state_store: DaprStateStore,
        validator: SchemaValidator,
        tracer: Any | None = None,
    ) -> None:
        self._store = state_store
        self._validator = validator
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Creation & loading
    # ------------------------------------------------------------------

    async def create(
        self,
        schema: ArtifactSchema,
        artifact_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Create a new Artifact from a schema.

        Initialises all sections as UNOWNED with version=0 and content=None.
        Generates artifact_id via uuid4 if not provided.

        Args:
            schema: Specification for the new artifact.
            artifact_id: Explicit ID, or None to auto-generate.
            metadata: Optional key/value annotations on the artifact.

        Returns:
            The persisted Artifact.
        """
        aid = artifact_id or str(uuid.uuid4())
        sections: dict[str, ArtifactSection] = {
            s.section_id: ArtifactSection(
                section_id=s.section_id,
                schema_ref=s.section_id,
                content=None,
                version=0,
                ownership_state=SectionOwnershipState.UNOWNED,
            )
            for s in schema.sections
        }
        artifact = Artifact(
            artifact_id=aid,
            artifact_type=schema.artifact_type,
            artifact_schema=schema,
            sections=sections,
            global_version=0,
            metadata=metadata or {},
        )
        await self._save_artifact(artifact)
        for section in sections.values():
            await self._save_section(aid, section)
        _log.debug("artifact_created", artifact_id=aid, sections=len(sections))
        return artifact

    async def load(self, artifact_id: str) -> Artifact:
        """Load artifact metadata from Dapr.

        Args:
            artifact_id: ID of the artifact to load.

        Returns:
            The Artifact (sections populated from Dapr).

        Raises:
            ArtifactSectionNotFoundError: If artifact_id not found.
        """
        artifact, _ = await self._store.get(_ENTITY_ARTIFACT, artifact_id, Artifact)
        if artifact is None:
            raise ArtifactSectionNotFoundError(
                f"Artifact '{artifact_id}' not found",
                code="ARTIFACT_NOT_FOUND",
            )
        return artifact

    # ------------------------------------------------------------------
    # Write / read operations
    # ------------------------------------------------------------------

    async def write_section(self, op: EditOperation) -> ArtifactEditResult:
        """Atomically write content to a claimed section.

        Validation order (STORM write-time detection, arXiv 2605.20563):
        1. Load current section state
        2. Ownership check: op.agent_id must be current owner (state == CLAIMED)
        3. Version check: expected_version must match if provided
        4. Schema validation via SchemaValidator
        5. Increment version, update content, persist

        Returns ArtifactEditResult with success=True or conflict populated.
        Never raises on conflict — caller handles the result.

        Args:
            op: EditOperation with op_type="write" and content set.

        Returns:
            ArtifactEditResult indicating success or describing the conflict.
        """
        section = await self._load_section(op.artifact_id, op.section_id)
        if section is None:
            return ArtifactEditResult(
                success=False,
                op_type=op.op_type,
                section_id=op.section_id,
                agent_id=op.agent_id,
                conflict=SectionConflict(
                    section_id=op.section_id,
                    conflict_type=ConflictType.OWNERSHIP,
                    description=f"Section '{op.section_id}' not found in artifact",
                    resolution="reject",
                ),
            )

        ownership_conflict = self._check_ownership(section, op.agent_id, op.section_id)
        if ownership_conflict:
            _log.warning(
                "artifact_write_ownership_conflict",
                artifact_id=op.artifact_id,
                section_id=op.section_id,
                agent_id=op.agent_id,
            )
            if self._tracer:
                with (
                    contextlib.suppress(Exception),
                    self._tracer._tracer.start_as_current_span("artifact.write_section") as span,
                ):
                    span.set_attribute("artifact_id", op.artifact_id)
                    span.set_attribute("section_id", op.section_id)
                    span.set_attribute("agent_id", op.agent_id)
                    span.set_attribute("conflict_type", ConflictType.OWNERSHIP.value)
            return ArtifactEditResult(
                success=False,
                op_type=op.op_type,
                section_id=op.section_id,
                agent_id=op.agent_id,
                conflict=ownership_conflict,
            )

        if op.expected_version is not None and section.version != op.expected_version:
            conflict = SectionConflict(
                section_id=op.section_id,
                conflict_type=ConflictType.VERSION_MISMATCH,
                description=(
                    f"Section '{op.section_id}' version mismatch: "
                    f"expected {op.expected_version}, actual {section.version}"
                ),
                resolution="retry",
            )
            return ArtifactEditResult(
                success=False,
                op_type=op.op_type,
                section_id=op.section_id,
                agent_id=op.agent_id,
                conflict=conflict,
            )

        section_schema = None
        artifact = await self.load(op.artifact_id)
        section_schema = artifact.artifact_schema.get_section(op.section_id)
        if section_schema:
            candidate = section.model_copy(update={"content": op.content})
            schema_conflict = self._validator.validate(candidate, section_schema)
            if schema_conflict:
                return ArtifactEditResult(
                    success=False,
                    op_type=op.op_type,
                    section_id=op.section_id,
                    agent_id=op.agent_id,
                    conflict=schema_conflict,
                )

        section.content = op.content
        section.version += 1
        section.last_modified = datetime.now(UTC)
        section.modification_count += 1
        await self._save_section(op.artifact_id, section)

        artifact.global_version += 1
        await self._save_artifact(artifact)

        new_version = section.version
        _log.debug(
            "artifact_section_written",
            artifact_id=op.artifact_id,
            section_id=op.section_id,
            agent_id=op.agent_id,
            new_version=new_version,
        )

        if self._tracer:
            with (
                contextlib.suppress(Exception),
                self._tracer._tracer.start_as_current_span("artifact.write_section") as span,
            ):
                span.set_attribute("artifact_id", op.artifact_id)
                span.set_attribute("section_id", op.section_id)
                span.set_attribute("agent_id", op.agent_id)
                span.set_attribute("new_version", new_version)

        return ArtifactEditResult(
            success=True,
            op_type=op.op_type,
            section_id=op.section_id,
            new_version=new_version,
            agent_id=op.agent_id,
        )

    async def read_section(
        self,
        artifact_id: str,
        section_id: str,
        agent_id: str,
    ) -> ArtifactSection:
        """Read a section, transitioning UNOWNED → REVIEWING for shared reads.

        CLAIMED sections can be read by the owner; others can read UNOWNED/REVIEWING.

        Args:
            artifact_id: Target artifact.
            section_id: Target section.
            agent_id: Reading agent's ID.

        Returns:
            Current ArtifactSection state.

        Raises:
            ArtifactSectionNotFoundError: If section_id not in artifact schema.
        """
        artifact = await self.load(artifact_id)
        if artifact.artifact_schema.get_section(section_id) is None:
            raise ArtifactSectionNotFoundError(
                f"Section '{section_id}' not found in artifact '{artifact_id}'",
                code="SECTION_NOT_FOUND",
            )

        section = await self._load_section(artifact_id, section_id)
        if section is None:
            raise ArtifactSectionNotFoundError(
                f"Section '{section_id}' state not found for artifact '{artifact_id}'",
                code="SECTION_STATE_NOT_FOUND",
            )

        if section.ownership_state == SectionOwnershipState.UNOWNED:
            section.ownership_state = SectionOwnershipState.REVIEWING
            await self._save_section(artifact_id, section)

        if self._tracer:
            with (
                contextlib.suppress(Exception),
                self._tracer._tracer.start_as_current_span("artifact.read_section") as span,
            ):
                span.set_attribute("artifact_id", artifact_id)
                span.set_attribute("section_id", section_id)
                span.set_attribute("agent_id", agent_id)

        return section

    async def update_section_state(
        self,
        artifact_id: str,
        section_id: str,
        new_state: SectionOwnershipState,
        agent_id: str | None = None,
    ) -> None:
        """Set ownership state directly (used by SectionLockManager).

        Args:
            artifact_id: Target artifact.
            section_id: Target section.
            new_state: New ownership state to apply.
            agent_id: New owner (set for CLAIMED, None otherwise).
        """
        section = await self._load_section(artifact_id, section_id)
        if section is None:
            return
        section.ownership_state = new_state
        section.owner_agent_id = agent_id
        await self._save_section(artifact_id, section)

    async def get_snapshot(self, artifact_id: str) -> Artifact:
        """Load full artifact with all sections populated.

        Args:
            artifact_id: Target artifact.

        Returns:
            Artifact with freshly-loaded section states.
        """
        artifact = await self.load(artifact_id)
        for section_id in list(artifact.sections.keys()):
            section = await self._load_section(artifact_id, section_id)
            if section is not None:
                artifact.sections[section_id] = section
        return artifact

    async def complete_artifact(self, artifact_id: str) -> Artifact:
        """Mark artifact as complete.

        Validates all sections are in MERGED state and sets completed_at.

        Args:
            artifact_id: Target artifact.

        Returns:
            The completed Artifact.

        Raises:
            ArtifactConflictError: If any section is not MERGED.
        """
        artifact = await self.get_snapshot(artifact_id)
        non_merged = [
            sid
            for sid, sec in artifact.sections.items()
            if sec.ownership_state != SectionOwnershipState.MERGED
        ]
        if non_merged:
            raise ArtifactConflictError(
                f"Cannot complete artifact '{artifact_id}': sections not MERGED: {non_merged}",
                code="SECTIONS_NOT_MERGED",
                details={"non_merged_sections": non_merged},
            )
        artifact.completed_at = datetime.now(UTC)
        await self._save_artifact(artifact)
        _log.debug("artifact_completed", artifact_id=artifact_id)
        return artifact

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_section_key(self, artifact_id: str, section_id: str) -> str:
        return f"{artifact_id}:{section_id}"

    def _build_artifact_key(self, artifact_id: str) -> str:
        return artifact_id

    def _check_ownership(
        self,
        section: ArtifactSection,
        agent_id: str,
        section_id: str,
    ) -> SectionConflict | None:
        if section.ownership_state != SectionOwnershipState.CLAIMED:
            return SectionConflict(
                section_id=section_id,
                conflict_type=ConflictType.OWNERSHIP,
                description=(
                    f"Section '{section_id}' is not CLAIMED "
                    f"(current state: {section.ownership_state.value}). "
                    "Agent must claim the section before writing."
                ),
                resolution="reject",
            )
        if section.owner_agent_id != agent_id:
            return SectionConflict(
                section_id=section_id,
                conflict_type=ConflictType.OWNERSHIP,
                description=(
                    f"Section '{section_id}' is CLAIMED by '{section.owner_agent_id}', "
                    f"not '{agent_id}'"
                ),
                resolution="reject",
            )
        return None

    async def _save_artifact(self, artifact: Artifact) -> None:
        await self._store.save(
            _ENTITY_ARTIFACT, self._build_artifact_key(artifact.artifact_id), artifact
        )

    async def _save_section(self, artifact_id: str, section: ArtifactSection) -> None:
        await self._store.save(
            _ENTITY_SECTION, self._build_section_key(artifact_id, section.section_id), section
        )

    async def _load_section(self, artifact_id: str, section_id: str) -> ArtifactSection | None:
        section, _ = await self._store.get(
            _ENTITY_SECTION,
            self._build_section_key(artifact_id, section_id),
            ArtifactSection,
        )
        return section
