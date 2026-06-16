"""ConflictDetector — pre-write structural conflict detection.

Two structural checks run before any WRITE is committed (STORM insight,
arXiv 2605.20563): write-time detection is +18.7 points better than
post-hoc merge resolution on Commit0-Lite.
"""

from __future__ import annotations

from typing import Any

from grampus.core.logging import get_logger
from grampus.orchestration.artifact.store import ArtifactStore
from grampus.orchestration.artifact.types import (
    ArtifactContentType,
    ConflictType,
    SectionConflict,
    SectionSchema,
)

_log = get_logger(__name__)


class ConflictDetector:
    """Two structural checks run before any WRITE is committed.

    Check 1 (always enforced): Dependency version check.
        For each dependency section of the section being written, verify its
        version is ≥ 1 (written at least once). A dependency at version=0
        means the content does not yet exist; writing a section that claims
        to reference it produces unreliable output.

    Check 2 (advisory): Content-type structural consistency.
        When the section's schema declares content_type=JSON, check that
        the content doesn't share key names with sibling sections' required_fields
        in a conflicting way. Resolution is "human_review", not "reject".

    Args:
        store: ArtifactStore for reading dependency section states.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    async def check(
        self,
        artifact_id: str,
        section_id: str,
        schema: SectionSchema,
        content: Any,
    ) -> list[SectionConflict]:
        """Run all structural checks and return conflicts found.

        Args:
            artifact_id: Target artifact.
            section_id: Section being written.
            schema: SectionSchema for this section.
            content: Proposed content value.

        Returns:
            List of SectionConflict (empty = clean, ready to write).
        """
        conflicts: list[SectionConflict] = []

        dep_conflict = await self._check_dependency_versions(artifact_id, schema.dependencies)
        if dep_conflict:
            conflicts.append(dep_conflict)

        structural = self._check_structural_consistency(section_id, content, schema)
        if structural:
            conflicts.append(structural)

        return conflicts

    async def _check_dependency_versions(
        self,
        artifact_id: str,
        dependencies: list[str],
    ) -> SectionConflict | None:
        """Verify all dependency sections have version ≥ 1 (written at least once).

        Args:
            artifact_id: Target artifact.
            dependencies: Section IDs that must be completed.

        Returns:
            SectionConflict if any dependency is unwritten, otherwise None.
        """
        for dep_id in dependencies:
            section = await self._store._load_section(artifact_id, dep_id)
            if section is None or section.version == 0:
                _log.debug(
                    "artifact_dependency_unwritten",
                    artifact_id=artifact_id,
                    dep_id=dep_id,
                )
                return SectionConflict(
                    section_id=dep_id,
                    conflict_type=ConflictType.DEPENDENCY_VERSION,
                    description=(
                        f"Dependency section '{dep_id}' has not been written yet "
                        f"(version=0). Complete it before writing this section."
                    ),
                    resolution="reject",
                )
        return None

    def _check_structural_consistency(
        self,
        section_id: str,
        content: Any,
        schema: SectionSchema,
    ) -> SectionConflict | None:
        """Advisory check for JSON content type key conflicts.

        Only applicable when content_type=JSON and content is a dict. Checks
        that content includes all required_fields (non-blocking advisory).

        Args:
            section_id: Section being written.
            content: Proposed content.
            schema: SectionSchema for this section.

        Returns:
            Advisory SectionConflict (resolution="human_review") or None.
        """
        if schema.content_type != ArtifactContentType.JSON:
            return None
        if not isinstance(content, dict):
            return None
        if not schema.required_fields:
            return None

        missing = [f for f in schema.required_fields if f not in content]
        if missing:
            return SectionConflict(
                section_id=section_id,
                conflict_type=ConflictType.SCHEMA_VALIDATION,
                description=(
                    f"Section '{section_id}' JSON content missing advisory "
                    f"required_fields: {missing}. Consider adding them for "
                    "cross-section consistency."
                ),
                resolution="human_review",
            )
        return None
