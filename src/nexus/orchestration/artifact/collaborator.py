"""ArtifactCollaborator — per-agent facade for artifact collaboration.

Implements the AWCP workspace delegation pattern (arXiv 2602.20493) and
CAID scoped context (arXiv 2603.21489): each agent receives only the artifact
description + its section schema + one-line summaries of completed deps.
"""

from __future__ import annotations

import json
from typing import Any

from nexus.core.logging import get_logger
from nexus.orchestration.artifact.conflict_detector import ConflictDetector
from nexus.orchestration.artifact.lock_manager import SectionLockManager
from nexus.orchestration.artifact.store import ArtifactStore
from nexus.orchestration.artifact.types import (
    ArtifactEditResult,
    ArtifactSection,
    EditOperation,
    ScopedContext,
    SectionOwnershipState,
)

_log = get_logger(__name__)

_SUMMARY_MAX_CHARS = 200


class ArtifactCollaborator:
    """Per-agent facade for the claim → read → write → release lifecycle.

    Each ArtifactCollaborator is bound to a specific agent_id. All operations
    are scoped to that agent. The facade wraps store + lock_manager +
    conflict_detector so agent code never touches those directly.

    Args:
        agent_id: Unique ID of the agent using this collaborator.
        store: ArtifactStore for persistence.
        lock_manager: SectionLockManager for TODO-claim protocol.
        conflict_detector: ConflictDetector for pre-write structural checks.
        model_client: Optional LLM client for generating summaries. When None,
            summaries are produced by truncating to _SUMMARY_MAX_CHARS.
    """

    def __init__(
        self,
        agent_id: str,
        store: ArtifactStore,
        lock_manager: SectionLockManager,
        conflict_detector: ConflictDetector,
        model_client: Any | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._store = store
        self._lock_manager = lock_manager
        self._conflict_detector = conflict_detector
        self._model_client = model_client

    # ------------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------------

    async def claim_section(
        self,
        artifact_id: str,
        section_id: str,
        timeout_secs: int = 300,
    ) -> bool:
        """Claim exclusive write on a section.

        Args:
            artifact_id: Target artifact.
            section_id: Section to claim.
            timeout_secs: Lock expiry in seconds.

        Returns:
            True when the claim succeeds.
        """
        result = await self._lock_manager.claim(
            artifact_id, section_id, self._agent_id, timeout_secs
        )
        _log.debug(
            "collaborator_claim",
            agent_id=self._agent_id,
            artifact_id=artifact_id,
            section_id=section_id,
            success=result,
        )
        return result

    async def read_section(
        self,
        artifact_id: str,
        section_id: str,
    ) -> ArtifactSection | None:
        """Read current content of a section.

        Args:
            artifact_id: Target artifact.
            section_id: Section to read.

        Returns:
            ArtifactSection, or None if not yet written (version=0).
        """
        try:
            section = await self._store.read_section(artifact_id, section_id, self._agent_id)
            return section if section.version > 0 else section
        except Exception:
            return None

    async def write_section(
        self,
        artifact_id: str,
        section_id: str,
        content: str | dict[str, Any] | list[Any],
        expected_version: int | None = None,
    ) -> ArtifactEditResult:
        """Write content to a claimed section.

        Runs ConflictDetector.check() before calling store.write_section().
        If ConflictDetector returns conflicts with resolution="reject", returns
        ArtifactEditResult(success=False) without writing.

        Args:
            artifact_id: Target artifact.
            section_id: Section to write (must be claimed by this agent).
            content: Content to persist.
            expected_version: If set, write fails on version mismatch.

        Returns:
            ArtifactEditResult indicating success or describing the conflict.
        """
        artifact = await self._store.load(artifact_id)
        schema = artifact.artifact_schema.get_section(section_id)
        if schema is not None:
            conflicts = await self._conflict_detector.check(
                artifact_id, section_id, schema, content
            )
            blocking = [c for c in conflicts if c.resolution == "reject"]
            if blocking:
                return ArtifactEditResult(
                    success=False,
                    op_type="write",
                    section_id=section_id,
                    agent_id=self._agent_id,
                    conflict=blocking[0],
                )

        op = EditOperation(
            op_type="write",
            artifact_id=artifact_id,
            section_id=section_id,
            agent_id=self._agent_id,
            content=content,
            expected_version=expected_version,
        )
        return await self._store.write_section(op)

    async def release_section(
        self,
        artifact_id: str,
        section_id: str,
        mark_complete: bool = True,
    ) -> None:
        """Release a section after writing.

        Args:
            artifact_id: Target artifact.
            section_id: Section to release.
            mark_complete: True → MERGED (done); False → UNOWNED (abandoned).
        """
        await self._lock_manager.release(
            artifact_id, section_id, self._agent_id, mark_complete=mark_complete
        )

    # ------------------------------------------------------------------
    # CAID scoped context (arXiv 2603.21489)
    # ------------------------------------------------------------------

    async def get_scoped_context(
        self,
        artifact_id: str,
        section_id: str,
    ) -> ScopedContext:
        """Build CAID scoped context for this agent's section.

        Returns only: artifact description + section schema + one-line summaries
        of MERGED dependency sections. Full artifact history is never included —
        this prevents error propagation from one section to another.

        Args:
            artifact_id: Target artifact.
            section_id: Section this agent will write.

        Returns:
            ScopedContext ready to inject into the agent's prompt.
        """
        artifact = await self._store.get_snapshot(artifact_id)
        section_schema = artifact.artifact_schema.get_section(section_id)
        if section_schema is None:
            from nexus.core.errors import ArtifactSectionNotFoundError

            raise ArtifactSectionNotFoundError(
                f"Section '{section_id}' not found in artifact '{artifact_id}'",
                code="SECTION_NOT_FOUND",
            )

        completed_deps: dict[str, str] = {}
        for dep_id in section_schema.dependencies:
            dep_section = artifact.sections.get(dep_id)
            if dep_section is None or dep_section.ownership_state != SectionOwnershipState.MERGED:
                continue
            if dep_section.content is not None:
                completed_deps[dep_id] = self._summarize_content(dep_section.content)

        return ScopedContext(
            artifact_description=artifact.artifact_schema.description,
            section_schema=section_schema,
            completed_dependencies=completed_deps,
            global_constraints=artifact.artifact_schema.global_constraints,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _summarize_content(self, content: Any) -> str:
        """Produce a one-line summary: first 200 chars or first JSON value."""
        raw = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        return raw[:_SUMMARY_MAX_CHARS]
