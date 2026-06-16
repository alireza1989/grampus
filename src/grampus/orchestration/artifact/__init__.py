"""Artifact-centric collaboration for multi-agent structured document creation (E36)."""

from grampus.orchestration.artifact.collaborator import ArtifactCollaborator
from grampus.orchestration.artifact.conflict_detector import ConflictDetector
from grampus.orchestration.artifact.crew import ArtifactCrew
from grampus.orchestration.artifact.lock_manager import SectionLockManager
from grampus.orchestration.artifact.schema import SchemaValidator
from grampus.orchestration.artifact.store import ArtifactStore
from grampus.orchestration.artifact.types import (
    Artifact,
    ArtifactContentType,
    ArtifactEditResult,
    ArtifactSchema,
    ArtifactSection,
    ConflictType,
    EditOperation,
    ScopedContext,
    SectionConflict,
    SectionOwnershipState,
    SectionSchema,
)

__all__ = [
    "ArtifactCollaborator",
    "ArtifactContentType",
    "ArtifactCrew",
    "ArtifactEditResult",
    "ArtifactSchema",
    "ArtifactSection",
    "Artifact",
    "ConflictDetector",
    "ConflictType",
    "EditOperation",
    "ScopedContext",
    "SectionConflict",
    "SectionLockManager",
    "SectionOwnershipState",
    "SectionSchema",
    "SchemaValidator",
    "ArtifactStore",
]
