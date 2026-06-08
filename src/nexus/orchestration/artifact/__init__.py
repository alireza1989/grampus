"""Artifact-centric collaboration for multi-agent structured document creation (E36)."""

from nexus.orchestration.artifact.collaborator import ArtifactCollaborator
from nexus.orchestration.artifact.conflict_detector import ConflictDetector
from nexus.orchestration.artifact.crew import ArtifactCrew
from nexus.orchestration.artifact.lock_manager import SectionLockManager
from nexus.orchestration.artifact.schema import SchemaValidator
from nexus.orchestration.artifact.store import ArtifactStore
from nexus.orchestration.artifact.types import (
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
