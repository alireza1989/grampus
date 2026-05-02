"""Memory provenance: source tracking and content-hash verification."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """Trust tiers for memory write sources, ordered highest → lowest."""

    SYSTEM = "system"
    USER_INPUT = "user_input"
    LLM_GENERATED = "llm_generated"
    TOOL_RESULT = "tool_result"
    EXTERNAL_DATA = "external_data"

    @property
    def trust_level(self) -> float:
        """Baseline trust score for this source type."""
        _levels: dict[str, float] = {
            "system": 1.0,
            "user_input": 0.9,
            "llm_generated": 0.7,
            "tool_result": 0.6,
            "external_data": 0.3,
        }
        return _levels[self.value]


class Provenance(BaseModel):
    """Immutable provenance record attached to every memory write.

    Args:
        id: Unique provenance record identifier.
        source_type: Category of the entity that produced the content.
        source_id: Identifier of the specific source (agent ID, user ID, etc.).
        trust_level: Baseline trust score inherited from source_type at creation time.
        timestamp: UTC wall-clock time of the write.
        content_hash_sha256: SHA-256 hex digest of the raw content bytes.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: SourceType
    source_id: str
    trust_level: float
    timestamp: datetime
    content_hash_sha256: str


class ProvenanceTracker:
    """Creates and verifies :class:`Provenance` records for memory writes.

    Stateless — safe to share across components.
    """

    def create(
        self,
        content: str,
        source_type: SourceType,
        *,
        source_id: str,
    ) -> Provenance:
        """Build a provenance record for *content*.

        Args:
            content: Raw text that will be stored in memory.
            source_type: Category of the producing entity.
            source_id: Identifier of the specific source.

        Returns:
            A :class:`Provenance` with a SHA-256 content hash and UTC timestamp.
        """
        return Provenance(
            id=str(uuid.uuid4()),
            source_type=source_type,
            source_id=source_id,
            trust_level=source_type.trust_level,
            timestamp=datetime.now(UTC),
            content_hash_sha256=hashlib.sha256(content.encode()).hexdigest(),
        )

    def verify(self, content: str, provenance: Provenance) -> bool:
        """Return True if *content* matches the hash stored in *provenance*.

        Args:
            content: Current content of the memory record.
            provenance: The provenance record to verify against.

        Returns:
            True when hashes match (content is unmodified), False otherwise.
        """
        expected = hashlib.sha256(content.encode()).hexdigest()
        return expected == provenance.content_hash_sha256
