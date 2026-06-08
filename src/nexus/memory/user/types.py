"""Pydantic models for the three-tier user memory hierarchy (F2)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class UserFactCategory(StrEnum):
    """Category discriminator for extracted user facts."""

    EXPERTISE = "expertise"
    PREFERENCE = "preference"
    DECISION = "decision"
    CONTEXT = "context"
    CONSTRAINT = "constraint"


class UserFact(BaseModel):
    """A single temporally-grounded fact about a user (Tier 2).

    Temporal grounding (Beyond Dialogue Time, arXiv 2601.07468):
    valid_until=None means still true. Set valid_until=now to deprecate.
    """

    id: str
    user_id: str
    content: str
    category: UserFactCategory
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    source_session_id: str
    access_count: int = 0
    last_accessed: datetime | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True if this fact has not expired."""
        if self.valid_until is None:
            return True
        return datetime.now(UTC) < self.valid_until


class UserProfile(BaseModel):
    """Synthesized persona — one record per user (Tier 1).

    Synthesized by ProfileSynthesizer from all valid UserFacts.
    Replaces the previous version entirely on each re-synthesis.
    """

    user_id: str
    expertise_level: int = Field(default=3, ge=1, le=5)
    expertise_domains: list[str] = Field(default_factory=list)
    communication_style: str = "balanced"
    preferred_depth: str = "balanced"
    active_goals: list[str] = Field(default_factory=list)
    past_key_decisions: list[str] = Field(default_factory=list)
    active_constraints: list[str] = Field(default_factory=list)
    last_synthesized: datetime = Field(default_factory=lambda: datetime.now(UTC))
    synthesis_fact_count: int = 0
    version: int = 0


class UserMemoryContext(BaseModel):
    """Context returned by UserMemoryAdapter.get_context() for injection."""

    user_id: str
    profile: UserProfile | None = None
    relevant_facts: list[UserFact] = Field(default_factory=list)
    formatted_context: str = ""


class FactExtractionResult(BaseModel):
    """Returned by FactExtractor.extract_from_session()."""

    user_id: str
    session_id: str
    facts_extracted: int
    facts_updated: int
    facts_expired: int
    new_fact_ids: list[str] = Field(default_factory=list)


class ProfileSynthesisResult(BaseModel):
    """Returned by ProfileSynthesizer.synthesize()."""

    user_id: str
    triggered: bool
    previous_version: int | None = None
    new_version: int | None = None
    facts_used: int = 0


class _FactIndex(BaseModel):
    """Internal Pydantic wrapper for the per-user fact ID index."""

    ids: list[str] = Field(default_factory=list)
