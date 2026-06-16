"""Pydantic models for the memory layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProcedureType(StrEnum):
    """Discriminator for Procedure records."""

    WORKFLOW = "workflow"
    SKILL = "skill"
    REFLECTION = "reflection"


class ProcedureStep(BaseModel):
    """A single step within a learned procedure."""

    action: str
    tool_name: str | None = None
    parameters_template: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str | None = None


class Procedure(BaseModel):
    """A learned, reusable workflow extracted from completed tool-call sequences."""

    id: str
    name: str
    description: str
    steps: list[ProcedureStep] = Field(default_factory=list)
    trigger_conditions: list[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_used: datetime | None = None
    agent_id: str
    embedding: list[float] | None = None
    procedure_type: ProcedureType = ProcedureType.WORKFLOW
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodicRecord(BaseModel):
    """A single episodic memory record persisted across sessions."""

    id: str
    agent_id: str
    user_id: str | None = None
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    trust_score: float = 1.0
    provenance: str | None = None
    embedding: list[float] | None = None
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: datetime | None = None

    @field_validator("trust_score", "importance_score")
    @classmethod
    def _validate_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {v}")
        return v


class SemanticFact(BaseModel):
    """A discrete, structured fact extracted from episodic memory.

    Stored as a subject-predicate-object triple with a confidence score.
    Deduplication is performed on (subject, predicate) when storing.
    """

    id: str
    subject: str
    predicate: str
    object_value: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_episode_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0
    embedding: list[float] | None = None


class RetrievedRecord(BaseModel):
    """An episodic record returned by the retriever, annotated with scores."""

    record: EpisodicRecord
    score: float
    recency_score: float
    similarity_score: float
    importance_score: float

    @field_validator("score", "recency_score", "similarity_score", "importance_score")
    @classmethod
    def _validate_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {v}")
        return v
