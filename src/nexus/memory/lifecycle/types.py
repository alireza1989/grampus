"""Pydantic models for the memory lifecycle tier system (F3)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryTier(StrEnum):
    """Storage tier for a memory record."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class MemoryType(StrEnum):
    """Type of memory record."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class TierRecord(BaseModel):
    """Lifecycle metadata for a single memory record."""

    record_id: str
    memory_type: MemoryType
    agent_id: str
    current_tier: MemoryTier = MemoryTier.COLD
    access_count_total: int = 0
    access_count_7d: int = 0
    last_accessed: datetime | None = None
    promoted_at: datetime | None = None
    demoted_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LifecycleStats(BaseModel):
    """Summary of tier distribution for an agent."""

    agent_id: str
    hot_count: int = 0
    warm_count: int = 0
    cold_count: int = 0
    total_promotions: int = 0
    total_demotions: int = 0
    last_run: datetime | None = None


class QueryClassification(StrEnum):
    """Classification of a memory query for routing."""

    GRAPH = "graph"
    FLAT = "flat"
    SEQUENTIAL = "sequential"
