"""Pydantic models for the graph-structured memory layer (F3)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ConceptNode(BaseModel):
    """A concept in the topic-associative-network (stable, persistent)."""

    node_id: str
    label: str
    description: str
    embedding: list[float] | None = None
    frequency: int = 1
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_episode_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationshipEdge(BaseModel):
    """A directed relationship between two ConceptNodes."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    weight: float = Field(default=1.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryGraph(BaseModel):
    """The topic-associative-network for one agent (persistent)."""

    graph_id: str
    nodes: dict[str, ConceptNode] = Field(default_factory=dict)
    edges: list[RelationshipEdge] = Field(default_factory=list)
    version: int = 0
    last_consolidated: datetime | None = None


class EventNode(BaseModel):
    """A single event in the session-scoped event-progression-graph (transient)."""

    event_id: str
    event_type: str
    content_summary: str
    embedding: list[float] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    agent_id: str


class EventGraph(BaseModel):
    """Session-scoped event-progression-graph (transient — not persisted to Dapr)."""

    session_id: str
    agent_id: str
    events: list[EventNode] = Field(default_factory=list)
    last_embedding: list[float] | None = None


class SemanticShiftEvent(BaseModel):
    """Emitted when a semantic shift is detected during a session."""

    session_id: str
    agent_id: str
    shift_distance: float
    trigger_event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GraphQueryResult(BaseModel):
    """Result from GraphRetriever.query()."""

    nodes: list[ConceptNode] = Field(default_factory=list)
    query: str
    traversal_depth: int = 0
