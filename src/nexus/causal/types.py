"""Pydantic models for the causal analysis layer (F4). No logic."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EdgeType(StrEnum):
    """Three structural edge types plus LLM-extracted causal claims."""

    SEQUENTIAL = "sequential"
    DATA_DEPENDENCY = "data_dep"
    FAILURE_CASCADE = "failure"
    CAUSAL_CLAIM = "causal_claim"


class CausalEdge(BaseModel):
    """A directed causal edge in an execution or world-model graph."""

    edge_id: str
    source_event_id: str
    target_event_id: str
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0)
    evidence: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CausalGraph(BaseModel):
    """Causal graph reconstructed from a single session's execution log."""

    graph_id: str
    agent_id: str
    nodes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    edges: list[CausalEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RootCauseCandidate(BaseModel):
    """A ranked candidate for root cause of a failure."""

    event_id: str
    event_type: str
    description: str
    structural_score: float = Field(ge=0.0, le=1.0)
    positional_score: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    causal_chain: list[str] = Field(default_factory=list)


class CausalDiagnosis(BaseModel):
    """Full diagnosis result for a failed session."""

    session_id: str
    agent_id: str
    failure_event_id: str
    root_causes: list[RootCauseCandidate] = Field(default_factory=list)
    causal_graph: CausalGraph
    diagnosed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CausalRelation(BaseModel):
    """A single causal claim extracted from agent reasoning text."""

    relation_id: str
    cause_description: str
    effect_description: str
    relation_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str
    session_id: str
    agent_id: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorldModelGraph(BaseModel):
    """Persistent SCM for one agent — adjacency-list representation.

    Variables are named causal entities (e.g. ``tool_web_search``,
    ``state_user_query_answered``). Adjacency maps cause → list of effects.
    """

    agent_id: str
    variables: dict[str, str] = Field(default_factory=dict)
    adjacency: dict[str, list[str]] = Field(default_factory=dict)
    relations: list[CausalRelation] = Field(default_factory=list)
    version: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InterventionQuery(BaseModel):
    """A do-calculus intervention query: P(outcome | do(target = value))."""

    natural_language: str
    target_variable: str
    outcome_variable: str
    intervention_value: str


class InterventionResult(BaseModel):
    """Result of a do-calculus intervention query."""

    query: InterventionQuery
    answer: str
    causal_path: list[str]
    backdoor_vars: list[str] = Field(default_factory=list)
    backdoor_adjustment_applied: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    is_identifiable: bool = True
