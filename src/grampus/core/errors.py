"""Nexus exception hierarchy — all exceptions carry a machine-readable code."""

from __future__ import annotations


class GrampusError(Exception):
    """Root exception for all Nexus errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, object] | None = None,
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = details or {}
        self.hint = hint


class ConfigError(GrampusError):
    """Raised when configuration is invalid or missing."""


class MemoryError(GrampusError):
    """Raised for memory subsystem failures."""


class MemorySecurityError(MemoryError):
    """Raised when a memory security violation is detected (e.g. poisoning)."""


class ToolError(GrampusError):
    """Raised when tool execution fails."""


class ToolTimeoutError(ToolError):
    """Raised when a tool exceeds its execution timeout."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool name is not registered."""


class ToolValidationError(ToolError):
    """Raised when tool arguments fail schema validation."""


class OrchestrationError(GrampusError):
    """Raised for orchestration / graph execution failures."""


class BudgetExceededError(OrchestrationError):
    """Raised when an agent exceeds its cost or token budget."""


class HandoffError(OrchestrationError):
    """Raised when an agent handoff is rejected or fails."""


class SafetyError(GrampusError):
    """Raised when a safety check blocks an action."""


class ModelError(GrampusError):
    """Raised when an LLM provider call fails."""


class DaprError(GrampusError):
    """Base for all Dapr integration errors."""


class DaprConnectionError(DaprError):
    """Raised when the Dapr sidecar is unreachable or times out."""


class ConcurrencyError(DaprError):
    """Raised when an ETag mismatch is detected (optimistic concurrency failure)."""


class LockAcquisitionError(DaprError):
    """Raised when a distributed lock cannot be acquired."""


class StateSerializationError(DaprError):
    """Raised when state store bytes cannot be deserialized into the expected model."""


class DaprJobsError(DaprError):
    """Raised when the Dapr Jobs API returns an unexpected response."""


class SnapshotError(GrampusError):
    """Raised when a snapshot export, import, or restore operation fails."""


class UncertaintyError(GrampusError):
    """Raised when agent uncertainty reaches an unrecoverable CRITICAL level."""


class PlanningError(GrampusError):
    """Raised for unrecoverable planning failures.

    Codes: CIRCULAR_DEPENDENCY, MAX_REPLANS_EXCEEDED, REPLAN_PARSE_FAILED,
    PLAN_PARSE_FAILED, NO_SUBGOALS.
    """


class MarketAllocationError(GrampusError):
    """Raised when market allocation fails (no capable bidders, all bids below threshold, etc.)."""


class ArtifactConflictError(GrampusError):
    """Raised when an artifact operation fails: ownership conflict, schema validation,
    version mismatch, or circular dependency in section DAG."""


class ArtifactSectionNotFoundError(GrampusError):
    """Raised when accessing a section_id not defined in the artifact's schema."""


class CausalError(GrampusError):
    """Raised when causal inference fails in an unrecoverable way."""


class RedTeamError(GrampusError):
    """Raised when a red-team campaign cannot be executed."""


class EmbeddingError(GrampusError):
    """Raised when an embedding API call fails."""


class VersioningError(GrampusError):
    """Raised when version management operations fail."""


class RAGError(GrampusError):
    """Raised when RAG pipeline operations fail (ingestion, retrieval, store setup)."""
