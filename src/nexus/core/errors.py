"""Nexus exception hierarchy — all exceptions carry a machine-readable code."""

from __future__ import annotations


class NexusError(Exception):
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


class ConfigError(NexusError):
    """Raised when configuration is invalid or missing."""


class MemoryError(NexusError):
    """Raised for memory subsystem failures."""


class MemorySecurityError(MemoryError):
    """Raised when a memory security violation is detected (e.g. poisoning)."""


class ToolError(NexusError):
    """Raised when tool execution fails."""


class ToolTimeoutError(ToolError):
    """Raised when a tool exceeds its execution timeout."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool name is not registered."""


class ToolValidationError(ToolError):
    """Raised when tool arguments fail schema validation."""


class OrchestrationError(NexusError):
    """Raised for orchestration / graph execution failures."""


class BudgetExceededError(OrchestrationError):
    """Raised when an agent exceeds its cost or token budget."""


class HandoffError(OrchestrationError):
    """Raised when an agent handoff is rejected or fails."""


class SafetyError(NexusError):
    """Raised when a safety check blocks an action."""


class ModelError(NexusError):
    """Raised when an LLM provider call fails."""


class DaprError(NexusError):
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


class SnapshotError(NexusError):
    """Raised when a snapshot export, import, or restore operation fails."""


class UncertaintyError(NexusError):
    """Raised when agent uncertainty reaches an unrecoverable CRITICAL level."""
