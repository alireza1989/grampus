"""Nexus exception hierarchy — all exceptions carry a machine-readable code."""

from __future__ import annotations


class NexusError(Exception):
    """Root exception for all Nexus errors."""

    def __init__(
        self, message: str, *, code: str, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = details or {}


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


class OrchestrationError(NexusError):
    """Raised for orchestration / graph execution failures."""


class BudgetExceededError(OrchestrationError):
    """Raised when an agent exceeds its cost or token budget."""


class SafetyError(NexusError):
    """Raised when a safety check blocks an action."""


class ModelError(NexusError):
    """Raised when an LLM provider call fails."""
