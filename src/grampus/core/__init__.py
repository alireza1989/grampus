"""Core layer: config, errors, logging, base types, and model clients."""

from grampus.core.config import (
    DaprConfig,
    GrampusConfig,
    MemoryConfig,
    ModelConfig,
    ObservabilityConfig,
    SafetyConfig,
)
from grampus.core.errors import (
    BudgetExceededError,
    ConcurrencyError,
    ConfigError,
    DaprConnectionError,
    DaprError,
    GrampusError,
    LockAcquisitionError,
    MemoryError,
    MemorySecurityError,
    ModelError,
    OrchestrationError,
    SafetyError,
    StateSerializationError,
    ToolError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from grampus.core.logging import bind_correlation_id, configure_logging, get_logger
from grampus.core.models.anthropic import AnthropicClient
from grampus.core.models.base import ModelClient, ModelResponse
from grampus.core.models.openai import OpenAIClient
from grampus.core.types import (
    AgentDefinition,
    AgentState,
    AgentStatus,
    ExecutionResult,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

__all__ = [
    # config
    "GrampusConfig",
    "ModelConfig",
    "MemoryConfig",
    "SafetyConfig",
    "DaprConfig",
    "ObservabilityConfig",
    # errors
    "GrampusError",
    "ConfigError",
    "MemoryError",
    "MemorySecurityError",
    "ToolError",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "ToolValidationError",
    "OrchestrationError",
    "BudgetExceededError",
    "SafetyError",
    "ModelError",
    "DaprError",
    "DaprConnectionError",
    "ConcurrencyError",
    "LockAcquisitionError",
    "StateSerializationError",
    # logging
    "get_logger",
    "configure_logging",
    "bind_correlation_id",
    # models
    "ModelClient",
    "ModelResponse",
    "AnthropicClient",
    "OpenAIClient",
    # types
    "Role",
    "AgentStatus",
    "ToolCall",
    "ToolResult",
    "Message",
    "ToolParameter",
    "ToolDefinition",
    "AgentDefinition",
    "TokenUsage",
    "AgentState",
    "ExecutionResult",
]
