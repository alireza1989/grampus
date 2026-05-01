"""Core layer: config, errors, logging, base types, and model clients."""

from nexus.core.config import (
    DaprConfig,
    MemoryConfig,
    ModelConfig,
    NexusConfig,
    ObservabilityConfig,
    SafetyConfig,
)
from nexus.core.errors import (
    BudgetExceededError,
    ConfigError,
    MemoryError,
    MemorySecurityError,
    ModelError,
    NexusError,
    OrchestrationError,
    SafetyError,
    ToolError,
    ToolTimeoutError,
)
from nexus.core.logging import bind_correlation_id, configure_logging, get_logger
from nexus.core.models.anthropic import AnthropicClient
from nexus.core.models.base import ModelClient, ModelResponse
from nexus.core.models.openai import OpenAIClient
from nexus.core.types import (
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
    "NexusConfig",
    "ModelConfig",
    "MemoryConfig",
    "SafetyConfig",
    "DaprConfig",
    "ObservabilityConfig",
    # errors
    "NexusError",
    "ConfigError",
    "MemoryError",
    "MemorySecurityError",
    "ToolError",
    "ToolTimeoutError",
    "OrchestrationError",
    "BudgetExceededError",
    "SafetyError",
    "ModelError",
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
