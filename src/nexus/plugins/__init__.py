"""Nexus plugin system — lifecycle hook infrastructure for third-party extensions."""

from nexus.plugins.base import NexusPlugin
from nexus.plugins.loader import create_manager_from_entry_points, load_entry_point_plugins
from nexus.plugins.manager import PluginManager
from nexus.plugins.types import (
    AgentEndContext,
    AgentStartContext,
    ErrorContext,
    HookBlockedError,
    LLMCallContext,
    LLMCallModification,
    MemoryWriteContext,
    MemoryWriteModification,
    ToolCallContext,
    ToolCallModification,
    ToolResultContext,
)

__all__ = [
    "NexusPlugin",
    "PluginManager",
    "HookBlockedError",
    "AgentStartContext",
    "AgentEndContext",
    "LLMCallContext",
    "ToolCallContext",
    "ToolResultContext",
    "MemoryWriteContext",
    "ErrorContext",
    "LLMCallModification",
    "ToolCallModification",
    "MemoryWriteModification",
    "load_entry_point_plugins",
    "create_manager_from_entry_points",
]
