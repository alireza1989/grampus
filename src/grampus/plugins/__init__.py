"""Nexus plugin system — lifecycle hook infrastructure for third-party extensions."""

from grampus.plugins.base import GrampusPlugin
from grampus.plugins.loader import create_manager_from_entry_points, load_entry_point_plugins
from grampus.plugins.manager import PluginManager
from grampus.plugins.types import (
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
    "GrampusPlugin",
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
