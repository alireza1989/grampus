"""Base class for Nexus lifecycle hook plugins."""

from __future__ import annotations

from typing import Any

from nexus.plugins.types import (
    AgentEndContext,
    AgentStartContext,
    ErrorContext,
    LLMCallContext,
    LLMCallModification,
    MemoryWriteContext,
    MemoryWriteModification,
    ToolCallContext,
    ToolCallModification,
    ToolResultContext,
)


class NexusPlugin:
    """Base class for Nexus lifecycle hook plugins.

    Override only the hooks your plugin needs. All methods default to no-ops.

    Pre-hooks (pre_llm_call, pre_tool_call, pre_memory_write) return None to
    pass the input through unchanged, or a modified value to replace it. Raise
    HookBlockedError to cancel the operation entirely.

    Observational hooks (all others) — return values are ignored; failures are
    suppressed by PluginManager so a broken plugin never crashes the agent.

    Args:
        name: Human-readable plugin name used in log messages. Defaults to
              the class name.
        priority: Execution order for pre-hooks (lower = earlier). Default 50.
                  Observational hooks always run concurrently so priority is
                  ignored for them.
        enabled: Set False to disable all hooks without unregistering.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        priority: int = 50,
        enabled: bool = True,
    ) -> None:
        self.name: str = name or type(self).__name__
        self.priority: int = priority
        self.enabled: bool = enabled

    # ------------------------------------------------------------------
    # Lifecycle hooks — override in subclasses
    # ------------------------------------------------------------------

    async def on_agent_start(self, ctx: AgentStartContext) -> None:
        """Called once when an agent run begins."""
        return

    async def on_agent_end(self, ctx: AgentEndContext) -> None:
        """Called once when an agent run completes (success or failure)."""
        return

    async def pre_llm_call(
        self,
        ctx: LLMCallContext,
        messages: list[Any],
        tools: list[Any] | None,
    ) -> LLMCallModification:
        """Called before each LLM API call. Return None to pass through unchanged."""
        return None

    async def post_llm_call(
        self,
        ctx: LLMCallContext,
        content: str | None,
        token_usage: Any,
    ) -> None:
        """Called after each LLM API call completes."""
        return

    async def pre_tool_call(
        self,
        ctx: ToolCallContext,
        arguments: dict[str, Any],
    ) -> ToolCallModification:
        """Called before each tool invocation. Return None to pass through unchanged."""
        return None

    async def post_tool_call(
        self,
        ctx: ToolResultContext,
        result: Any,
    ) -> None:
        """Called after each tool invocation completes."""
        return

    async def pre_memory_write(
        self,
        ctx: MemoryWriteContext,
        content: str,
    ) -> MemoryWriteModification:
        """Called before each memory write. Return None to pass through unchanged."""
        return None

    async def post_memory_write(
        self,
        ctx: MemoryWriteContext,
        record_id: str | None,
    ) -> None:
        """Called after a memory record is persisted."""
        return

    async def on_error(self, ctx: ErrorContext) -> None:
        """Called when the agent loop raises an unhandled exception."""
        return
