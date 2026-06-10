"""PluginManager — registration and hook dispatch for the Nexus plugin system."""

from __future__ import annotations

import asyncio
from typing import Any

from nexus.core.logging import get_logger
from nexus.plugins.base import NexusPlugin
from nexus.plugins.types import (
    AgentEndContext,
    AgentStartContext,
    ErrorContext,
    HookBlockedError,
    LLMCallContext,
    MemoryWriteContext,
    ToolCallContext,
    ToolResultContext,
)

_log = get_logger(__name__)


class PluginManager:
    """Manages plugin registration and hook dispatch.

    Pre-hooks are called sequentially in priority order (lowest number first).
    Each plugin receives the value returned by the previous plugin, enabling a
    pipeline of transformations. Raise HookBlockedError to abort the operation.

    Observational hooks (on_*, post_*) run concurrently via asyncio.gather.
    Individual plugin failures are logged and suppressed — a broken plugin
    never crashes the agent.

    Args:
        plugins: Initial list of NexusPlugin instances. Additional plugins can
                 be added later via .register().
    """

    def __init__(self, plugins: list[NexusPlugin] | None = None) -> None:
        self._plugins: list[NexusPlugin] = list(plugins or [])

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin: NexusPlugin) -> None:
        """Register a plugin. Raises ValueError if a plugin with the same name exists."""
        if any(p.name == plugin.name for p in self._plugins):
            raise ValueError(f"Plugin already registered: {plugin.name!r}")
        self._plugins.append(plugin)
        _log.info("plugin_registered", plugin=plugin.name, priority=plugin.priority)

    def unregister(self, name: str) -> None:
        """Unregister a plugin by name. Silent no-op if not found."""
        before = len(self._plugins)
        self._plugins = [p for p in self._plugins if p.name != name]
        if len(self._plugins) < before:
            _log.info("plugin_unregistered", plugin=name)

    def list_plugins(self) -> list[str]:
        """Return names of all registered plugins in priority order."""
        return [p.name for p in self._sorted_plugins()]

    # ------------------------------------------------------------------
    # Pre-hook dispatch (sequential, value threading, HookBlockedError bubbles)
    # ------------------------------------------------------------------

    async def call_pre_llm(
        self,
        ctx: LLMCallContext,
        messages: list[Any],
        tools: list[Any] | None,
    ) -> list[Any]:
        """Run all pre_llm_call hooks. Returns the (possibly modified) messages."""
        current = messages
        for plugin in self._sorted_plugins():
            try:
                result = await plugin.pre_llm_call(ctx, current, tools)
                if result is not None:
                    current = result
            except HookBlockedError:
                raise
            except Exception:
                _log.warning("plugin_hook_error", plugin=plugin.name, hook="pre_llm_call")
        return current

    async def call_pre_tool(
        self,
        ctx: ToolCallContext,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Run all pre_tool_call hooks. Returns the (possibly modified) arguments."""
        current = arguments
        for plugin in self._sorted_plugins():
            try:
                result = await plugin.pre_tool_call(ctx, current)
                if result is not None:
                    current = result
            except HookBlockedError:
                raise
            except Exception:
                _log.warning("plugin_hook_error", plugin=plugin.name, hook="pre_tool_call")
        return current

    async def call_pre_memory_write(
        self,
        ctx: MemoryWriteContext,
        content: str,
    ) -> str:
        """Run all pre_memory_write hooks. Returns the (possibly modified) content."""
        current = content
        for plugin in self._sorted_plugins():
            try:
                result = await plugin.pre_memory_write(ctx, current)
                if result is not None:
                    current = result
            except HookBlockedError:
                raise
            except Exception:
                _log.warning("plugin_hook_error", plugin=plugin.name, hook="pre_memory_write")
        return current

    # ------------------------------------------------------------------
    # Observational hook dispatch (concurrent, failures suppressed)
    # ------------------------------------------------------------------

    async def call_on_agent_start(self, ctx: AgentStartContext) -> None:
        """Fire on_agent_start on all active plugins concurrently."""
        await self._gather("on_agent_start", [p.on_agent_start(ctx) for p in self._active()])

    async def call_on_agent_end(self, ctx: AgentEndContext) -> None:
        """Fire on_agent_end on all active plugins concurrently."""
        await self._gather("on_agent_end", [p.on_agent_end(ctx) for p in self._active()])

    async def call_post_llm(
        self, ctx: LLMCallContext, content: str | None, token_usage: Any
    ) -> None:
        """Fire post_llm_call on all active plugins concurrently."""
        coros = [p.post_llm_call(ctx, content, token_usage) for p in self._active()]
        await self._gather("post_llm_call", coros)

    async def call_post_tool(self, ctx: ToolResultContext, result: Any) -> None:
        """Fire post_tool_call on all active plugins concurrently."""
        await self._gather(
            "post_tool_call", [p.post_tool_call(ctx, result) for p in self._active()]
        )

    async def call_post_memory_write(self, ctx: MemoryWriteContext, record_id: str | None) -> None:
        """Fire post_memory_write on all active plugins concurrently."""
        coros = [p.post_memory_write(ctx, record_id) for p in self._active()]
        await self._gather("post_memory_write", coros)

    async def call_on_error(self, ctx: ErrorContext) -> None:
        """Fire on_error on all active plugins concurrently."""
        await self._gather("on_error", [p.on_error(ctx) for p in self._active()])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sorted_plugins(self) -> list[NexusPlugin]:
        """Return enabled plugins sorted by priority (lowest first)."""
        return sorted(
            (p for p in self._plugins if p.enabled),
            key=lambda p: p.priority,
        )

    def _active(self) -> list[NexusPlugin]:
        """Return enabled plugins in insertion order (for concurrent dispatch)."""
        return [p for p in self._plugins if p.enabled]

    async def _gather(self, hook_name: str, coros: list[Any]) -> None:
        """Run coroutines concurrently; log and suppress individual failures."""
        if not coros:
            return
        results = await asyncio.gather(*coros, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                _log.warning("plugin_hook_error", hook=hook_name, error=str(r))
