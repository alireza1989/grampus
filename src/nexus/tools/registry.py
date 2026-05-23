"""Tool registry — register, look up, and list tools by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from nexus.core.errors import ToolError, ToolNotFoundError
from nexus.core.logging import get_logger
from nexus.core.types import ToolDefinition, ToolParameter

logger = get_logger(__name__)


class RegisteredTool(BaseModel):
    """A tool stored in the registry alongside its callable."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    definition: ToolDefinition
    fn: Any


class ToolRegistry:
    """Central registry for all tools available to agents.

    Tools are registered by name and can be retrieved individually or
    exported as a list of ToolDefinition objects for passing to a ModelClient.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None,
        version: str = "1.0.0",
    ) -> RegisteredTool:
        """Register a callable as a named tool.

        Args:
            fn: The callable to invoke when the tool is called.
            name: Unique tool name.
            description: Human-readable description.
            parameters: Optional list of ToolParameter specs.
            version: Semantic version string.

        Returns:
            The RegisteredTool instance stored in the registry.

        Raises:
            ToolError: If a tool with *name* is already registered.
        """
        if name in self._tools:
            raise ToolError(
                f"Tool '{name}' is already registered",
                code="tool.duplicate_registration",
                hint="Choose a unique name for this tool or call registry.unregister() first.",
            )

        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
            version=version,
        )
        registered = RegisteredTool(
            name=name, description=description, definition=definition, fn=fn
        )
        self._tools[name] = registered
        logger.debug("tool.registered", name=name, version=version)
        return registered

    def tool(
        self,
        *,
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None,
        version: str = "1.0.0",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that registers a function and returns it unchanged.

        Usage::

            @registry.tool(name="add", description="Add two numbers")
            def add(a: int, b: int) -> int:
                return a + b
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                fn, name=name, description=description, parameters=parameters, version=version
            )
            return fn

        return decorator

    def get(self, name: str) -> RegisteredTool | None:
        """Return the RegisteredTool for *name*, or None if not found."""
        return self._tools.get(name)

    def get_or_raise(self, name: str) -> RegisteredTool:
        """Return the RegisteredTool for *name*, raising ToolNotFoundError if absent.

        Raises:
            ToolNotFoundError: If no tool with *name* is registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered",
                code="tool.not_found",
                details={"tool_name": name},
                hint="Register the tool with @registry.tool() or check the tool name for typos.",
            )
        return tool

    def list_all(self) -> list[RegisteredTool]:
        """Return all registered tools in registration order."""
        return list(self._tools.values())

    def to_definitions(self) -> list[ToolDefinition]:
        """Return ToolDefinition objects for all registered tools.

        Suitable for passing directly to a ModelClient.
        """
        return [t.definition for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
