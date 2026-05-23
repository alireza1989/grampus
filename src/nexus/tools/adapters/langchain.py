"""LangChain tool adapter — wrap any LangChain tool as a Nexus RegisteredTool.

This module is importable without LangChain installed. All LangChain imports
are deferred inside function bodies behind try/except guards.
"""

from __future__ import annotations

import asyncio
import types as _stdlib_types
import typing
from typing import Any

from nexus.core.errors import ToolError
from nexus.core.logging import get_logger
from nexus.core.types import ToolDefinition, ToolParameter
from nexus.tools.registry import RegisteredTool, ToolRegistry

logger = get_logger(__name__)

_RUN_METHODS: tuple[str, ...] = ("run", "_run", "arun", "_arun")

# Python 3.10+ union type (``str | None``) vs typing.Union
_UNION_TYPE: type | None = getattr(_stdlib_types, "UnionType", None)


def _validate_langchain_tool(lc_tool: Any) -> None:
    """Raise ToolError if *lc_tool* does not look like a LangChain BaseTool."""
    for attr in ("name", "description"):
        if not hasattr(lc_tool, attr):
            raise ToolError(
                f"Object is missing required attribute '{attr}'",
                code="INVALID_LANGCHAIN_TOOL",
                hint="Pass a LangChain BaseTool instance.",
            )
    if not any(hasattr(lc_tool, m) for m in _RUN_METHODS):
        raise ToolError(
            "Object has no callable run method (.run, ._run, .arun, or ._arun)",
            code="INVALID_LANGCHAIN_TOOL",
            hint="Pass a LangChain BaseTool instance.",
        )


def _map_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON schema type string."""
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    origin = typing.get_origin(annotation)
    is_union = origin is typing.Union
    if not is_union and _UNION_TYPE is not None:
        is_union = isinstance(annotation, _UNION_TYPE)
    if is_union:
        inner = [a for a in typing.get_args(annotation) if a is not type(None)]
        if inner:
            return _map_type(inner[0])
    return "string"


def _extract_parameters(lc_tool: Any) -> list[ToolParameter]:
    """Extract ToolParameter list from a LangChain tool's args_schema."""
    schema = getattr(lc_tool, "args_schema", None)
    if schema is None:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Input to the tool.",
                required=True,
            )
        ]
    model_fields: dict[str, Any] = getattr(schema, "model_fields", {})
    params: list[ToolParameter] = []
    for field_name, field_info in model_fields.items():
        annotation = getattr(field_info, "annotation", None)
        is_req_method = getattr(field_info, "is_required", None)
        required = bool(is_req_method()) if callable(is_req_method) else True
        description: str = getattr(field_info, "description", None) or ""
        params.append(
            ToolParameter(
                name=field_name,
                type=_map_type(annotation),
                description=description,
                required=required,
            )
        )
    return params


def from_langchain(
    lc_tool: Any,
    *,
    registry: ToolRegistry | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
) -> RegisteredTool:
    """Wrap a LangChain tool as a Nexus RegisteredTool.

    Args:
        lc_tool: A LangChain BaseTool instance (duck-typed as Any).
        registry: If provided, register the resulting tool into this registry.
        name_override: Override the tool's name.
        description_override: Override the tool's description.

    Returns:
        A RegisteredTool wrapping the LangChain tool.

    Raises:
        ToolError: If *lc_tool* does not look like a LangChain BaseTool.
    """
    _validate_langchain_tool(lc_tool)

    name: str = name_override if name_override is not None else str(lc_tool.name)
    description: str = (
        description_override if description_override is not None else str(lc_tool.description)
    )
    parameters = _extract_parameters(lc_tool)

    async def _wrapped(**kwargs: Any) -> dict[str, Any]:
        try:
            if hasattr(lc_tool, "arun"):
                result = await lc_tool.arun(**kwargs)
            elif hasattr(lc_tool, "_arun"):
                result = await lc_tool._arun(**kwargs)
            else:
                if kwargs:
                    result = await asyncio.to_thread(lc_tool.run, kwargs)
                else:
                    result = await asyncio.to_thread(lc_tool.run, "")
            return {"ok": True, "result": str(result)}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "code": "LANGCHAIN_TOOL_ERROR"}

    if registry is not None:
        return registry.register(
            _wrapped,
            name=name,
            description=description,
            parameters=parameters,
        )

    definition = ToolDefinition(name=name, description=description, parameters=parameters)
    return RegisteredTool(name=name, description=description, definition=definition, fn=_wrapped)


def register_langchain_tools(
    lc_tools: list[Any],
    registry: ToolRegistry,
    *,
    skip_on_error: bool = False,
) -> list[RegisteredTool]:
    """Register a list of LangChain tools into a Nexus ToolRegistry.

    Args:
        lc_tools: List of LangChain BaseTool instances.
        registry: Target ToolRegistry to register into.
        skip_on_error: When True, log warnings for invalid tools instead of raising.

    Returns:
        List of successfully registered RegisteredTool instances.
    """
    result: list[RegisteredTool] = []
    for lc_tool in lc_tools:
        try:
            registered = from_langchain(lc_tool, registry=registry)
            result.append(registered)
        except ToolError as exc:
            if skip_on_error:
                logger.warning(
                    "langchain_adapter.skip_tool",
                    error=str(exc),
                    tool=getattr(lc_tool, "name", repr(lc_tool)),
                )
            else:
                raise
    return result
