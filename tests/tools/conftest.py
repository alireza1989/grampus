"""Shared fixtures for tools test suite."""

from __future__ import annotations

import pytest

from grampus.core.types import ToolCall, ToolDefinition, ToolParameter
from grampus.tools.registry import ToolRegistry


@pytest.fixture()
def registry() -> ToolRegistry:
    """Empty ToolRegistry for each test."""
    return ToolRegistry()


@pytest.fixture()
def simple_tool_def() -> ToolDefinition:
    """A minimal ToolDefinition with one required string parameter."""
    return ToolDefinition(
        name="greet",
        description="Say hello",
        parameters=[
            ToolParameter(name="name", type="string", description="Person's name", required=True)
        ],
    )


@pytest.fixture()
def tool_call_factory():
    """Return a factory that builds ToolCall instances."""

    def _factory(name: str, arguments: dict | None = None, call_id: str = "tc-1") -> ToolCall:
        return ToolCall(id=call_id, name=name, arguments=arguments or {})

    return _factory
