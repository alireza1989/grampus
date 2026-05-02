"""Tests for nexus.tools.registry — ToolRegistry."""

from __future__ import annotations

import pytest

from nexus.core.errors import ToolError, ToolNotFoundError
from nexus.core.types import ToolDefinition, ToolParameter
from nexus.tools.registry import RegisteredTool, ToolRegistry


def _add_greet(registry: ToolRegistry) -> RegisteredTool:
    """Helper: register a simple greet tool and return the RegisteredTool."""

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    return registry.register(
        greet,
        name="greet",
        description="Say hello",
        parameters=[
            ToolParameter(name="name", type="string", description="Person's name", required=True)
        ],
    )


class TestRegister:
    def test_register_returns_registered_tool(self, registry: ToolRegistry) -> None:
        registered = _add_greet(registry)
        assert isinstance(registered, RegisteredTool)

    def test_registered_tool_has_correct_name(self, registry: ToolRegistry) -> None:
        registered = _add_greet(registry)
        assert registered.name == "greet"

    def test_registered_tool_has_correct_description(self, registry: ToolRegistry) -> None:
        registered = _add_greet(registry)
        assert registered.description == "Say hello"

    def test_registered_tool_stores_callable(self, registry: ToolRegistry) -> None:
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        registered = registry.register(greet, name="greet", description="Say hello")
        assert registered.fn is greet

    def test_registered_tool_has_definition(self, registry: ToolRegistry) -> None:
        registered = _add_greet(registry)
        assert isinstance(registered.definition, ToolDefinition)

    def test_duplicate_name_raises_tool_error(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        with pytest.raises(ToolError):
            _add_greet(registry)

    def test_register_no_parameters_allowed(self, registry: ToolRegistry) -> None:
        def ping() -> str:
            return "pong"

        registered = registry.register(ping, name="ping", description="Ping tool")
        assert registered.name == "ping"
        assert registered.definition.parameters == []

    def test_version_stored_in_definition(self, registry: ToolRegistry) -> None:
        def fn() -> None:
            pass

        registered = registry.register(fn, name="fn", description="d", version="2.0.0")
        assert registered.definition.version == "2.0.0"


class TestToolDecorator:
    def test_decorator_registers_function(self, registry: ToolRegistry) -> None:
        @registry.tool(name="add", description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert "add" in registry

    def test_decorator_returns_original_function_unchanged(self, registry: ToolRegistry) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        result = registry.tool(name="add", description="Add")(add)
        assert result is add

    def test_decorated_function_still_callable(self, registry: ToolRegistry) -> None:
        @registry.tool(name="double", description="Double a number")
        def double(x: int) -> int:
            return x * 2

        assert double(5) == 10


class TestGet:
    def test_get_returns_none_for_unknown_name(self, registry: ToolRegistry) -> None:
        assert registry.get("nonexistent") is None

    def test_get_returns_registered_tool(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        result = registry.get("greet")
        assert result is not None
        assert result.name == "greet"


class TestGetOrRaise:
    def test_get_or_raise_raises_tool_not_found_for_unknown(
        self, registry: ToolRegistry
    ) -> None:
        with pytest.raises(ToolNotFoundError):
            registry.get_or_raise("unknown")

    def test_tool_not_found_error_is_tool_error(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolError):
            registry.get_or_raise("unknown")

    def test_get_or_raise_returns_tool_when_found(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        result = registry.get_or_raise("greet")
        assert result.name == "greet"


class TestListAll:
    def test_list_all_empty_by_default(self, registry: ToolRegistry) -> None:
        assert registry.list_all() == []

    def test_list_all_returns_all_registered(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        registry.register(lambda: None, name="ping", description="ping")
        items = registry.list_all()
        assert len(items) == 2

    def test_list_all_names_match(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        names = {t.name for t in registry.list_all()}
        assert "greet" in names


class TestToDefinitions:
    def test_to_definitions_returns_list_of_tool_definitions(
        self, registry: ToolRegistry
    ) -> None:
        _add_greet(registry)
        defs = registry.to_definitions()
        assert all(isinstance(d, ToolDefinition) for d in defs)

    def test_to_definitions_correct_count(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        registry.register(lambda: None, name="ping", description="ping")
        assert len(registry.to_definitions()) == 2

    def test_to_definitions_produces_valid_function_schema(
        self, registry: ToolRegistry
    ) -> None:
        _add_greet(registry)
        defs = registry.to_definitions()
        schema = defs[0].to_function_schema()
        assert schema["name"] == "greet"
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "name" in schema["parameters"]["properties"]

    def test_to_definitions_required_params_in_schema(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        schema = registry.to_definitions()[0].to_function_schema()
        assert "name" in schema["parameters"]["required"]


class TestDunderMethods:
    def test_len_empty(self, registry: ToolRegistry) -> None:
        assert len(registry) == 0

    def test_len_after_register(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        assert len(registry) == 1

    def test_len_multiple(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        registry.register(lambda: None, name="ping", description="ping")
        assert len(registry) == 2

    def test_contains_false_for_unknown(self, registry: ToolRegistry) -> None:
        assert "greet" not in registry

    def test_contains_true_after_register(self, registry: ToolRegistry) -> None:
        _add_greet(registry)
        assert "greet" in registry