"""Tests for nexus.tools.library — LIBRARY_REGISTRY integration."""

from __future__ import annotations

from nexus.tools.library import LIBRARY_REGISTRY, get_library_registry, get_tool_names

EXPECTED_TOOLS = {
    "calculator",
    "http_request",
    "file_read",
    "file_write",
    "web_search",
    "sql_query",
    "send_email",
}


class TestLibraryRegistry:
    def test_library_registry_has_all_tools(self) -> None:
        names = {t.name for t in LIBRARY_REGISTRY.list_all()}
        assert names == EXPECTED_TOOLS

    def test_get_tool_names_returns_list(self) -> None:
        names = get_tool_names()
        assert isinstance(names, list)
        assert set(names) == EXPECTED_TOOLS

    def test_library_registry_is_singleton(self) -> None:
        assert get_library_registry() is LIBRARY_REGISTRY

    def test_each_tool_has_description(self) -> None:
        for tool in LIBRARY_REGISTRY.list_all():
            assert tool.description, f"Tool '{tool.name}' has no description"
            assert len(tool.description) > 5

    def test_each_tool_has_parameters(self) -> None:
        for tool in LIBRARY_REGISTRY.list_all():
            assert len(tool.definition.parameters) > 0, f"Tool '{tool.name}' has no parameters"

    def test_each_tool_has_callable(self) -> None:
        for tool in LIBRARY_REGISTRY.list_all():
            assert callable(tool.fn), f"Tool '{tool.name}' fn is not callable"

    def test_tool_definitions_produce_valid_schema(self) -> None:
        for tool in LIBRARY_REGISTRY.list_all():
            schema = tool.definition.to_function_schema()
            assert schema["name"] == tool.name
            assert "parameters" in schema
