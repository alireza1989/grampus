"""Tests for nexus.tools.adapters.langchain — LangChain tool adapter."""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest
from pydantic import BaseModel

from nexus.core.errors import ToolError
from nexus.tools.adapters.langchain import (
    _extract_parameters,
    _map_type,
    from_langchain,
    register_langchain_tools,
)
from nexus.tools.registry import RegisteredTool, ToolRegistry

# ---------------------------------------------------------------------------
# Fake LangChain tool objects (no LangChain dependency)
# ---------------------------------------------------------------------------


class FakeLangChainTool:
    """Minimal duck-type of a LangChain BaseTool with async support."""

    name = "fake_tool"
    description = "A fake tool for testing."
    args_schema = None

    def run(self, tool_input: Any) -> str:
        return f"result:{tool_input}"

    async def arun(self, **kwargs: Any) -> str:
        return f"async_result:{kwargs}"


class FakeSyncLangChainTool:
    """Fake tool that only has a synchronous run method."""

    name = "sync_tool"
    description = "A sync-only fake tool."
    args_schema = None

    def run(self, tool_input: Any) -> str:
        return f"sync_result:{tool_input}"


class FakeErrorLangChainTool:
    """Fake tool whose async execution always raises."""

    name = "error_tool"
    description = "A tool that always raises."
    args_schema = None

    async def arun(self, **kwargs: Any) -> str:
        raise RuntimeError("Tool exploded!")


class SearchInput(BaseModel):
    """Pydantic schema for FakeSearchTool."""

    query: str
    max_results: int = 5


class FakeSearchTool:
    """Fake tool with a Pydantic args_schema for schema extraction tests."""

    name = "search"
    description = "Search the web."
    args_schema = SearchInput

    async def arun(self, query: str, max_results: int = 5) -> str:
        return f"results for {query}"


class FakeMissingName:
    """Duck-type missing the required .name attribute."""

    description = "has description, no name"

    async def arun(self, **kwargs: Any) -> str:
        return ""


# ---------------------------------------------------------------------------
# TestFromLangchain
# ---------------------------------------------------------------------------


class TestFromLangchain:
    def test_from_langchain_returns_registered_tool(self) -> None:
        result = from_langchain(FakeLangChainTool())
        assert isinstance(result, RegisteredTool)

    def test_from_langchain_uses_tool_name(self) -> None:
        result = from_langchain(FakeLangChainTool())
        assert result.name == "fake_tool"

    def test_from_langchain_uses_tool_description(self) -> None:
        result = from_langchain(FakeLangChainTool())
        assert result.description == "A fake tool for testing."

    def test_from_langchain_name_override_applied(self) -> None:
        result = from_langchain(FakeLangChainTool(), name_override="my_tool")
        assert result.name == "my_tool"

    def test_from_langchain_description_override_applied(self) -> None:
        result = from_langchain(FakeLangChainTool(), description_override="Custom description")
        assert result.description == "Custom description"

    def test_from_langchain_invalid_object_raises_tool_error(self) -> None:
        with pytest.raises(ToolError) as exc_info:
            from_langchain("not a tool")
        assert exc_info.value.code == "INVALID_LANGCHAIN_TOOL"

    def test_from_langchain_missing_name_raises_tool_error(self) -> None:
        with pytest.raises(ToolError) as exc_info:
            from_langchain(FakeMissingName())
        assert exc_info.value.code == "INVALID_LANGCHAIN_TOOL"

    def test_from_langchain_no_registry_does_not_register(self) -> None:
        registry = ToolRegistry()
        from_langchain(FakeLangChainTool())
        assert len(registry) == 0

    def test_from_langchain_with_registry_registers_tool(self, registry: ToolRegistry) -> None:
        from_langchain(FakeLangChainTool(), registry=registry)
        assert "fake_tool" in registry


# ---------------------------------------------------------------------------
# TestWrappedExecution
# ---------------------------------------------------------------------------


class TestWrappedExecution:
    async def test_wrapped_tool_async_path_returns_ok_dict(self) -> None:
        registered = from_langchain(FakeLangChainTool())
        result = await registered.fn(input="hello")
        assert result["ok"] is True
        assert "async_result" in result["result"]

    async def test_wrapped_tool_sync_fallback_returns_ok_dict(self) -> None:
        registered = from_langchain(FakeSyncLangChainTool())
        result = await registered.fn(input="hello")
        assert result["ok"] is True
        assert "sync_result" in result["result"]

    async def test_wrapped_tool_exception_returns_err_dict(self) -> None:
        registered = from_langchain(FakeErrorLangChainTool())
        result = await registered.fn()
        assert result["ok"] is False
        assert "Tool exploded!" in result["error"]
        assert result["code"] == "LANGCHAIN_TOOL_ERROR"

    async def test_wrapped_tool_result_is_string(self) -> None:
        registered = from_langchain(FakeLangChainTool())
        result = await registered.fn()
        assert isinstance(result["result"], str)


# ---------------------------------------------------------------------------
# TestSchemaExtraction
# ---------------------------------------------------------------------------


class TestSchemaExtraction:
    def test_no_args_schema_produces_single_input_param(self) -> None:
        registered = from_langchain(FakeLangChainTool())
        params = registered.definition.parameters
        assert len(params) == 1
        assert params[0].name == "input"
        assert params[0].type == "string"
        assert params[0].required is True

    def test_args_schema_extracts_field_names(self) -> None:
        registered = from_langchain(FakeSearchTool())
        names = {p.name for p in registered.definition.parameters}
        assert "query" in names
        assert "max_results" in names

    def test_args_schema_extracts_field_types(self) -> None:
        registered = from_langchain(FakeSearchTool())
        by_name = {p.name: p.type for p in registered.definition.parameters}
        assert by_name["query"] == "string"
        assert by_name["max_results"] == "integer"

    def test_args_schema_required_field_when_no_default(self) -> None:
        registered = from_langchain(FakeSearchTool())
        by_name = {p.name: p for p in registered.definition.parameters}
        assert by_name["query"].required is True

    def test_args_schema_optional_field_when_has_default(self) -> None:
        registered = from_langchain(FakeSearchTool())
        by_name = {p.name: p for p in registered.definition.parameters}
        assert by_name["max_results"].required is False

    def test_type_mapping_str_to_string(self) -> None:
        assert _map_type(str) == "string"

    def test_type_mapping_int_to_integer(self) -> None:
        assert _map_type(int) == "integer"

    def test_type_mapping_bool_to_boolean(self) -> None:
        assert _map_type(bool) == "boolean"

    def test_type_mapping_unknown_to_string(self) -> None:
        assert _map_type(list) == "string"
        assert _map_type(None) == "string"
        assert _map_type(object) == "string"

    def test_extract_parameters_no_schema(self) -> None:
        params = _extract_parameters(FakeLangChainTool())
        assert len(params) == 1
        assert params[0].name == "input"

    def test_extract_parameters_with_schema(self) -> None:
        params = _extract_parameters(FakeSearchTool())
        assert len(params) == 2


# ---------------------------------------------------------------------------
# TestRegisterLangchainTools
# ---------------------------------------------------------------------------


class TestRegisterLangchainTools:
    def test_register_multiple_tools(self, registry: ToolRegistry) -> None:
        tools: list[Any] = [FakeLangChainTool(), FakeSyncLangChainTool()]
        result = register_langchain_tools(tools, registry)
        assert len(result) == 2

    def test_register_tools_all_in_registry(self, registry: ToolRegistry) -> None:
        tools: list[Any] = [FakeLangChainTool(), FakeSyncLangChainTool()]
        register_langchain_tools(tools, registry)
        assert "fake_tool" in registry
        assert "sync_tool" in registry

    def test_skip_on_error_continues_after_bad_tool(self, registry: ToolRegistry) -> None:
        tools: list[Any] = [object(), FakeLangChainTool()]
        result = register_langchain_tools(tools, registry, skip_on_error=True)
        assert len(result) == 1
        assert result[0].name == "fake_tool"

    def test_skip_on_error_false_raises_on_bad_tool(self, registry: ToolRegistry) -> None:
        tools: list[Any] = [object(), FakeLangChainTool()]
        with pytest.raises(ToolError):
            register_langchain_tools(tools, registry, skip_on_error=False)

    def test_returns_list_of_registered_tools(self, registry: ToolRegistry) -> None:
        tools: list[Any] = [FakeLangChainTool()]
        result = register_langchain_tools(tools, registry)
        assert all(isinstance(t, RegisteredTool) for t in result)


# ---------------------------------------------------------------------------
# TestNoLangchainInstalled
# ---------------------------------------------------------------------------


class TestNoLangchainInstalled:
    def test_module_importable_without_langchain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Module must not trigger any top-level LangChain import."""
        original = builtins.__import__
        seen: list[str] = []

        def _track(name: str, *args: Any, **kwargs: Any) -> Any:
            # Only flag imports of the actual LangChain package, not our own module
            if name.startswith("langchain"):
                seen.append(name)
            return original(name, *args, **kwargs)

        # Evict module cache so the import statement is re-evaluated under tracking
        removed = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k in {"nexus.tools.adapters.langchain", "nexus.tools.adapters"}
        }
        monkeypatch.setattr(builtins, "__import__", _track)

        try:
            import importlib

            mod = importlib.import_module("nexus.tools.adapters.langchain")
            assert callable(getattr(mod, "from_langchain", None))
            assert seen == [], f"Unexpected langchain imports at module level: {seen}"
        finally:
            sys.modules.update(removed)
