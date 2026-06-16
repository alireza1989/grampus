"""Tests for SandboxManager and CodeExecutor — always exercises the local fallback."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

import pytest

from grampus.core.errors import ToolTimeoutError
from grampus.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers — force the local-fallback path by blocking `import docker`
# ---------------------------------------------------------------------------


def _import_sandbox_without_docker() -> Any:
    """Import sandbox module with docker unavailable."""
    # Remove cached module so re-import picks up the patch.
    for key in list(sys.modules.keys()):
        if "grampus.tools.sandbox" in key:
            del sys.modules[key]

    with patch.dict("sys.modules", {"docker": None}):
        from grampus.tools.sandbox import manager  # type: ignore[import]

    return manager


# ---------------------------------------------------------------------------
# SandboxResult model
# ---------------------------------------------------------------------------


def test_sandbox_result_round_trips_json() -> None:
    for key in list(sys.modules.keys()):
        if "grampus.tools.sandbox" in key:
            del sys.modules[key]
    with patch.dict("sys.modules", {"docker": None}):
        from grampus.tools.sandbox.manager import SandboxResult

    result = SandboxResult(
        stdout="hi",
        stderr="",
        return_value=None,
        exit_code=0,
        duration_ms=5.0,
    )
    assert SandboxResult.model_validate_json(result.model_dump_json()) == result


# ---------------------------------------------------------------------------
# SandboxManager — local subprocess fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_captures_stdout() -> None:
    mod = _import_sandbox_without_docker()
    manager = mod.SandboxManager()
    result = await manager.execute('print("hello world")')
    assert result.exit_code == 0
    assert "hello world" in result.stdout
    await manager.close()


@pytest.mark.asyncio
async def test_sandbox_captures_stderr_on_syntax_error() -> None:
    mod = _import_sandbox_without_docker()
    manager = mod.SandboxManager()
    result = await manager.execute("def bad(:)")
    assert result.exit_code != 0
    await manager.close()


@pytest.mark.asyncio
async def test_sandbox_nonzero_exit_for_runtime_error() -> None:
    mod = _import_sandbox_without_docker()
    manager = mod.SandboxManager()
    result = await manager.execute("raise ValueError('boom')")
    assert result.exit_code != 0
    await manager.close()


@pytest.mark.asyncio
async def test_sandbox_timeout_raises_tool_timeout_error() -> None:
    for key in list(sys.modules.keys()):
        if "grampus.tools.sandbox" in key:
            del sys.modules[key]
    with patch.dict("sys.modules", {"docker": None}):
        from grampus.tools.sandbox.manager import SandboxConfig, SandboxManager

    manager = SandboxManager(SandboxConfig(execution_timeout_seconds=1))
    with pytest.raises(ToolTimeoutError):
        await manager.execute("import time; time.sleep(10)")
    await manager.close()


@pytest.mark.asyncio
async def test_sandbox_duration_ms_is_positive() -> None:
    mod = _import_sandbox_without_docker()
    manager = mod.SandboxManager()
    result = await manager.execute("x = 1 + 1")
    assert result.duration_ms > 0
    await manager.close()


# ---------------------------------------------------------------------------
# CodeExecutionResult model
# ---------------------------------------------------------------------------


def test_code_execution_result_round_trips_json() -> None:
    for key in list(sys.modules.keys()):
        if "grampus.tools.sandbox" in key:
            del sys.modules[key]
    with patch.dict("sys.modules", {"docker": None}):
        from grampus.tools.sandbox.code_executor import CodeExecutionResult

    result = CodeExecutionResult(
        stdout="out",
        stderr="",
        return_value=None,
        duration_ms=12.0,
        tools_called=["my_tool"],
    )
    assert CodeExecutionResult.model_validate_json(result.model_dump_json()) == result


# ---------------------------------------------------------------------------
# CodeExecutor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_executor_returns_result() -> None:
    for key in list(sys.modules.keys()):
        if "grampus.tools.sandbox" in key:
            del sys.modules[key]
    with patch.dict("sys.modules", {"docker": None}):
        from grampus.tools.sandbox.code_executor import CodeExecutor
        from grampus.tools.sandbox.manager import SandboxManager

    registry = ToolRegistry()
    manager = SandboxManager()
    executor = CodeExecutor(sandbox=manager, tool_registry=registry)
    result = await executor.execute('print("from executor")')
    assert "from executor" in result.stdout
    assert result.error is None
    await manager.close()


@pytest.mark.asyncio
async def test_code_executor_populates_tools_called() -> None:
    for key in list(sys.modules.keys()):
        if "grampus.tools.sandbox" in key:
            del sys.modules[key]
    with patch.dict("sys.modules", {"docker": None}):
        from grampus.tools.sandbox.code_executor import CodeExecutor
        from grampus.tools.sandbox.manager import SandboxManager

    registry = ToolRegistry()
    registry.register(lambda: None, name="my_tool", description="A test tool")
    registry.register(lambda: None, name="other_tool", description="Another tool")

    manager = SandboxManager()
    executor = CodeExecutor(sandbox=manager, tool_registry=registry)

    code = "result = my_tool(x=1)\nprint(result)"
    result = await executor.execute(code)

    assert "my_tool" in result.tools_called
    assert "other_tool" not in result.tools_called
    await manager.close()


@pytest.mark.asyncio
async def test_code_executor_sets_error_on_failure() -> None:
    for key in list(sys.modules.keys()):
        if "grampus.tools.sandbox" in key:
            del sys.modules[key]
    with patch.dict("sys.modules", {"docker": None}):
        from grampus.tools.sandbox.code_executor import CodeExecutor
        from grampus.tools.sandbox.manager import SandboxManager

    registry = ToolRegistry()
    manager = SandboxManager()
    executor = CodeExecutor(sandbox=manager, tool_registry=registry)

    result = await executor.execute("raise RuntimeError('test error')")
    assert result.error is not None
    await manager.close()


@pytest.mark.asyncio
async def test_code_executor_tools_called_empty_when_no_match() -> None:
    for key in list(sys.modules.keys()):
        if "grampus.tools.sandbox" in key:
            del sys.modules[key]
    with patch.dict("sys.modules", {"docker": None}):
        from grampus.tools.sandbox.code_executor import CodeExecutor
        from grampus.tools.sandbox.manager import SandboxManager

    registry = ToolRegistry()
    registry.register(lambda: None, name="unused_tool", description="Never called")

    manager = SandboxManager()
    executor = CodeExecutor(sandbox=manager, tool_registry=registry)

    result = await executor.execute("x = 1 + 1")
    assert result.tools_called == []
    await manager.close()
