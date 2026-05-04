"""Integration tests for SandboxManager subprocess execution."""

from __future__ import annotations

import pytest

from nexus.core.errors import ToolTimeoutError
from nexus.tools.sandbox.manager import SandboxConfig, SandboxManager


@pytest.mark.integration
class TestSandboxIntegration:
    async def test_execute_simple_print_captures_stdout(self) -> None:
        mgr = SandboxManager(SandboxConfig(execution_timeout_seconds=10))
        result = await mgr.execute("print('hello sandbox')")
        assert "hello sandbox" in result.stdout
        assert result.exit_code == 0

    async def test_execute_arithmetic_outputs_result(self) -> None:
        mgr = SandboxManager(SandboxConfig(execution_timeout_seconds=10))
        result = await mgr.execute("print(2 + 3)")
        assert "5" in result.stdout
        assert result.exit_code == 0

    async def test_execute_captures_stderr(self) -> None:
        mgr = SandboxManager(SandboxConfig(execution_timeout_seconds=10))
        result = await mgr.execute("import sys; sys.stderr.write('err output\\n')")
        assert "err output" in result.stderr

    async def test_execute_syntax_error_returns_nonzero_exit(self) -> None:
        mgr = SandboxManager(SandboxConfig(execution_timeout_seconds=10))
        result = await mgr.execute("def broken(: pass")
        assert result.exit_code != 0

    async def test_execute_runtime_error_returns_nonzero_exit(self) -> None:
        mgr = SandboxManager(SandboxConfig(execution_timeout_seconds=10))
        result = await mgr.execute("raise ValueError('test error')")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    async def test_execute_timeout_raises_tool_timeout_error(self) -> None:
        mgr = SandboxManager(SandboxConfig(execution_timeout_seconds=1))
        with pytest.raises(ToolTimeoutError):
            await mgr.execute("import time; time.sleep(10)")

    async def test_execute_multiline_code(self) -> None:
        mgr = SandboxManager(SandboxConfig(execution_timeout_seconds=10))
        code = "x = 10\ny = 20\nprint(x + y)"
        result = await mgr.execute(code)
        assert "30" in result.stdout
        assert result.exit_code == 0

    async def test_execute_imports_standard_library(self) -> None:
        mgr = SandboxManager(SandboxConfig(execution_timeout_seconds=10))
        result = await mgr.execute("import json; print(json.dumps({'a': 1}))")
        assert '"a"' in result.stdout
        assert result.exit_code == 0

    async def test_close_does_not_raise(self) -> None:
        mgr = SandboxManager()
        await mgr.execute("print('cleanup test')")
        await mgr.close()
