"""Code executor — runs LLM-generated Python inside the sandbox with tool injection."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from nexus.core.logging import get_logger
from nexus.tools.registry import ToolRegistry
from nexus.tools.sandbox.manager import SandboxManager

logger = get_logger(__name__)


class CodeExecutionResult(BaseModel):
    """Result of a CodeExecutor run.

    Attributes:
        stdout: Captured standard output from the executed code.
        stderr: Captured standard error.
        return_value: Value set as ``__result__`` inside the code (if any).
        duration_ms: Wall-clock execution time in milliseconds.
        tools_called: Names of registered tools referenced as calls in the code.
        error: Human-readable error message when execution failed; None on success.
    """

    stdout: str
    stderr: str
    return_value: Any
    duration_ms: float
    tools_called: list[str]
    error: str | None = None


class CodeExecutor:
    """Executes LLM-generated Python in the sandbox with registered tools injected.

    Tool callables cannot cross a Docker process boundary, so tool injection
    works only with the local fallback backend. In both cases, ``tools_called``
    is populated by a regex scan of the code string — no runtime instrumentation.

    Args:
        sandbox: SandboxManager to delegate execution to.
        tool_registry: Registry whose tools are injected into the execution namespace.
    """

    def __init__(self, sandbox: SandboxManager, tool_registry: ToolRegistry) -> None:
        self._sandbox = sandbox
        self._registry = tool_registry

    async def execute(
        self,
        code: str,
        *,
        extra_namespace: dict[str, Any] | None = None,
    ) -> CodeExecutionResult:
        """Execute *code* in the sandbox and return a structured result.

        Args:
            code: Python source code to execute.
            extra_namespace: Additional variables merged into the execution namespace
                             (local fallback only; ignored by Docker backend).

        Returns:
            CodeExecutionResult with output, duration, and tool call information.
        """
        tools_called = self._detect_tool_calls(code)
        namespace = self._build_namespace(extra_namespace)

        logger.debug(
            "code_executor.execute",
            tools_in_code=tools_called,
            namespace_keys=list(namespace.keys()),
        )

        sandbox_result = await self._sandbox.execute(code, namespace=namespace)

        error: str | None = None
        if sandbox_result.error is not None:
            error = sandbox_result.error
        elif sandbox_result.exit_code != 0:
            error = sandbox_result.stderr or f"Process exited with code {sandbox_result.exit_code}"

        return CodeExecutionResult(
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            return_value=sandbox_result.return_value,
            duration_ms=sandbox_result.duration_ms,
            tools_called=tools_called,
            error=error,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_namespace(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        """Build the namespace dict from the registry plus any extra values."""
        namespace: dict[str, Any] = {tool.name: tool.fn for tool in self._registry.list_all()}
        if extra:
            namespace.update(extra)
        return namespace

    def _detect_tool_calls(self, code: str) -> list[str]:
        """Scan *code* for call-patterns matching registered tool names."""
        called: list[str] = []
        for tool in self._registry.list_all():
            pattern = rf"\b{re.escape(tool.name)}\s*\("
            if re.search(pattern, code):
                called.append(tool.name)
        return called
