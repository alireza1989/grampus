"""Tool executor — validates, executes, retries, and records tool calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from grampus.core.errors import (
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from grampus.core.logging import get_logger
from grampus.core.types import ToolCall, ToolResult
from grampus.tools.registry import ToolRegistry

logger = get_logger(__name__)


class ToolExecutionRecord(BaseModel):
    """Immutable record of a single tool call execution."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult
    started_at: datetime
    duration_ms: int


class ToolExecutor:
    """Executes tool calls from the registry with validation, retry, and idempotency.

    Args:
        registry: Source of registered tools.
        timeout_seconds: Per-call execution timeout.
        max_retries: How many times to retry on retriable errors (0 = no retry).
        retry_delay_seconds: Sleep between retries.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        self._registry = registry
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._records: dict[str, ToolExecutionRecord] = {}

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute *tool_call*, honouring idempotency, validation, and retry policy.

        Returns:
            ToolResult on success.

        Raises:
            ToolNotFoundError: If the tool name is not in the registry.
            ToolValidationError: If required arguments are missing.
            ToolTimeoutError: If the call exceeds *timeout_seconds*.
            Exception: Any unretriable exception from the tool function after
                retries are exhausted.
        """
        if tool_call.id in self._records:
            logger.debug("tool.idempotent_hit", tool_call_id=tool_call.id)
            return self._records[tool_call.id].result

        registered = self._registry.get_or_raise(tool_call.name)

        missing = [
            p.name
            for p in registered.definition.parameters
            if p.required and p.name not in tool_call.arguments
        ]
        if missing:
            raise ToolValidationError(
                f"Tool '{tool_call.name}' missing required arguments: {missing}",
                code="tool.missing_args",
                details={"tool_name": tool_call.name, "missing": missing},
                hint="Check that all required arguments are provided and match the tool's parameter schema.",
            )

        started = time.monotonic()
        started_at = datetime.now(UTC)
        last_exc: BaseException = RuntimeError("unreachable")

        for attempt in range(self._max_retries + 1):
            if attempt > 0 and self._retry_delay > 0:
                await asyncio.sleep(self._retry_delay)
            try:
                output = await self._call_fn(registered.fn, tool_call.arguments)
                duration_ms = int((time.monotonic() - started) * 1000)
                result = ToolResult(
                    tool_call_id=tool_call.id,
                    output=output,
                    error=None,
                    duration_ms=duration_ms,
                )
                self._records[tool_call.id] = ToolExecutionRecord(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    result=result,
                    started_at=started_at,
                    duration_ms=duration_ms,
                )
                logger.debug(
                    "tool.executed",
                    tool=tool_call.name,
                    attempt=attempt,
                    duration_ms=duration_ms,
                )
                return result
            except (ToolNotFoundError, ToolValidationError, ToolTimeoutError):
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "tool.attempt_failed",
                    tool=tool_call.name,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt == self._max_retries:
                    break

        raise last_exc

    async def _call_fn(self, fn: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        """Invoke *fn* with *arguments*, respecting timeout and async/sync detection."""
        try:
            if asyncio.iscoroutinefunction(fn):
                coro = fn(**arguments)
            else:
                coro = asyncio.to_thread(fn, **arguments)
            return await asyncio.wait_for(coro, timeout=self._timeout)
        except TimeoutError as exc:
            raise ToolTimeoutError(
                f"Tool exceeded timeout of {self._timeout}s",
                code="tool.timeout",
                details={"timeout_seconds": self._timeout},
                hint="Increase the tool timeout in ToolExecutor config or optimize the tool implementation.",
            ) from exc

    def get_record(self, tool_call_id: str) -> ToolExecutionRecord | None:
        """Return the execution record for *tool_call_id*, or None."""
        return self._records.get(tool_call_id)

    def all_records(self) -> list[ToolExecutionRecord]:
        """Return all stored execution records in insertion order."""
        return list(self._records.values())
