"""Integration tests for ToolExecutor: timeout, retry, idempotency, validation."""

from __future__ import annotations

import asyncio

import pytest

from nexus.core.errors import ToolNotFoundError, ToolTimeoutError, ToolValidationError
from nexus.core.types import ToolCall, ToolParameter
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool(
        name="echo",
        description="Echo text",
        parameters=[ToolParameter(name="text", type="string", description="", required=True)],
    )
    async def echo(text: str) -> str:
        return text

    @registry.tool(
        name="add",
        description="Add two numbers",
        parameters=[
            ToolParameter(name="a", type="number", description="", required=True),
            ToolParameter(name="b", type="number", description="", required=True),
        ],
    )
    def add(a: float, b: float) -> float:
        return a + b

    @registry.tool(
        name="slow",
        description="Sleeps longer than timeout",
        parameters=[],
    )
    async def slow() -> str:
        await asyncio.sleep(10)
        return "done"

    return registry


@pytest.mark.integration
class TestToolExecutorIntegration:
    async def test_execute_registered_tool_returns_result(self) -> None:
        executor = ToolExecutor(_make_registry(), timeout_seconds=5.0)
        tc = ToolCall(id="tc-1", name="echo", arguments={"text": "hello"})
        result = await executor.execute(tc)
        assert result.error is None
        assert result.output == "hello"
        assert result.tool_call_id == "tc-1"

    async def test_execute_sync_tool_returns_result(self) -> None:
        executor = ToolExecutor(_make_registry(), timeout_seconds=5.0)
        tc = ToolCall(id="tc-add", name="add", arguments={"a": 3.0, "b": 4.0})
        result = await executor.execute(tc)
        assert result.error is None
        assert result.output == pytest.approx(7.0)

    async def test_execute_missing_tool_raises_not_found(self) -> None:
        executor = ToolExecutor(_make_registry(), timeout_seconds=5.0)
        tc = ToolCall(id="tc-x", name="nonexistent", arguments={})
        with pytest.raises(ToolNotFoundError):
            await executor.execute(tc)

    async def test_execute_respects_timeout(self) -> None:
        executor = ToolExecutor(_make_registry(), timeout_seconds=0.1, max_retries=0)
        tc = ToolCall(id="tc-slow", name="slow", arguments={})
        with pytest.raises(ToolTimeoutError):
            await executor.execute(tc)

    async def test_execute_retries_on_transient_error(self) -> None:
        registry = ToolRegistry()
        call_count = [0]

        @registry.tool(
            name="flaky",
            description="Fails first, then succeeds",
            parameters=[],
        )
        async def flaky() -> str:
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("transient error")
            return "success"

        executor = ToolExecutor(
            registry, timeout_seconds=5.0, max_retries=2, retry_delay_seconds=0.0
        )
        tc = ToolCall(id="tc-flaky", name="flaky", arguments={})
        result = await executor.execute(tc)
        assert result.output == "success"
        assert call_count[0] == 2

    async def test_idempotency_returns_cached_result_on_replay(self) -> None:
        call_count = [0]
        registry = ToolRegistry()

        @registry.tool(name="counted", description="Counts calls", parameters=[])
        async def counted() -> int:
            call_count[0] += 1
            return call_count[0]

        executor = ToolExecutor(registry, timeout_seconds=5.0)
        tc = ToolCall(id="idem-1", name="counted", arguments={})
        result1 = await executor.execute(tc)
        result2 = await executor.execute(tc)
        assert result1.output == result2.output
        assert call_count[0] == 1

    async def test_tool_execution_record_stored(self) -> None:
        executor = ToolExecutor(_make_registry(), timeout_seconds=5.0)
        tc = ToolCall(id="tc-rec", name="echo", arguments={"text": "test"})
        await executor.execute(tc)
        record = executor.get_record("tc-rec")
        assert record is not None
        assert record.tool_name == "echo"
        assert record.duration_ms >= 0

    async def test_missing_required_arg_raises_validation_error(self) -> None:
        executor = ToolExecutor(_make_registry(), timeout_seconds=5.0)
        tc = ToolCall(id="tc-bad", name="echo", arguments={})
        with pytest.raises(ToolValidationError):
            await executor.execute(tc)

    async def test_all_records_returns_execution_history(self) -> None:
        executor = ToolExecutor(_make_registry(), timeout_seconds=5.0)
        for i in range(3):
            tc = ToolCall(id=f"tc-{i}", name="echo", arguments={"text": f"msg{i}"})
            await executor.execute(tc)
        records = executor.all_records()
        assert len(records) == 3
