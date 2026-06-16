"""Tests for grampus.tools.executor — ToolExecutor."""

from __future__ import annotations

import asyncio

import pytest

from grampus.core.errors import (
    ToolError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from grampus.core.types import ToolCall, ToolParameter
from grampus.tools.executor import ToolExecutionRecord, ToolExecutor
from grampus.tools.registry import ToolRegistry


def _make_executor(
    *,
    timeout_seconds: float = 5.0,
    max_retries: int = 2,
    retry_delay_seconds: float = 0.0,
) -> tuple[ToolRegistry, ToolExecutor]:
    registry = ToolRegistry()
    executor = ToolExecutor(
        registry,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    return registry, executor


def _tc(name: str, arguments: dict | None = None, call_id: str = "tc-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments or {})


class TestExecuteSyncFunction:
    async def test_sync_function_returns_tool_result(self) -> None:
        registry, executor = _make_executor()
        registry.register(lambda name: f"Hello, {name}!", name="greet", description="greet")
        result = await executor.execute(_tc("greet", {"name": "World"}))
        assert result.output == "Hello, World!"
        assert result.error is None

    async def test_tool_call_id_matches(self) -> None:
        registry, executor = _make_executor()
        registry.register(lambda: "ok", name="ping", description="ping")
        result = await executor.execute(_tc("ping", call_id="my-id"))
        assert result.tool_call_id == "my-id"

    async def test_duration_ms_non_negative(self) -> None:
        registry, executor = _make_executor()
        registry.register(lambda: "ok", name="ping", description="ping")
        result = await executor.execute(_tc("ping"))
        assert result.duration_ms >= 0


class TestExecuteAsyncFunction:
    async def test_async_function_returns_tool_result(self) -> None:
        registry, executor = _make_executor()

        async def async_greet(name: str) -> str:
            return f"Hi, {name}!"

        registry.register(async_greet, name="async_greet", description="async greet")
        result = await executor.execute(_tc("async_greet", {"name": "Alice"}))
        assert result.output == "Hi, Alice!"
        assert result.error is None

    async def test_async_tool_call_id_matches(self) -> None:
        registry, executor = _make_executor()

        async def noop() -> str:
            return "noop"

        registry.register(noop, name="noop", description="noop")
        result = await executor.execute(_tc("noop", call_id="async-id"))
        assert result.tool_call_id == "async-id"


class TestExecuteErrors:
    async def test_unknown_tool_raises_tool_not_found(self) -> None:
        _, executor = _make_executor()
        with pytest.raises(ToolNotFoundError):
            await executor.execute(_tc("nonexistent"))

    async def test_missing_required_arg_raises_tool_validation_error(self) -> None:
        registry, executor = _make_executor()
        registry.register(
            lambda name: name,
            name="greet",
            description="greet",
            parameters=[
                ToolParameter(name="name", type="string", description="name", required=True)
            ],
        )
        with pytest.raises(ToolValidationError):
            await executor.execute(_tc("greet", {}))

    async def test_optional_arg_absent_does_not_raise(self) -> None:
        registry, executor = _make_executor()
        registry.register(
            lambda name="World": f"Hello, {name}!",
            name="greet",
            description="greet",
            parameters=[
                ToolParameter(
                    name="name", type="string", description="name", required=False, default="World"
                )
            ],
        )
        result = await executor.execute(_tc("greet", {}))
        assert result.error is None

    async def test_timeout_raises_tool_timeout_error(self) -> None:
        registry, executor = _make_executor(timeout_seconds=0.05, max_retries=0)

        async def slow() -> str:
            await asyncio.sleep(10)
            return "done"

        registry.register(slow, name="slow", description="slow")
        with pytest.raises(ToolTimeoutError):
            await executor.execute(_tc("slow"))


class TestRetry:
    async def test_retries_on_failure_then_succeeds(self) -> None:
        registry, executor = _make_executor(max_retries=2, retry_delay_seconds=0.0)
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temporary failure")
            return "success"

        registry.register(flaky, name="flaky", description="flaky")
        result = await executor.execute(_tc("flaky"))
        assert result.output == "success"
        assert call_count == 3

    async def test_tool_not_found_not_retried(self) -> None:
        _, executor = _make_executor(max_retries=2)
        with pytest.raises(ToolNotFoundError):
            await executor.execute(_tc("ghost"))

    async def test_tool_validation_error_not_retried(self) -> None:
        registry, executor = _make_executor(max_retries=2)
        registry.register(
            lambda x: x,
            name="fn",
            description="fn",
            parameters=[ToolParameter(name="x", type="string", description="x", required=True)],
        )
        with pytest.raises(ToolValidationError):
            await executor.execute(_tc("fn", {}))

    async def test_max_retries_exhausted_raises(self) -> None:
        registry, executor = _make_executor(max_retries=2, retry_delay_seconds=0.0)

        def always_fails() -> str:
            raise RuntimeError("always fails")

        registry.register(always_fails, name="fail", description="fail")
        with pytest.raises(RuntimeError, match="always fails"):
            await executor.execute(_tc("fail"))

    async def test_tool_error_is_retried(self) -> None:
        registry, executor = _make_executor(max_retries=2, retry_delay_seconds=0.0)
        call_count = 0

        def tool_error_then_ok() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ToolError("transient", code="transient")
            return "ok"

        registry.register(tool_error_then_ok, name="te", description="te")
        result = await executor.execute(_tc("te"))
        assert result.output == "ok"
        assert call_count == 3


class TestIdempotency:
    async def test_same_call_id_executes_only_once(self) -> None:
        registry, executor = _make_executor()
        call_count = 0

        def counter() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        registry.register(counter, name="counter", description="counter")
        tc = _tc("counter", call_id="idem-1")
        result1 = await executor.execute(tc)
        result2 = await executor.execute(tc)
        assert result1.output == result2.output
        assert call_count == 1

    async def test_different_call_ids_execute_separately(self) -> None:
        registry, executor = _make_executor()
        call_count = 0

        def counter() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        registry.register(counter, name="counter", description="counter")
        await executor.execute(_tc("counter", call_id="id-1"))
        await executor.execute(_tc("counter", call_id="id-2"))
        assert call_count == 2


class TestRecords:
    async def test_get_record_returns_none_for_unknown(self) -> None:
        _, executor = _make_executor()
        assert executor.get_record("nonexistent") is None

    async def test_get_record_returns_stored_record(self) -> None:
        registry, executor = _make_executor()
        registry.register(lambda: "ok", name="ping", description="ping")
        await executor.execute(_tc("ping", call_id="r-1"))
        record = executor.get_record("r-1")
        assert isinstance(record, ToolExecutionRecord)
        assert record.tool_call_id == "r-1"
        assert record.tool_name == "ping"

    async def test_all_records_returns_all(self) -> None:
        registry, executor = _make_executor()
        registry.register(lambda: "ok", name="ping", description="ping")
        await executor.execute(_tc("ping", call_id="r-1"))
        await executor.execute(_tc("ping", call_id="r-2"))
        records = executor.all_records()
        assert len(records) == 2

    async def test_record_duration_ms_non_negative(self) -> None:
        registry, executor = _make_executor()
        registry.register(lambda: "ok", name="ping", description="ping")
        await executor.execute(_tc("ping", call_id="r-dur"))
        record = executor.get_record("r-dur")
        assert record is not None
        assert record.duration_ms >= 0

    async def test_record_stores_arguments(self) -> None:
        registry, executor = _make_executor()
        registry.register(lambda x: x, name="echo", description="echo")
        await executor.execute(_tc("echo", {"x": "hello"}, call_id="r-args"))
        record = executor.get_record("r-args")
        assert record is not None
        assert record.arguments == {"x": "hello"}
