"""Performance benchmarks — no Docker/Dapr required. Pure Python overhead checks.

These verify that safety and evaluation components stay within their latency budgets:
  - PromptInjectionDetector.check()  < 5ms p99
  - PIIDetector.scan()               < 5ms p99
  - MemoryValidator.validate()       < 2ms p99
  - contains/not_contains/matches_regex/tool_was_called assertions < 1ms p99 each
"""

from __future__ import annotations

import statistics
import time

import pytest

from nexus.core.types import AgentStatus, ExecutionResult, TokenUsage

_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Python's async/await pattern allows writing concurrent code "
    "that reads like synchronous code. The GIL still applies. "
) * 4


def _p99(times_s: list[float]) -> float:
    return statistics.quantiles(times_s, n=100)[98]


@pytest.mark.benchmark
class TestSafetyOverhead:
    async def test_injection_check_under_5ms(self) -> None:
        from nexus.safety.injection import DetectionLevel, PromptInjectionDetector

        detector = PromptInjectionDetector(level=DetectionLevel.BALANCED)
        times: list[float] = []
        for _ in range(1000):
            t0 = time.perf_counter()
            detector.check(_SAMPLE_TEXT)
            times.append(time.perf_counter() - t0)

        p99_ms = _p99(times) * 1000
        assert p99_ms < 5.0, f"p99 injection check latency {p99_ms:.2f}ms > 5ms budget"

    async def test_pii_scan_under_5ms(self) -> None:
        from nexus.safety.pii import PIIDetector

        detector = PIIDetector()
        times: list[float] = []
        for _ in range(1000):
            t0 = time.perf_counter()
            detector.scan(_SAMPLE_TEXT)
            times.append(time.perf_counter() - t0)

        p99_ms = _p99(times) * 1000
        assert p99_ms < 5.0, f"p99 PII scan latency {p99_ms:.2f}ms > 5ms budget"


@pytest.mark.benchmark
class TestMemoryOverhead:
    async def test_validator_check_under_2ms(self) -> None:
        from nexus.memory.validator import MemoryValidator

        validator = MemoryValidator()
        times: list[float] = []
        for i in range(1000):
            content = f"This is a normal memory entry number {i}."
            t0 = time.perf_counter()
            validator.validate(content, source_id="benchmark-source")
            times.append(time.perf_counter() - t0)

        p99_ms = _p99(times) * 1000
        assert p99_ms < 2.0, f"p99 validator latency {p99_ms:.2f}ms > 2ms budget"


def _make_exec_result(output: str = "test output") -> ExecutionResult:
    from nexus.core.types import Message, Role

    return ExecutionResult(
        output=output,
        messages=[Message(role=Role.ASSISTANT, content=output)],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=10, total_tokens=20, cost_usd=0.001, model="mock"
        ),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


@pytest.mark.benchmark
class TestEvalOverhead:
    async def test_contains_assertion_under_1ms(self) -> None:
        from nexus.evaluation.assertions import contains

        assertion = contains("test")
        result = _make_exec_result()
        times: list[float] = []
        for _ in range(1000):
            t0 = time.perf_counter()
            await assertion(result)
            times.append(time.perf_counter() - t0)

        p99_ms = _p99(times) * 1000
        assert p99_ms < 1.0, f"p99 contains latency {p99_ms:.2f}ms > 1ms budget"

    async def test_not_contains_assertion_under_1ms(self) -> None:
        from nexus.evaluation.assertions import not_contains

        assertion = not_contains("forbidden")
        result = _make_exec_result()
        times: list[float] = []
        for _ in range(1000):
            t0 = time.perf_counter()
            await assertion(result)
            times.append(time.perf_counter() - t0)

        p99_ms = _p99(times) * 1000
        assert p99_ms < 1.0, f"p99 not_contains latency {p99_ms:.2f}ms > 1ms budget"

    async def test_matches_regex_assertion_under_1ms(self) -> None:
        from nexus.evaluation.assertions import matches_regex

        assertion = matches_regex(r"\w+")
        result = _make_exec_result()
        times: list[float] = []
        for _ in range(1000):
            t0 = time.perf_counter()
            await assertion(result)
            times.append(time.perf_counter() - t0)

        p99_ms = _p99(times) * 1000
        assert p99_ms < 1.0, f"p99 matches_regex latency {p99_ms:.2f}ms > 1ms budget"

    async def test_tool_was_called_assertion_under_1ms(self) -> None:
        from nexus.evaluation.assertions import tool_was_called

        assertion = tool_was_called("search")
        result = _make_exec_result()
        times: list[float] = []
        for _ in range(1000):
            t0 = time.perf_counter()
            await assertion(result)
            times.append(time.perf_counter() - t0)

        p99_ms = _p99(times) * 1000
        assert p99_ms < 1.0, f"p99 tool_was_called latency {p99_ms:.2f}ms > 1ms budget"
