"""Tests for streaming evaluation assertions and StreamingEvalSuite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from grampus.core.types import (
    StreamChunk,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from grampus.evaluation.streaming import (
    StreamingEvalCase,
    StreamingEvalSuite,
    StreamingResult,
    chunk_count_between,
    first_token_within,
    min_throughput,
    no_repetition,
    no_stall,
    stream_completes,
    stream_contains,
    stream_not_empty,
    stream_output_length,
    token_usage_reported,
)
from grampus.evaluation.suite import CaseResult, SuiteResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_result(
    *,
    first_token_seconds: float | None = 0.2,
    total_seconds: float = 1.0,
    inter_chunk_delays: list[float] | None = None,
    full_output: str = "Hello world",
    chunk_count: int | None = None,
    chunk_deltas: list[str] | None = None,
    token_usage: TokenUsage | None = None,
    agent_ended: bool = True,
    tool_calls_made: int = 0,
    error: str | None = None,
    events: list[StreamEvent] | None = None,
) -> StreamingResult:
    deltas = chunk_deltas or list(full_output)
    return StreamingResult(
        first_token_seconds=first_token_seconds,
        total_seconds=total_seconds,
        inter_chunk_delays=inter_chunk_delays or [],
        full_output=full_output,
        output_length=len(full_output),
        chunk_count=chunk_count if chunk_count is not None else len(deltas),
        chunk_deltas=deltas,
        token_usage=token_usage,
        agent_ended=agent_ended,
        tool_calls_made=tool_calls_made,
        error=error,
        events=events or [],
    )


def _make_mock_stream(deltas: list[str], *, error: Exception | None = None) -> Any:
    """Returns an async generator that yields controlled StreamEvents."""

    async def _gen(*args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(event_type=StreamEventType.AGENT_START, message="test")
        yield StreamEvent(event_type=StreamEventType.ITERATION_START, iteration=1)
        for d in deltas:
            yield StreamEvent(
                event_type=StreamEventType.TOKEN,
                chunk=StreamChunk(delta=d, model="test"),
            )
        if error:
            raise error
        yield StreamEvent(
            event_type=StreamEventType.AGENT_END,
            chunk=StreamChunk(
                is_final=True,
                token_usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    cost_usd=0.0,
                    model="test",
                ),
            ),
        )

    return _gen


# ---------------------------------------------------------------------------
# TestStreamingResult
# ---------------------------------------------------------------------------


class TestStreamingResult:
    def test_output_length_auto_computed(self) -> None:
        result = _make_stream_result(full_output="hello")
        assert result.output_length == 5

    def test_error_field_set(self) -> None:
        result = _make_stream_result(error="boom")
        assert result.error == "boom"


# ---------------------------------------------------------------------------
# TestCollectStream
# ---------------------------------------------------------------------------


class TestCollectStream:
    @pytest.fixture
    def runner(self) -> MagicMock:
        mock = MagicMock()
        mock.stream = _make_mock_stream(["Hello", " world"])
        return mock

    @pytest.fixture
    def agent_def(self) -> Any:
        from grampus.core.types import AgentDefinition

        return AgentDefinition(name="test", model="test-model", system_prompt="be helpful")

    @pytest.mark.asyncio
    async def test_collects_full_output(self, runner: MagicMock, agent_def: Any) -> None:
        from grampus.evaluation.streaming import _collect_stream

        result = await _collect_stream(runner, agent_def, "hi", "session-1")
        assert result.full_output == "Hello world"

    @pytest.mark.asyncio
    async def test_chunk_count(self, runner: MagicMock, agent_def: Any) -> None:
        from grampus.evaluation.streaming import _collect_stream

        result = await _collect_stream(runner, agent_def, "hi", "session-1")
        assert result.chunk_count == 2

    @pytest.mark.asyncio
    async def test_agent_ended_true(self, runner: MagicMock, agent_def: Any) -> None:
        from grampus.evaluation.streaming import _collect_stream

        result = await _collect_stream(runner, agent_def, "hi", "session-1")
        assert result.agent_ended is True

    @pytest.mark.asyncio
    async def test_first_token_seconds_populated(self, runner: MagicMock, agent_def: Any) -> None:
        from grampus.evaluation.streaming import _collect_stream

        result = await _collect_stream(runner, agent_def, "hi", "session-1")
        assert result.first_token_seconds is not None
        assert result.first_token_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_token_usage_populated(self, runner: MagicMock, agent_def: Any) -> None:
        from grampus.evaluation.streaming import _collect_stream

        result = await _collect_stream(runner, agent_def, "hi", "session-1")
        assert result.token_usage is not None
        assert result.token_usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_tool_calls_counted(self, agent_def: Any) -> None:
        from grampus.evaluation.streaming import _collect_stream

        async def _gen_with_tools(*args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(event_type=StreamEventType.AGENT_START)
            yield StreamEvent(event_type=StreamEventType.TOOL_CALL_START)
            yield StreamEvent(event_type=StreamEventType.TOOL_CALL_END)
            yield StreamEvent(event_type=StreamEventType.TOOL_CALL_START)
            yield StreamEvent(event_type=StreamEventType.TOOL_CALL_END)
            yield StreamEvent(
                event_type=StreamEventType.AGENT_END,
                chunk=StreamChunk(is_final=True),
            )

        runner = MagicMock()
        runner.stream = _gen_with_tools
        result = await _collect_stream(runner, agent_def, "hi", "session-1")
        assert result.tool_calls_made == 2

    @pytest.mark.asyncio
    async def test_exception_captured_as_error(self, agent_def: Any) -> None:
        from grampus.evaluation.streaming import _collect_stream

        runner = MagicMock()
        runner.stream = _make_mock_stream(["partial"], error=RuntimeError("network timeout"))
        result = await _collect_stream(runner, agent_def, "hi", "session-1")
        assert result.error is not None
        assert "network timeout" in result.error


# ---------------------------------------------------------------------------
# TestFirstTokenWithin
# ---------------------------------------------------------------------------


class TestFirstTokenWithin:
    @pytest.mark.asyncio
    async def test_pass_within_limit(self) -> None:
        result = _make_stream_result(first_token_seconds=0.5)
        ar = await first_token_within(1.0)(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_exceeds_limit(self) -> None:
        result = _make_stream_result(first_token_seconds=2.0)
        ar = await first_token_within(1.0)(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_fail_no_tokens(self) -> None:
        result = _make_stream_result(first_token_seconds=None)
        ar = await first_token_within(1.0)(result)
        assert ar.passed is False
        assert "unmeasured" in ar.detail.lower() or "no tokens" in ar.detail.lower()

    @pytest.mark.asyncio
    async def test_assertion_type(self) -> None:
        result = _make_stream_result()
        ar = await first_token_within(1.0)(result)
        assert ar.assertion_type == "first_token_within"


# ---------------------------------------------------------------------------
# TestStreamCompletes
# ---------------------------------------------------------------------------


class TestStreamCompletes:
    @pytest.mark.asyncio
    async def test_pass_when_agent_ended(self) -> None:
        result = _make_stream_result(agent_ended=True, error=None)
        ar = await stream_completes()(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_when_not_ended(self) -> None:
        result = _make_stream_result(agent_ended=False, error=None)
        ar = await stream_completes()(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_fail_when_error(self) -> None:
        result = _make_stream_result(agent_ended=True, error="timeout")
        ar = await stream_completes()(result)
        assert ar.passed is False


# ---------------------------------------------------------------------------
# TestNoStall
# ---------------------------------------------------------------------------


class TestNoStall:
    @pytest.mark.asyncio
    async def test_pass_no_delays(self) -> None:
        result = _make_stream_result(inter_chunk_delays=[])
        ar = await no_stall(2.0)(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_pass_all_within_limit(self) -> None:
        result = _make_stream_result(inter_chunk_delays=[0.1, 0.2, 0.15])
        ar = await no_stall(1.0)(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_one_exceeds_limit(self) -> None:
        result = _make_stream_result(inter_chunk_delays=[0.1, 3.0, 0.1])
        ar = await no_stall(2.0)(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_detail_includes_chunk_index(self) -> None:
        result = _make_stream_result(inter_chunk_delays=[0.1, 3.0, 0.1])
        ar = await no_stall(2.0)(result)
        assert "1" in ar.detail  # index 1 is the stall


# ---------------------------------------------------------------------------
# TestMinThroughput
# ---------------------------------------------------------------------------


class TestMinThroughput:
    @pytest.mark.asyncio
    async def test_pass(self) -> None:
        result = _make_stream_result(
            full_output="a" * 10,
            chunk_count=10,
            chunk_deltas=list("a" * 10),
            total_seconds=1.0,
        )
        ar = await min_throughput(5.0)(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail(self) -> None:
        result = _make_stream_result(
            full_output="ab",
            chunk_count=2,
            chunk_deltas=["a", "b"],
            total_seconds=5.0,
        )
        ar = await min_throughput(5.0)(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_fail_no_chunks(self) -> None:
        result = _make_stream_result(
            full_output="",
            chunk_count=0,
            chunk_deltas=[],
            total_seconds=1.0,
        )
        ar = await min_throughput(5.0)(result)
        assert ar.passed is False


# ---------------------------------------------------------------------------
# TestStreamContains
# ---------------------------------------------------------------------------


class TestStreamContains:
    @pytest.mark.asyncio
    async def test_pass_found(self) -> None:
        result = _make_stream_result(full_output="Hello world")
        ar = await stream_contains("world")(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_not_found(self) -> None:
        result = _make_stream_result(full_output="Hello world")
        ar = await stream_contains("NOTHERE")(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_case_insensitive(self) -> None:
        result = _make_stream_result(full_output="Hello world")
        ar = await stream_contains("WORLD", case_sensitive=False)(result)
        assert ar.passed is True


# ---------------------------------------------------------------------------
# TestStreamNotEmpty
# ---------------------------------------------------------------------------


class TestStreamNotEmpty:
    @pytest.mark.asyncio
    async def test_pass_has_chunks(self) -> None:
        result = _make_stream_result(full_output="hi", chunk_count=2)
        ar = await stream_not_empty()(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_zero_chunks(self) -> None:
        result = _make_stream_result(full_output="", chunk_count=0, chunk_deltas=[])
        ar = await stream_not_empty()(result)
        assert ar.passed is False


# ---------------------------------------------------------------------------
# TestStreamOutputLength
# ---------------------------------------------------------------------------


class TestStreamOutputLength:
    @pytest.mark.asyncio
    async def test_pass_in_range(self) -> None:
        result = _make_stream_result(full_output="hello")
        ar = await stream_output_length(min_chars=3, max_chars=10)(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_too_short(self) -> None:
        result = _make_stream_result(full_output="hi")
        ar = await stream_output_length(min_chars=5)(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_fail_too_long(self) -> None:
        result = _make_stream_result(full_output="hello world")
        ar = await stream_output_length(max_chars=5)(result)
        assert ar.passed is False


# ---------------------------------------------------------------------------
# TestNoRepetition
# ---------------------------------------------------------------------------


class TestNoRepetition:
    @pytest.mark.asyncio
    async def test_pass_no_repetition(self) -> None:
        output = "The quick brown fox jumps over the lazy dog " * 5
        result = _make_stream_result(full_output=output)
        ar = await no_repetition(window=20, threshold=0.8)(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_repeated_output(self) -> None:
        output = "abcdefghij" * 50
        result = _make_stream_result(full_output=output)
        ar = await no_repetition(window=50, threshold=0.8)(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_pass_too_short(self) -> None:
        result = _make_stream_result(full_output="short")
        ar = await no_repetition(window=100, threshold=0.8)(result)
        assert ar.passed is True
        assert "too short" in ar.detail.lower()

    @pytest.mark.asyncio
    async def test_ratio_edge_identical(self) -> None:
        segment = "x" * 100
        output = segment + segment
        result = _make_stream_result(full_output=output)
        ar = await no_repetition(window=100, threshold=0.8)(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_window_size_respected(self) -> None:
        # First 200 chars non-repeating, last 200 chars repeating
        normal = "abcdefghij" * 10  # 100 chars
        repeating = "z" * 100
        output = normal + repeating + repeating
        result = _make_stream_result(full_output=output)
        ar = await no_repetition(window=100, threshold=0.8)(result)
        assert ar.passed is False


# ---------------------------------------------------------------------------
# TestChunkCountBetween
# ---------------------------------------------------------------------------


class TestChunkCountBetween:
    @pytest.mark.asyncio
    async def test_pass_in_range(self) -> None:
        result = _make_stream_result(full_output="hello", chunk_count=5)
        ar = await chunk_count_between(min_chunks=3, max_chunks=10)(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_too_few(self) -> None:
        result = _make_stream_result(full_output="hi", chunk_count=2)
        ar = await chunk_count_between(min_chunks=5)(result)
        assert ar.passed is False

    @pytest.mark.asyncio
    async def test_fail_too_many(self) -> None:
        result = _make_stream_result(full_output="hello world", chunk_count=20)
        ar = await chunk_count_between(max_chunks=5)(result)
        assert ar.passed is False


# ---------------------------------------------------------------------------
# TestTokenUsageReported
# ---------------------------------------------------------------------------


class TestTokenUsageReported:
    @pytest.mark.asyncio
    async def test_pass_with_usage(self) -> None:
        usage = TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="test"
        )
        result = _make_stream_result(token_usage=usage)
        ar = await token_usage_reported()(result)
        assert ar.passed is True

    @pytest.mark.asyncio
    async def test_fail_without_usage(self) -> None:
        result = _make_stream_result(token_usage=None)
        ar = await token_usage_reported()(result)
        assert ar.passed is False


# ---------------------------------------------------------------------------
# TestStreamingEvalSuite
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_def() -> Any:
    from grampus.core.types import AgentDefinition

    return AgentDefinition(name="test", model="test-model", system_prompt="be helpful")


@pytest.fixture
def mock_runner() -> MagicMock:
    runner = MagicMock()
    runner.stream = _make_mock_stream(["Hello", " world"])
    return runner


class TestStreamingEvalSuite:
    @pytest.mark.asyncio
    async def test_run_single_case_passes(self, mock_runner: MagicMock, agent_def: Any) -> None:
        case = StreamingEvalCase(
            name="basic",
            input="hi",
            assertions=[stream_not_empty()],
        )
        suite = StreamingEvalSuite("test-suite", agent_runner=mock_runner, agent_def=agent_def)
        suite.add_case(case)
        result = await suite.run()
        assert result.passed == 1
        assert result.case_results[0].passed is True

    @pytest.mark.asyncio
    async def test_run_single_case_fails(self, mock_runner: MagicMock, agent_def: Any) -> None:
        case = StreamingEvalCase(
            name="basic",
            input="hi",
            assertions=[stream_contains("NOTHERE")],
        )
        suite = StreamingEvalSuite("test-suite", agent_runner=mock_runner, agent_def=agent_def)
        suite.add_case(case)
        result = await suite.run()
        assert result.failed == 1
        assert result.case_results[0].passed is False

    @pytest.mark.asyncio
    async def test_suite_result_type(self, mock_runner: MagicMock, agent_def: Any) -> None:
        suite = StreamingEvalSuite("test-suite", agent_runner=mock_runner, agent_def=agent_def)
        suite.add_case(StreamingEvalCase(name="c", input="hi"))
        result = await suite.run()
        assert isinstance(result, SuiteResult)

    @pytest.mark.asyncio
    async def test_case_result_has_streaming_result(
        self, mock_runner: MagicMock, agent_def: Any
    ) -> None:
        suite = StreamingEvalSuite("test-suite", agent_runner=mock_runner, agent_def=agent_def)
        suite.add_case(StreamingEvalCase(name="c", input="hi"))
        result = await suite.run()
        cr = result.case_results[0]
        assert isinstance(cr, CaseResult)
        assert cr.streaming_result is not None
        assert isinstance(cr.streaming_result, StreamingResult)

    @pytest.mark.asyncio
    async def test_case_result_has_no_execution_result(
        self, mock_runner: MagicMock, agent_def: Any
    ) -> None:
        suite = StreamingEvalSuite("test-suite", agent_runner=mock_runner, agent_def=agent_def)
        suite.add_case(StreamingEvalCase(name="c", input="hi"))
        result = await suite.run()
        assert result.case_results[0].execution_result is None

    @pytest.mark.asyncio
    async def test_error_captured(self, agent_def: Any) -> None:
        runner = MagicMock()
        runner.stream = _make_mock_stream([], error=RuntimeError("stream broke"))
        case = StreamingEvalCase(name="err", input="hi")
        suite = StreamingEvalSuite("test", agent_runner=runner, agent_def=agent_def)
        suite.add_case(case)
        result = await suite.run()
        cr = result.case_results[0]
        assert cr.error is not None
        assert cr.passed is False

    @pytest.mark.asyncio
    async def test_tag_filtering(self, mock_runner: MagicMock, agent_def: Any) -> None:
        smoke_case = StreamingEvalCase(name="smoke", input="hi", tags=["smoke"])
        other_case = StreamingEvalCase(name="other", input="hi", tags=["regression"])
        suite = StreamingEvalSuite(
            "test",
            agent_runner=mock_runner,
            agent_def=agent_def,
            tags=["smoke"],
        )
        suite.add_cases([smoke_case, other_case])
        result = await suite.run()
        assert result.total_cases == 1
        assert result.case_results[0].case_name == "smoke"

    @pytest.mark.asyncio
    async def test_concurrency(self, agent_def: Any) -> None:
        runner = MagicMock()
        runner.stream = _make_mock_stream(["ok"])
        cases = [StreamingEvalCase(name=f"c{i}", input="hi") for i in range(4)]
        suite = StreamingEvalSuite("test", agent_runner=runner, agent_def=agent_def, concurrency=2)
        suite.add_cases(cases)
        result = await suite.run()
        assert result.total_cases == 4
        assert result.passed == 4

    @pytest.mark.asyncio
    async def test_add_cases_chaining(self, mock_runner: MagicMock, agent_def: Any) -> None:
        suite = StreamingEvalSuite("test", agent_runner=mock_runner, agent_def=agent_def)
        returned = suite.add_case(StreamingEvalCase(name="a", input="hi")).add_case(
            StreamingEvalCase(name="b", input="bye")
        )
        assert returned is suite
        result = await suite.run()
        assert result.total_cases == 2

    @pytest.mark.asyncio
    async def test_empty_suite_result(self, mock_runner: MagicMock, agent_def: Any) -> None:
        suite = StreamingEvalSuite("empty", agent_runner=mock_runner, agent_def=agent_def)
        result = await suite.run()
        assert result.total_cases == 0
