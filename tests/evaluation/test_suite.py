"""Tests for EvalSuite and related types."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.types import AgentDefinition, AgentStatus, ExecutionResult, TokenUsage


def _make_execution_result(
    output: str = "ok",
    status: AgentStatus = AgentStatus.COMPLETED,
    cost_usd: float = 0.001,
) -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            cost_usd=cost_usd,
            model="test-model",
        ),
        duration_seconds=0.1,
        steps_taken=1,
        status=status,
    )


def _make_runner(output: str = "ok", raises: Exception | None = None) -> Any:
    runner = MagicMock()
    if raises:
        runner.run = AsyncMock(side_effect=raises)
    else:
        runner.run = AsyncMock(return_value=_make_execution_result(output=output))
    return runner


def _make_agent_def() -> AgentDefinition:
    return AgentDefinition(
        name="test-agent",
        model="claude-3-5-sonnet",
        system_prompt="You are a test agent",
    )


class TestEvalSuiteBasic:
    @pytest.mark.asyncio
    async def test_add_case_returns_self(self) -> None:
        from grampus.evaluation.suite import EvalCase, EvalSuite

        suite = EvalSuite(
            "test",
            agent_runner=_make_runner(),
            agent_def=_make_agent_def(),
        )
        case = EvalCase(name="case1", input="hello")
        result = suite.add_case(case)
        assert result is suite

    @pytest.mark.asyncio
    async def test_run_empty_suite_returns_zero_results(self) -> None:
        from grampus.evaluation.suite import EvalSuite

        suite = EvalSuite(
            "empty",
            agent_runner=_make_runner(),
            agent_def=_make_agent_def(),
        )
        sr = await suite.run()
        assert sr.total_cases == 0
        assert sr.passed == 0

    @pytest.mark.asyncio
    async def test_run_single_case_passing(self) -> None:
        from grampus.evaluation.assertions import contains
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner(output="Hello world")
        suite = EvalSuite(
            "suite1",
            agent_runner=runner,
            agent_def=_make_agent_def(),
        )
        case = EvalCase(name="c1", input="hi", assertions=[contains("Hello")])
        suite.add_case(case)
        sr = await suite.run()
        assert sr.total_cases == 1
        assert sr.passed == 1
        assert sr.failed == 0
        assert sr.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_single_case_failing(self) -> None:
        from grampus.evaluation.assertions import contains
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner(output="Hello world")
        suite = EvalSuite(
            "suite1",
            agent_runner=runner,
            agent_def=_make_agent_def(),
        )
        case = EvalCase(name="c1", input="hi", assertions=[contains("MISSING")])
        suite.add_case(case)
        sr = await suite.run()
        assert sr.passed == 0
        assert sr.failed == 1

    @pytest.mark.asyncio
    async def test_suite_result_pass_rate_computed(self) -> None:
        from grampus.evaluation.assertions import contains
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner(output="Hello world")
        suite = EvalSuite(
            "suite1",
            agent_runner=runner,
            agent_def=_make_agent_def(),
        )
        suite.add_cases(
            [
                EvalCase(name="c1", input="q1", assertions=[contains("Hello")]),
                EvalCase(name="c2", input="q2", assertions=[contains("MISSING")]),
            ]
        )
        sr = await suite.run()
        assert sr.pass_rate == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_suite_result_totals_correct(self) -> None:
        from grampus.evaluation.assertions import contains
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner(output="Hello world")
        suite = EvalSuite(
            "suite1",
            agent_runner=runner,
            agent_def=_make_agent_def(),
        )
        suite.add_cases(
            [
                EvalCase(name="c1", input="q", assertions=[contains("Hello")]),
                EvalCase(name="c2", input="q", assertions=[contains("Hello")]),
                EvalCase(name="c3", input="q", assertions=[contains("MISSING")]),
            ]
        )
        sr = await suite.run()
        assert sr.total_cases == 3
        assert sr.passed == 2
        assert sr.failed == 1


class TestEvalSuiteFiltering:
    @pytest.mark.asyncio
    async def test_tag_filter_runs_only_matching_cases(self) -> None:
        from grampus.evaluation.assertions import contains
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner(output="ok")
        suite = EvalSuite(
            "suite",
            agent_runner=runner,
            agent_def=_make_agent_def(),
            tags=["smoke"],
        )
        suite.add_cases(
            [
                EvalCase(name="c1", input="q", tags=["smoke"], assertions=[contains("ok")]),
                EvalCase(name="c2", input="q", tags=["regression"], assertions=[contains("ok")]),
            ]
        )
        sr = await suite.run()
        assert sr.total_cases == 1

    @pytest.mark.asyncio
    async def test_tag_filter_empty_intersection_skips_all(self) -> None:
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner()
        suite = EvalSuite(
            "suite",
            agent_runner=runner,
            agent_def=_make_agent_def(),
            tags=["nonexistent"],
        )
        suite.add_cases(
            [
                EvalCase(name="c1", input="q", tags=["smoke"]),
                EvalCase(name="c2", input="q", tags=["regression"]),
            ]
        )
        sr = await suite.run()
        assert sr.total_cases == 0

    @pytest.mark.asyncio
    async def test_no_tag_filter_runs_all(self) -> None:
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner()
        suite = EvalSuite(
            "suite",
            agent_runner=runner,
            agent_def=_make_agent_def(),
        )
        suite.add_cases(
            [
                EvalCase(name="c1", input="q", tags=["smoke"]),
                EvalCase(name="c2", input="q", tags=["regression"]),
            ]
        )
        sr = await suite.run()
        assert sr.total_cases == 2


class TestEvalSuiteConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_1_runs_sequentially(self) -> None:
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner()
        suite = EvalSuite(
            "suite",
            agent_runner=runner,
            agent_def=_make_agent_def(),
            concurrency=1,
        )
        suite.add_cases([EvalCase(name=f"c{i}", input="q") for i in range(3)])
        sr = await suite.run()
        assert sr.total_cases == 3

    @pytest.mark.asyncio
    async def test_concurrency_n_runs_all_cases(self) -> None:
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner()
        suite = EvalSuite(
            "suite",
            agent_runner=runner,
            agent_def=_make_agent_def(),
            concurrency=4,
        )
        suite.add_cases([EvalCase(name=f"c{i}", input="q") for i in range(5)])
        sr = await suite.run()
        assert sr.total_cases == 5


class TestEvalSuiteErrors:
    @pytest.mark.asyncio
    async def test_agent_exception_captured_in_case_result(self) -> None:
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner(raises=RuntimeError("agent exploded"))
        suite = EvalSuite(
            "suite",
            agent_runner=runner,
            agent_def=_make_agent_def(),
        )
        suite.add_case(EvalCase(name="boom", input="q"))
        sr = await suite.run()
        assert sr.case_results[0].error is not None
        assert "agent exploded" in sr.case_results[0].error

    @pytest.mark.asyncio
    async def test_failed_case_does_not_abort_suite(self) -> None:
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner(raises=RuntimeError("boom"))
        suite = EvalSuite(
            "suite",
            agent_runner=runner,
            agent_def=_make_agent_def(),
        )
        suite.add_cases(
            [
                EvalCase(name="c1", input="q"),
                EvalCase(name="c2", input="q"),
            ]
        )
        sr = await suite.run()
        assert sr.total_cases == 2

    @pytest.mark.asyncio
    async def test_error_counted_in_suite_result(self) -> None:
        from grampus.evaluation.suite import EvalCase, EvalSuite

        runner = _make_runner(raises=RuntimeError("boom"))
        suite = EvalSuite(
            "suite",
            agent_runner=runner,
            agent_def=_make_agent_def(),
        )
        suite.add_case(EvalCase(name="c1", input="q"))
        sr = await suite.run()
        assert sr.errors == 1
