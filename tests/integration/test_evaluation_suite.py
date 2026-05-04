"""Integration tests for EvalSuite: assertions, baselines, concurrency."""

from __future__ import annotations

import pytest

from nexus.core.types import AgentDefinition, AgentStatus
from nexus.evaluation.assertions import (
    contains,
    max_cost,
    not_contains,
    status_is,
    tool_was_called,
)
from nexus.evaluation.suite import EvalCase, EvalSuite
from tests.integration.conftest import MockModelClient


def _agent_def(name: str = "eval-agent") -> AgentDefinition:
    return AgentDefinition(
        name=name,
        model="mock-model",
        system_prompt="You are helpful.",
        tools=[],
        max_iterations=3,
        temperature=0.0,
        memory_enabled=False,
        cost_budget_usd=None,
    )


def _make_runner(response: str = "Done.", cost: float = 0.001) -> object:
    from nexus.orchestration.runner import AgentRunner, RunnerConfig
    from nexus.tools.executor import ToolExecutor
    from nexus.tools.registry import ToolRegistry

    client = MockModelClient()
    client.add_response(response, cost_usd=cost)
    runner = AgentRunner(
        client,
        ToolExecutor(ToolRegistry(), timeout_seconds=5.0),
        config=RunnerConfig(max_iterations=3, enable_memory=False),
    )
    return runner


@pytest.mark.integration
class TestEvalSuiteIntegration:
    async def test_suite_runs_all_cases_against_mock_runner(self) -> None:
        runner = _make_runner("I know Python.")
        suite = EvalSuite("test-suite", agent_runner=runner, agent_def=_agent_def())
        suite.add_cases([
            EvalCase(name="case1", input="Do you know Python?"),
            EvalCase(name="case2", input="What about Go?"),
        ])
        result = await suite.run()
        assert result.total_cases == 2

    async def test_contains_assertion_passes_on_matching_output(self) -> None:
        runner = _make_runner("Python uses async/await.")
        suite = EvalSuite("contains-suite", agent_runner=runner, agent_def=_agent_def())
        suite.add_case(
            EvalCase(
                name="contains-test",
                input="Describe async Python.",
                assertions=[contains("async")],
            )
        )
        result = await suite.run()
        assert result.passed == 1
        assert result.case_results[0].assertion_results[0].passed

    async def test_not_contains_assertion_passes_when_absent(self) -> None:
        runner = _make_runner("Python is great.")
        suite = EvalSuite("not-contains-suite", agent_runner=runner, agent_def=_agent_def())
        suite.add_case(
            EvalCase(
                name="nc-test",
                input="Describe Python.",
                assertions=[not_contains("Ruby")],
            )
        )
        result = await suite.run()
        assert result.passed == 1

    async def test_tool_was_called_assertion_passes(self) -> None:
        from nexus.core.types import ToolCall, ToolParameter
        from nexus.orchestration.runner import AgentRunner, RunnerConfig
        from nexus.tools.executor import ToolExecutor
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry()

        @registry.tool(
            name="search",
            description="Search",
            parameters=[ToolParameter(name="query", type="string", description="", required=True)],
        )
        async def search(query: str) -> str:
            return f"Results for {query}"

        client = MockModelClient()
        client.add_response(
            text=None,
            tool_calls=[ToolCall(id="tc-s", name="search", arguments={"query": "python"})],
        )
        client.add_response("Found it.")

        runner = AgentRunner(
            client,
            ToolExecutor(registry, timeout_seconds=5.0),
            config=RunnerConfig(max_iterations=5, enable_memory=False),
        )
        suite = EvalSuite("tool-suite", agent_runner=runner, agent_def=_agent_def())
        suite.add_case(
            EvalCase(
                name="tool-test",
                input="Search for Python.",
                assertions=[tool_was_called("search")],
            )
        )
        result = await suite.run()
        assert result.passed == 1

    async def test_max_cost_assertion_fails_when_exceeded(self) -> None:
        runner = _make_runner("Result.", cost=0.5)
        suite = EvalSuite("cost-suite", agent_runner=runner, agent_def=_agent_def())
        suite.add_case(
            EvalCase(
                name="cost-test",
                input="Do something.",
                assertions=[max_cost(0.001)],
            )
        )
        result = await suite.run()
        assert result.failed == 1

    async def test_tag_filtering_runs_subset(self) -> None:
        runner = _make_runner("OK")
        suite = EvalSuite(
            "tag-suite",
            agent_runner=runner,
            agent_def=_agent_def(),
            tags=["smoke"],
        )
        suite.add_case(EvalCase(name="tagged", input="Test.", tags=["smoke"]))
        suite.add_case(EvalCase(name="untagged", input="Other.", tags=["regression"]))
        result = await suite.run()
        assert result.total_cases == 1
        assert result.case_results[0].case_name == "tagged"

    async def test_concurrent_cases_all_complete(self) -> None:
        from nexus.orchestration.runner import AgentRunner, RunnerConfig
        from nexus.tools.executor import ToolExecutor
        from nexus.tools.registry import ToolRegistry

        client = MockModelClient(default_text="Answer.")
        runner = AgentRunner(
            client,
            ToolExecutor(ToolRegistry(), timeout_seconds=5.0),
            config=RunnerConfig(max_iterations=2, enable_memory=False),
        )
        suite = EvalSuite(
            "concurrent-suite",
            agent_runner=runner,
            agent_def=_agent_def(),
            concurrency=3,
        )
        for i in range(6):
            suite.add_case(EvalCase(name=f"c{i}", input=f"Question {i}?"))
        result = await suite.run()
        assert result.total_cases == 6

    async def test_regression_detected_against_baseline(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        baseline = QualityBaseline(suite_name="my-suite")
        baseline.record(pass_rate=0.9, avg_cost=0.001, avg_duration=1.0)

        regression = baseline.check_regression(pass_rate=0.7, threshold=0.1)
        assert regression is True

    async def test_no_regression_when_pass_rate_stable(self) -> None:
        from nexus.evaluation.baseline import QualityBaseline

        baseline = QualityBaseline(suite_name="my-suite")
        baseline.record(pass_rate=0.9, avg_cost=0.001, avg_duration=1.0)

        regression = baseline.check_regression(pass_rate=0.88, threshold=0.1)
        assert regression is False

    async def test_status_is_completed_assertion(self) -> None:
        runner = _make_runner("All done.")
        suite = EvalSuite("status-suite", agent_runner=runner, agent_def=_agent_def())
        suite.add_case(
            EvalCase(
                name="status-test",
                input="Finish.",
                assertions=[status_is(AgentStatus.COMPLETED)],
            )
        )
        result = await suite.run()
        assert result.passed == 1
