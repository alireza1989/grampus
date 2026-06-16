"""E2E scenario: 3-agent sequential crew with shared context."""

from __future__ import annotations

import pytest

from grampus.core.errors import OrchestrationError
from grampus.core.types import AgentDefinition
from grampus.orchestration.crew import Crew, CrewMember, CrewPattern
from tests.integration.conftest import MockModelClient, make_session_id


def _make_crew_member(name: str, response: str) -> CrewMember:
    from grampus.orchestration.runner import AgentRunner, RunnerConfig
    from grampus.tools.executor import ToolExecutor
    from grampus.tools.registry import ToolRegistry

    client = MockModelClient()
    client.add_response(response)
    runner = AgentRunner(
        client,
        ToolExecutor(ToolRegistry(), timeout_seconds=5.0),
        config=RunnerConfig(max_iterations=3, enable_memory=False),
    )
    return CrewMember(
        agent_def=AgentDefinition(
            name=name,
            model="mock-model",
            system_prompt=f"You are the {name}.",
            tools=[],
            max_iterations=3,
            temperature=0.0,
            memory_enabled=False,
            cost_budget_usd=None,
        ),
        runner=runner,
        role=name,
    )


@pytest.mark.integration
class TestMultiAgentCrewE2E:
    async def test_sequential_crew_chains_outputs(self) -> None:
        researcher = _make_crew_member("researcher", "Summary: Python is great for async work.")
        critic = _make_crew_member("critic", "Gap: missing mention of GIL.")
        writer = _make_crew_member("writer", "Final: Python async works around the GIL.")

        crew = Crew(
            [researcher, critic, writer],
            pattern=CrewPattern.SEQUENTIAL,
            session_id=make_session_id(),
        )
        result = await crew.run("Research async Python.")

        assert "researcher" in result.outputs
        assert "critic" in result.outputs
        assert "writer" in result.outputs
        assert result.pattern == CrewPattern.SEQUENTIAL

    async def test_parallel_crew_all_agents_run(self) -> None:
        members = [_make_crew_member(f"worker{i}", f"Worker {i} output.") for i in range(3)]
        crew = Crew(
            members,
            pattern=CrewPattern.PARALLEL,
            session_id=make_session_id(),
        )
        result = await crew.run("Work in parallel.")
        assert len(result.outputs) == 3
        for i in range(3):
            assert f"worker{i}" in result.outputs

    async def test_parallel_crew_produces_all_outputs(self) -> None:
        members = [
            _make_crew_member("alpha", "Alpha done."),
            _make_crew_member("beta", "Beta done."),
            _make_crew_member("gamma", "Gamma done."),
        ]
        crew = Crew(members, pattern=CrewPattern.PARALLEL, session_id=make_session_id())
        result = await crew.run("Process all.")
        assert result.outputs.get("alpha") is not None
        assert result.outputs.get("beta") is not None
        assert result.outputs.get("gamma") is not None

    async def test_crew_member_failure_does_not_silently_swallow(self) -> None:
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor
        from grampus.tools.registry import ToolRegistry

        class _FailClient:
            async def complete(self, **_: object) -> None:
                raise RuntimeError("simulated agent crash")

            def stream(self, **_: object) -> None:
                raise NotImplementedError

        good = _make_crew_member("good-agent", "OK output.")
        fail_runner = AgentRunner(
            _FailClient(),  # type: ignore[arg-type]
            ToolExecutor(ToolRegistry(), timeout_seconds=5.0),
            config=RunnerConfig(max_iterations=2, enable_memory=False),
        )
        fail_member = CrewMember(
            agent_def=AgentDefinition(
                name="fail-agent",
                model="mock",
                system_prompt="",
                tools=[],
                max_iterations=2,
                temperature=0.0,
                memory_enabled=False,
                cost_budget_usd=None,
            ),
            runner=fail_runner,
            role="worker",
        )

        crew = Crew(
            [good, fail_member],
            pattern=CrewPattern.SEQUENTIAL,
            session_id=make_session_id(),
        )
        with pytest.raises(OrchestrationError) as exc_info:
            await crew.run("Trigger failure.")
        assert exc_info.value.code == "CREW_MEMBER_FAILED"

    async def test_crew_result_has_correct_pattern(self) -> None:
        members = [_make_crew_member(f"m{i}", f"out {i}") for i in range(2)]
        crew = Crew(members, pattern=CrewPattern.SEQUENTIAL, session_id=make_session_id())
        result = await crew.run("Simple crew run.")
        assert result.pattern == CrewPattern.SEQUENTIAL
        assert result.duration_seconds >= 0.0
        assert result.total_cost_usd >= 0.0
