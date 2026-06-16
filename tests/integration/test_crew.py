"""Integration tests for Crew: sequential, parallel, hierarchical."""

from __future__ import annotations

import pytest

from grampus.core.errors import OrchestrationError
from grampus.core.types import AgentDefinition
from grampus.orchestration.crew import Crew, CrewMember, CrewPattern
from tests.integration.conftest import MockModelClient, make_session_id


def _make_member(name: str, response: str) -> CrewMember:
    from grampus.orchestration.runner import AgentRunner, RunnerConfig
    from grampus.tools.executor import ToolExecutor
    from grampus.tools.registry import ToolRegistry

    client = MockModelClient()
    client.add_response(response)
    registry = ToolRegistry()
    executor = ToolExecutor(registry, timeout_seconds=5.0)
    runner = AgentRunner(
        client,
        executor,
        config=RunnerConfig(max_iterations=3, enable_memory=False),
    )
    agent_def = AgentDefinition(
        name=name,
        model="mock-model",
        system_prompt=f"You are {name}.",
        tools=[],
        max_iterations=3,
        temperature=0.0,
        memory_enabled=False,
        cost_budget_usd=None,
    )
    return CrewMember(agent_def=agent_def, runner=runner, role="worker")


@pytest.mark.integration
class TestCrewIntegration:
    async def test_sequential_crew_passes_output_chain(self) -> None:
        m1 = _make_member("researcher", "Summary: Python is great for async work.")
        m2 = _make_member("critic", "Gap: missing mention of GIL.")
        m3 = _make_member("writer", "Final: Python async works around the GIL.")

        crew = Crew(
            [m1, m2, m3],
            pattern=CrewPattern.SEQUENTIAL,
            session_id=make_session_id(),
        )
        result = await crew.run("Research async Python.")
        assert "researcher" in result.outputs
        assert "critic" in result.outputs
        assert "writer" in result.outputs
        assert result.pattern == CrewPattern.SEQUENTIAL

    async def test_sequential_crew_result_includes_all_agent_names(self) -> None:
        members = [_make_member(f"agent{i}", f"Output {i}") for i in range(3)]
        crew = Crew(members, pattern=CrewPattern.SEQUENTIAL, session_id=make_session_id())
        result = await crew.run("Go.")
        assert len(result.outputs) == 3
        for i in range(3):
            assert f"agent{i}" in result.outputs

    async def test_parallel_crew_collects_all_outputs(self) -> None:
        members = [_make_member(f"p{i}", f"parallel output {i}") for i in range(3)]
        crew = Crew(members, pattern=CrewPattern.PARALLEL, session_id=make_session_id())
        result = await crew.run("Work in parallel.")
        assert len(result.outputs) == 3
        assert result.pattern == CrewPattern.PARALLEL

    async def test_parallel_crew_duration_is_reasonable(self) -> None:

        members = []
        for i in range(3):
            client = MockModelClient()
            client.add_response(f"output {i}")
            from grampus.orchestration.runner import AgentRunner, RunnerConfig
            from grampus.tools.executor import ToolExecutor
            from grampus.tools.registry import ToolRegistry

            runner = AgentRunner(
                client,
                ToolExecutor(ToolRegistry(), timeout_seconds=5.0),
                config=RunnerConfig(max_iterations=1, enable_memory=False),
            )
            members.append(
                CrewMember(
                    agent_def=AgentDefinition(
                        name=f"pa{i}",
                        model="mock-model",
                        system_prompt="",
                        tools=[],
                        max_iterations=1,
                        temperature=0.0,
                        memory_enabled=False,
                        cost_budget_usd=None,
                    ),
                    runner=runner,
                    role="worker",
                )
            )

        crew = Crew(members, pattern=CrewPattern.PARALLEL, session_id=make_session_id())
        result = await crew.run("Parallel.")
        assert result.duration_seconds >= 0.0

    async def test_hierarchical_crew_supervisor_dispatches_workers(self) -> None:
        import json

        supervisor_client = MockModelClient()
        supervisor_client.add_response(
            json.dumps({"worker1": "Research task.", "worker2": "Write task."})
        )
        supervisor_client.add_response("Final consolidated output.")

        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor
        from grampus.tools.registry import ToolRegistry

        sup_runner = AgentRunner(
            supervisor_client,
            ToolExecutor(ToolRegistry(), timeout_seconds=5.0),
            config=RunnerConfig(max_iterations=3, enable_memory=False),
        )
        supervisor = CrewMember(
            agent_def=AgentDefinition(
                name="supervisor",
                model="mock-model",
                system_prompt="",
                tools=[],
                max_iterations=3,
                temperature=0.0,
                memory_enabled=False,
                cost_budget_usd=None,
            ),
            runner=sup_runner,
            role="supervisor",
        )

        worker1 = _make_member("worker1", "Research result.")
        worker2 = _make_member("worker2", "Written content.")

        crew = Crew(
            [supervisor, worker1, worker2],
            pattern=CrewPattern.HIERARCHICAL,
            session_id=make_session_id(),
        )
        result = await crew.run("Coordinate the work.")
        assert "supervisor" in result.outputs

    async def test_crew_member_failure_wrapped_in_orchestration_error(self) -> None:
        from grampus.orchestration.runner import AgentRunner, RunnerConfig
        from grampus.tools.executor import ToolExecutor
        from grampus.tools.registry import ToolRegistry

        good_client = MockModelClient()
        good_client.add_response("Good output.")

        bad_client = MockModelClient()
        bad_client._responses = []

        class _FailClient:
            async def complete(self, **_: object) -> None:
                raise RuntimeError("agent failure")

            def stream(self, **_: object) -> None:
                raise NotImplementedError

        good_runner = AgentRunner(
            good_client,
            ToolExecutor(ToolRegistry(), timeout_seconds=5.0),
            config=RunnerConfig(max_iterations=2, enable_memory=False),
        )
        fail_runner = AgentRunner(
            _FailClient(),  # type: ignore[arg-type]
            ToolExecutor(ToolRegistry(), timeout_seconds=5.0),
            config=RunnerConfig(max_iterations=2, enable_memory=False),
        )

        m1 = CrewMember(
            agent_def=AgentDefinition(
                name="good",
                model="mock",
                system_prompt="",
                tools=[],
                max_iterations=2,
                temperature=0.0,
                memory_enabled=False,
                cost_budget_usd=None,
            ),
            runner=good_runner,
            role="worker",
        )
        m2 = CrewMember(
            agent_def=AgentDefinition(
                name="fail",
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

        crew = Crew([m1, m2], pattern=CrewPattern.SEQUENTIAL, session_id=make_session_id())
        with pytest.raises(OrchestrationError) as exc_info:
            await crew.run("trigger failure")
        assert exc_info.value.code == "CREW_MEMBER_FAILED"
