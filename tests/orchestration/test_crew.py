"""Tests for Crew — sequential, parallel, hierarchical multi-agent coordination."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import OrchestrationError
from nexus.core.types import (
    AgentDefinition,
    AgentStatus,
    ExecutionResult,
    Message,
    Role,
    TokenUsage,
)
from nexus.orchestration.crew import Crew, CrewMember, CrewPattern, CrewResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_def(name: str) -> AgentDefinition:
    return AgentDefinition(name=name, model="test-model")


def _exec_result(output: str = "result") -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[Message(role=Role.ASSISTANT, content=output)],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
            model="test-model",
        ),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


def _mock_runner(output: str = "result") -> MagicMock:
    runner = MagicMock()
    runner.run = AsyncMock(return_value=_exec_result(output))
    runner.cost_summary = MagicMock(return_value=None)
    return runner


def _member(name: str, output: str = "result", role: str = "worker") -> CrewMember:
    return CrewMember(
        agent_def=_agent_def(name),
        runner=_mock_runner(output),
        role=role,
    )


def _capture_runner(name: str, call_log: list[str]) -> MagicMock:
    """Runner that records which agent ran."""

    async def _run(
        agent_def: AgentDefinition,
        user_input: str,
        *,
        session_id: str,
        agent_state: object = None,
    ) -> ExecutionResult:
        call_log.append(agent_def.name)
        return _exec_result(f"{agent_def.name}-output")

    runner = MagicMock()
    runner.run = AsyncMock(side_effect=_run)
    runner.cost_summary = MagicMock(return_value=None)
    return runner


# ---------------------------------------------------------------------------
# TestCrewSequential
# ---------------------------------------------------------------------------


class TestCrewSequential:
    async def test_sequential_runs_members_in_order(self) -> None:
        call_order: list[str] = []
        m1 = CrewMember(
            agent_def=_agent_def("a"),
            runner=_capture_runner("a", call_order),
            role="worker",
        )
        m2 = CrewMember(
            agent_def=_agent_def("b"),
            runner=_capture_runner("b", call_order),
            role="worker",
        )
        crew = Crew([m1, m2], pattern=CrewPattern.SEQUENTIAL, session_id="s1")
        await crew.run("initial")
        assert call_order == ["a", "b"]

    async def test_sequential_passes_output_as_next_input(self) -> None:
        received_inputs: list[str] = []

        async def _run_b(
            agent_def: AgentDefinition,
            user_input: str,
            *,
            session_id: str,
            agent_state: object = None,
        ) -> ExecutionResult:
            received_inputs.append(user_input)
            return _exec_result("b-output")

        m1 = _member("a", output="a-output")
        runner_b = MagicMock()
        runner_b.run = AsyncMock(side_effect=_run_b)
        runner_b.cost_summary = MagicMock(return_value=None)
        m2 = CrewMember(agent_def=_agent_def("b"), runner=runner_b, role="worker")

        crew = Crew([m1, m2], pattern=CrewPattern.SEQUENTIAL, session_id="s1")
        await crew.run("initial")
        assert received_inputs == ["a-output"]

    async def test_sequential_returns_crew_result(self) -> None:
        crew = Crew([_member("a"), _member("b")], pattern=CrewPattern.SEQUENTIAL, session_id="s1")
        result = await crew.run("initial")
        assert isinstance(result, CrewResult)

    async def test_sequential_final_output_is_last_member_output(self) -> None:
        crew = Crew(
            [_member("a", "first"), _member("b", "last")],
            pattern=CrewPattern.SEQUENTIAL,
            session_id="s1",
        )
        result = await crew.run("initial")
        assert result.outputs["b"] == "last"


# ---------------------------------------------------------------------------
# TestCrewParallel
# ---------------------------------------------------------------------------


class TestCrewParallel:
    async def test_parallel_runs_all_members_concurrently(self) -> None:
        started: list[str] = []

        async def _run_slow(
            agent_def: AgentDefinition,
            user_input: str,
            *,
            session_id: str,
            agent_state: object = None,
        ) -> ExecutionResult:
            started.append(agent_def.name)
            await asyncio.sleep(0.01)
            return _exec_result(f"{agent_def.name}-output")

        members = []
        for n in ["a", "b", "c"]:
            runner = MagicMock()
            runner.run = AsyncMock(side_effect=_run_slow)
            runner.cost_summary = MagicMock(return_value=None)
            members.append(CrewMember(agent_def=_agent_def(n), runner=runner, role="worker"))

        crew = Crew(members, pattern=CrewPattern.PARALLEL, session_id="s1")
        await crew.run("initial")
        assert set(started) == {"a", "b", "c"}

    async def test_parallel_collects_all_outputs(self) -> None:
        crew = Crew(
            [_member("a", "out-a"), _member("b", "out-b")],
            pattern=CrewPattern.PARALLEL,
            session_id="s1",
        )
        result = await crew.run("initial")
        assert result.outputs["a"] == "out-a"
        assert result.outputs["b"] == "out-b"

    async def test_parallel_returns_crew_result(self) -> None:
        crew = Crew([_member("a"), _member("b")], pattern=CrewPattern.PARALLEL, session_id="s1")
        result = await crew.run("initial")
        assert isinstance(result, CrewResult)
        assert result.pattern == CrewPattern.PARALLEL


# ---------------------------------------------------------------------------
# TestCrewHierarchical
# ---------------------------------------------------------------------------


class TestCrewHierarchical:
    async def test_hierarchical_supervisor_dispatches_to_workers(self) -> None:
        worker_inputs: dict[str, str] = {}

        async def _worker_run(
            agent_def: AgentDefinition,
            user_input: str,
            *,
            session_id: str,
            agent_state: object = None,
        ) -> ExecutionResult:
            worker_inputs[agent_def.name] = user_input
            return _exec_result(f"{agent_def.name}-done")

        call_count = 0

        async def _supervisor_run(
            agent_def: AgentDefinition,
            user_input: str,
            *,
            session_id: str,
            agent_state: object = None,
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tasks = json.dumps({"worker1": "task for worker1", "worker2": "task for worker2"})
                return _exec_result(tasks)
            return _exec_result("final synthesis")

        sup_runner = MagicMock()
        sup_runner.run = AsyncMock(side_effect=_supervisor_run)
        sup_runner.cost_summary = MagicMock(return_value=None)
        supervisor = CrewMember(
            agent_def=_agent_def("supervisor"), runner=sup_runner, role="supervisor"
        )

        w1_runner = MagicMock()
        w1_runner.run = AsyncMock(side_effect=_worker_run)
        w1_runner.cost_summary = MagicMock(return_value=None)
        w1 = CrewMember(agent_def=_agent_def("worker1"), runner=w1_runner, role="worker")

        w2_runner = MagicMock()
        w2_runner.run = AsyncMock(side_effect=_worker_run)
        w2_runner.cost_summary = MagicMock(return_value=None)
        w2 = CrewMember(agent_def=_agent_def("worker2"), runner=w2_runner, role="worker")

        crew = Crew([supervisor, w1, w2], pattern=CrewPattern.HIERARCHICAL, session_id="s1")
        await crew.run("initial task")

        assert "worker1" in worker_inputs
        assert "worker2" in worker_inputs

    async def test_hierarchical_supervisor_called_twice(self) -> None:
        call_count = 0

        async def _supervisor_run(
            agent_def: AgentDefinition,
            user_input: str,
            *,
            session_id: str,
            agent_state: object = None,
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _exec_result(json.dumps({"worker1": "task"}))
            return _exec_result("synthesis")

        sup_runner = MagicMock()
        sup_runner.run = AsyncMock(side_effect=_supervisor_run)
        sup_runner.cost_summary = MagicMock(return_value=None)
        supervisor = CrewMember(
            agent_def=_agent_def("supervisor"), runner=sup_runner, role="supervisor"
        )
        worker = _member("worker1", "done")

        crew = Crew([supervisor, worker], pattern=CrewPattern.HIERARCHICAL, session_id="s1")
        await crew.run("initial")
        assert call_count == 2

    async def test_hierarchical_invalid_json_falls_back_to_all_workers(self) -> None:
        worker_called: list[str] = []

        async def _worker_run(
            agent_def: AgentDefinition,
            user_input: str,
            *,
            session_id: str,
            agent_state: object = None,
        ) -> ExecutionResult:
            worker_called.append(agent_def.name)
            return _exec_result("done")

        sup_call = 0

        async def _supervisor_run(
            agent_def: AgentDefinition,
            user_input: str,
            *,
            session_id: str,
            agent_state: object = None,
        ) -> ExecutionResult:
            nonlocal sup_call
            sup_call += 1
            if sup_call == 1:
                return _exec_result("not valid json at all")
            return _exec_result("final")

        sup_runner = MagicMock()
        sup_runner.run = AsyncMock(side_effect=_supervisor_run)
        sup_runner.cost_summary = MagicMock(return_value=None)
        supervisor = CrewMember(
            agent_def=_agent_def("supervisor"), runner=sup_runner, role="supervisor"
        )

        w1_runner = MagicMock()
        w1_runner.run = AsyncMock(side_effect=_worker_run)
        w1_runner.cost_summary = MagicMock(return_value=None)
        w1 = CrewMember(agent_def=_agent_def("w1"), runner=w1_runner, role="worker")

        w2_runner = MagicMock()
        w2_runner.run = AsyncMock(side_effect=_worker_run)
        w2_runner.cost_summary = MagicMock(return_value=None)
        w2 = CrewMember(agent_def=_agent_def("w2"), runner=w2_runner, role="worker")

        crew = Crew([supervisor, w1, w2], pattern=CrewPattern.HIERARCHICAL, session_id="s1")
        await crew.run("initial")

        assert "w1" in worker_called
        assert "w2" in worker_called

    async def test_hierarchical_worker_outputs_fed_to_supervisor(self) -> None:
        supervisor_second_input: list[str] = []
        call_count = 0

        async def _supervisor_run(
            agent_def: AgentDefinition,
            user_input: str,
            *,
            session_id: str,
            agent_state: object = None,
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _exec_result(json.dumps({"worker1": "do the thing"}))
            supervisor_second_input.append(user_input)
            return _exec_result("synthesized")

        sup_runner = MagicMock()
        sup_runner.run = AsyncMock(side_effect=_supervisor_run)
        sup_runner.cost_summary = MagicMock(return_value=None)
        supervisor = CrewMember(
            agent_def=_agent_def("supervisor"), runner=sup_runner, role="supervisor"
        )
        worker = _member("worker1", "worker-result")

        crew = Crew([supervisor, worker], pattern=CrewPattern.HIERARCHICAL, session_id="s1")
        await crew.run("initial")

        assert len(supervisor_second_input) == 1
        assert "worker1" in supervisor_second_input[0]
        assert "worker-result" in supervisor_second_input[0]


# ---------------------------------------------------------------------------
# TestCrewErrors
# ---------------------------------------------------------------------------


class TestCrewErrors:
    async def test_crew_member_failed_wraps_exception(self) -> None:
        failing_runner = MagicMock()
        failing_runner.run = AsyncMock(side_effect=RuntimeError("boom"))
        failing_runner.cost_summary = MagicMock(return_value=None)
        member = CrewMember(agent_def=_agent_def("failing"), runner=failing_runner, role="worker")
        crew = Crew([member], pattern=CrewPattern.SEQUENTIAL, session_id="s1")
        with pytest.raises(OrchestrationError) as exc_info:
            await crew.run("initial")
        assert exc_info.value.code == "CREW_MEMBER_FAILED"

    async def test_crew_result_includes_duration(self) -> None:
        crew = Crew([_member("a")], pattern=CrewPattern.SEQUENTIAL, session_id="s1")
        result = await crew.run("initial")
        assert result.duration_seconds >= 0.0
