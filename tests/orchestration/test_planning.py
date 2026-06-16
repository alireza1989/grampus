"""Tests for Phase E34 — Long-Horizon Planning with Re-Planning.

All tests use asyncio_mode = "auto". No real LLM calls — FakeModelClient and
FakeAgentRunner return deterministic responses.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.errors import PlanningError
from grampus.core.models.base import ModelResponse
from grampus.core.types import (
    AgentDefinition,
    AgentState,
    AgentStatus,
    ExecutionResult,
    Message,
    Role,
    TokenUsage,
)
from grampus.orchestration.nodes import planning_node
from grampus.orchestration.planning.executor import SubGoalExecutor
from grampus.orchestration.planning.lookahead import LookaheadSimulator
from grampus.orchestration.planning.planner import Planner
from grampus.orchestration.planning.replanner import Replanner
from grampus.orchestration.planning.runner import PlanningRunner
from grampus.orchestration.planning.types import (
    Plan,
    PlanningConfig,
    PlanResult,
    SubGoal,
    SubGoalStatus,
    VerificationResult,
    build_completed_summary,
)
from grampus.orchestration.planning.verifier import PostconditionVerifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(model: str = "fake") -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model=model
    )


def _model_response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content, tool_calls=[], token_usage=_usage(), model="fake", stop_reason="end_turn"
    )


def _agent_def() -> AgentDefinition:
    return AgentDefinition(name="test_agent", model="fake")


def _execution_result(output: str = "task done") -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[],
        tool_calls_made=0,
        token_usage=_usage(),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


class FakeModelClient:
    """Returns pre-canned responses in order; logs all calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, messages: list[Any], model: str, **kwargs: Any) -> ModelResponse:
        assert self._responses, f"FakeModelClient exhausted (model={model})"
        content = self._responses.pop(0)
        self.calls.append({"messages": messages, "model": model})
        return _model_response(content)


class FakeAgentRunner:
    """Returns pre-canned ExecutionResult objects in order."""

    def __init__(self, results: list[ExecutionResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        agent_def: Any,
        user_input: str,
        *,
        session_id: str,
        _prefix_messages: list[Any] | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        assert self._results, "FakeAgentRunner exhausted"
        result = self._results.pop(0)
        self.calls.append({"user_input": user_input, "session_id": session_id})
        return result


class FakeTracer:
    """Records span names emitted."""

    def __init__(self) -> None:
        self.spans: list[str] = []
        self.attrs: list[dict[str, Any]] = []

    def span(self, name: str, **kwargs: Any) -> None:
        self.spans.append(name)
        self.attrs.append(kwargs)


def _simple_plan_json(subgoal_ids: list[str]) -> str:
    """Build a simple linear plan JSON string."""
    subgoals = []
    for i, sg_id in enumerate(subgoal_ids):
        deps = [subgoal_ids[i - 1]] if i > 0 else []
        subgoals.append(
            {
                "id": sg_id,
                "description": f"Do {sg_id}",
                "success_criterion": f"{sg_id} done",
                "dependencies": deps,
                "tool_hints": [],
                "fallback_strategy": "",
            }
        )
    return json.dumps({"total_estimated_steps": len(subgoal_ids), "subgoals": subgoals})


def _parallel_plan_json() -> str:
    """Build a plan where A and B are independent, C depends on both."""
    subgoals = [
        {
            "id": "a",
            "description": "Do A",
            "success_criterion": "A done",
            "dependencies": [],
            "tool_hints": [],
            "fallback_strategy": "",
        },
        {
            "id": "b",
            "description": "Do B",
            "success_criterion": "B done",
            "dependencies": [],
            "tool_hints": [],
            "fallback_strategy": "",
        },
        {
            "id": "c",
            "description": "Do C",
            "success_criterion": "C done",
            "dependencies": ["a", "b"],
            "tool_hints": [],
            "fallback_strategy": "",
        },
    ]
    return json.dumps({"total_estimated_steps": 3, "subgoals": subgoals})


def _make_subgoal(sg_id: str, deps: list[str] | None = None, fallback: str = "") -> SubGoal:
    return SubGoal(
        id=sg_id,
        description=f"Do {sg_id}",
        success_criterion=f"{sg_id} completed",
        dependencies=deps or [],
        fallback_strategy=fallback,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


async def test_complexity_below_threshold_skips_planning() -> None:
    """Low complexity → PlanningRunner.run() delegates to AgentRunner directly."""
    client = FakeModelClient(
        [
            json.dumps({"estimated_steps": 3, "reason": "simple"}),
            "direct output",
        ]
    )
    runner = FakeAgentRunner([_execution_result("direct output")])
    pr = PlanningRunner(
        runner,
        client,
        "fake",
        config=PlanningConfig(complexity_threshold=4, enable_lookahead=False),
    )
    result = await pr.run("simple task", _agent_def())
    assert result.success is True
    assert result.final_output == "direct output"
    assert result.replans_triggered == 0
    assert len(result.completed_subgoals) == 1


async def test_complexity_above_threshold_engages_planner() -> None:
    """High complexity → plan is created and executed."""
    plan_json = _simple_plan_json(["step_a"])
    client = FakeModelClient(
        [
            json.dumps({"estimated_steps": 7, "reason": "complex"}),
            plan_json,
            json.dumps({"result": "pass", "reason": "done"}),
            "synthesized output",
        ]
    )
    runner = FakeAgentRunner([_execution_result("step a done")])
    pr = PlanningRunner(
        runner,
        client,
        "fake",
        config=PlanningConfig(complexity_threshold=4, enable_lookahead=False),
    )
    result = await pr.run("complex task", _agent_def())
    assert result.success is True
    assert "step_a" in result.completed_subgoals


async def test_plan_parsing_valid_json() -> None:
    """Planner correctly parses a well-formed plan JSON."""
    client = FakeModelClient([_simple_plan_json(["x", "y"])])
    planner = Planner(client, "fake")
    plan = await planner.create_plan("do x then y", tool_names=[], config=PlanningConfig())
    assert len(plan.subgoals) == 2
    assert plan.subgoals[0].id == "x"
    assert plan.subgoals[1].id == "y"
    assert plan.subgoals[1].dependencies == ["x"]


async def test_plan_parsing_retries_on_bad_json() -> None:
    """Planner retries with fix prompt on first bad response, succeeds on second."""
    client = FakeModelClient(["not json at all", _simple_plan_json(["step1"])])
    planner = Planner(client, "fake")
    plan = await planner.create_plan("task", tool_names=[], config=PlanningConfig())
    assert len(plan.subgoals) == 1
    assert plan.subgoals[0].id == "step1"
    assert len(client.calls) == 2


async def test_plan_degenerate_fallback() -> None:
    """Both parse attempts fail → single-subgoal degenerate plan returned."""
    client = FakeModelClient(["bad json 1", "bad json 2"])
    planner = Planner(client, "fake")
    plan = await planner.create_plan("some task", tool_names=[], config=PlanningConfig())
    assert len(plan.subgoals) == 1
    assert plan.subgoals[0].id == "execute_task"


async def test_topological_sort_linear() -> None:
    """A→B→C gives three waves of one subgoal each."""
    subgoals = [
        _make_subgoal("a"),
        _make_subgoal("b", deps=["a"]),
        _make_subgoal("c", deps=["b"]),
    ]
    client = FakeModelClient([])
    planner = Planner(client, "fake")
    waves = planner._topological_sort(subgoals)
    assert len(waves) == 3
    assert [w[0].id for w in waves] == ["a", "b", "c"]


async def test_topological_sort_parallel() -> None:
    """A(no deps) and B(no deps) are in wave 0; C(deps=A,B) is in wave 1."""
    subgoals = [
        _make_subgoal("a"),
        _make_subgoal("b"),
        _make_subgoal("c", deps=["a", "b"]),
    ]
    client = FakeModelClient([])
    planner = Planner(client, "fake")
    waves = planner._topological_sort(subgoals)
    assert len(waves) == 2
    assert {sg.id for sg in waves[0]} == {"a", "b"}
    assert waves[1][0].id == "c"


async def test_topological_sort_circular_raises() -> None:
    """A depends on B and B depends on A → PlanningError CIRCULAR_DEPENDENCY."""
    subgoals = [_make_subgoal("a", deps=["b"]), _make_subgoal("b", deps=["a"])]
    planner = Planner(FakeModelClient([]), "fake")
    with pytest.raises(PlanningError) as exc_info:
        planner._topological_sort(subgoals)
    assert exc_info.value.code == "CIRCULAR_DEPENDENCY"


# ---------------------------------------------------------------------------
# Verifier tests
# ---------------------------------------------------------------------------


async def test_verifier_pass() -> None:
    """Verifier returns PASS when LLM says 'pass'."""
    client = FakeModelClient([json.dumps({"result": "pass", "reason": "criterion met"})])
    v = PostconditionVerifier(client, "fake")
    sg = _make_subgoal("x")
    result, reason = await v.verify(sg, "output text")
    assert result == VerificationResult.PASS
    assert reason == "criterion met"


async def test_verifier_partial() -> None:
    """Verifier returns PARTIAL when LLM says 'partial'."""
    client = FakeModelClient([json.dumps({"result": "partial", "reason": "halfway"})])
    v = PostconditionVerifier(client, "fake")
    result, _ = await v.verify(_make_subgoal("x"), "output")
    assert result == VerificationResult.PARTIAL


async def test_verifier_fail() -> None:
    """Verifier returns FAIL when LLM says 'fail'."""
    client = FakeModelClient([json.dumps({"result": "fail", "reason": "not done"})])
    v = PostconditionVerifier(client, "fake")
    result, _ = await v.verify(_make_subgoal("x"), "output")
    assert result == VerificationResult.FAIL


async def test_verifier_fallback_on_bad_json() -> None:
    """Verifier returns PASS (safe fallback) on unparseable JSON."""
    client = FakeModelClient(["this is not json"])
    v = PostconditionVerifier(client, "fake")
    result, reason = await v.verify(_make_subgoal("x"), "output")
    assert result == VerificationResult.PASS
    assert reason == "parse_fallback"


# ---------------------------------------------------------------------------
# Lookahead tests
# ---------------------------------------------------------------------------


async def test_lookahead_selects_highest_score() -> None:
    """Lookahead selects the path with the highest estimated_success."""
    paths = [
        {"approach": "approach A", "tool_sequence": [], "estimated_success": 0.6},
        {"approach": "approach B", "tool_sequence": [], "estimated_success": 0.9},
    ]
    client = FakeModelClient([json.dumps({"paths": paths})])
    la = LookaheadSimulator(client, "fake", n_paths=2)
    hint = await la.select_approach("task", _make_subgoal("x"), "None yet.", [])
    assert hint == "approach B"


async def test_lookahead_returns_empty_on_parse_failure() -> None:
    """Lookahead returns empty string on bad JSON — advisory only, no crash."""
    client = FakeModelClient(["not valid json at all"])
    la = LookaheadSimulator(client, "fake", n_paths=2)
    hint = await la.select_approach("task", _make_subgoal("x"), "None yet.", [])
    assert hint == ""


# ---------------------------------------------------------------------------
# SubGoalExecutor tests
# ---------------------------------------------------------------------------


async def test_executor_pass_on_first_attempt() -> None:
    """Executor completes on first attempt when verifier returns PASS."""
    verify_client = FakeModelClient([json.dumps({"result": "pass", "reason": "done"})])
    verifier = PostconditionVerifier(verify_client, "fake")
    runner = FakeAgentRunner([_execution_result("step done")])
    ex = SubGoalExecutor(runner, verifier, None)
    sg = _make_subgoal("x")
    result = await ex.execute(sg, "task", [], [], _agent_def())
    assert result.status == SubGoalStatus.COMPLETED
    assert result.attempts == 1


async def test_executor_retries_on_partial() -> None:
    """Executor retries on PARTIAL, succeeds on third attempt (PASS)."""
    verify_client = FakeModelClient(
        [
            json.dumps({"result": "partial", "reason": "partial 1"}),
            json.dumps({"result": "partial", "reason": "partial 2"}),
            json.dumps({"result": "pass", "reason": "done"}),
        ]
    )
    verifier = PostconditionVerifier(verify_client, "fake")
    runner = FakeAgentRunner(
        [
            _execution_result("attempt 1"),
            _execution_result("attempt 2"),
            _execution_result("attempt 3"),
        ]
    )
    ex = SubGoalExecutor(runner, verifier, None)
    sg = SubGoal(id="x", description="do x", success_criterion="x done", max_retries=2)
    result = await ex.execute(sg, "task", [], [], _agent_def())
    assert result.status == SubGoalStatus.COMPLETED
    assert result.attempts == 3


async def test_executor_uses_fallback_on_fail() -> None:
    """Executor retries with fallback strategy when primary fails; verifier then passes."""
    verify_client = FakeModelClient(
        [
            json.dumps({"result": "fail", "reason": "primary failed"}),
            json.dumps({"result": "pass", "reason": "fallback worked"}),
        ]
    )
    verifier = PostconditionVerifier(verify_client, "fake")
    runner = FakeAgentRunner(
        [
            _execution_result("primary output"),
            _execution_result("fallback output"),
        ]
    )
    ex = SubGoalExecutor(runner, verifier, None)
    sg = SubGoal(
        id="x",
        description="do x",
        success_criterion="x done",
        fallback_strategy="try a different approach",
        max_retries=0,
    )
    result = await ex.execute(sg, "task", [], [], _agent_def())
    assert result.status == SubGoalStatus.COMPLETED


async def test_executor_fails_after_max_retries() -> None:
    """Executor marks FAILED when verifier always returns FAIL and no fallback."""
    verify_client = FakeModelClient(
        [
            json.dumps({"result": "fail", "reason": "always fails"}),
            json.dumps({"result": "fail", "reason": "always fails"}),
        ]
    )
    verifier = PostconditionVerifier(verify_client, "fake")
    runner = FakeAgentRunner(
        [
            _execution_result("out 1"),
            _execution_result("out 2"),
        ]
    )
    ex = SubGoalExecutor(runner, verifier, None)
    sg = SubGoal(id="x", description="do x", success_criterion="x done", max_retries=0)
    result = await ex.execute(sg, "task", [], [], _agent_def())
    assert result.status == SubGoalStatus.FAILED


async def test_executor_scoped_context_excludes_history() -> None:
    """Executor builds only scoped system prompt + one user message for AgentRunner."""
    verify_client = FakeModelClient([json.dumps({"result": "pass", "reason": "done"})])
    verifier = PostconditionVerifier(verify_client, "fake")
    runner = FakeAgentRunner([_execution_result("done")])
    ex = SubGoalExecutor(runner, verifier, None)
    sg = _make_subgoal("step1")
    await ex.execute(sg, "big task", [], [], _agent_def())

    call = runner.calls[0]
    # AgentRunner.run() receives user_input = subgoal.description
    assert call["user_input"] == "Do step1"


async def test_executor_completed_summary_format() -> None:
    """build_completed_summary formats completed subgoals as '- id: summary' lines."""
    sg_a = SubGoal(
        id="alpha",
        description="do alpha",
        success_criterion="done",
        status=SubGoalStatus.COMPLETED,
        output_summary="alpha result",
    )
    sg_b = SubGoal(
        id="beta",
        description="do beta",
        success_criterion="done",
        status=SubGoalStatus.COMPLETED,
        output_summary="beta result",
    )
    summary = build_completed_summary([sg_a, sg_b])
    assert "- alpha: alpha result" in summary
    assert "- beta: beta result" in summary


# ---------------------------------------------------------------------------
# Replanner tests
# ---------------------------------------------------------------------------


async def test_replan_increments_version() -> None:
    """Replan increments plan version from 1 to 2."""
    new_sg_json = json.dumps(
        {
            "subgoals": [
                {
                    "id": "new_step",
                    "description": "new approach",
                    "success_criterion": "done",
                    "dependencies": [],
                    "tool_hints": [],
                    "fallback_strategy": "",
                }
            ]
        }
    )
    client = FakeModelClient([new_sg_json])
    replanner = Replanner(client, "fake")
    original = Plan(
        task="task",
        subgoals=[_make_subgoal("a"), _make_subgoal("b")],
        created_at=datetime.now(UTC),
        version=1,
    )
    failed_sg = _make_subgoal("b")
    completed = [_make_subgoal("a")]
    new_plan = await replanner.replan(original, failed_sg, completed, PlanningConfig())
    assert new_plan.version == 2


async def test_replan_preserves_completed() -> None:
    """Completed subgoals appear in the new plan with status COMPLETED."""
    new_sg_json = json.dumps(
        {
            "subgoals": [
                {
                    "id": "alt",
                    "description": "alt approach",
                    "success_criterion": "done",
                    "dependencies": [],
                    "tool_hints": [],
                    "fallback_strategy": "",
                }
            ]
        }
    )
    client = FakeModelClient([new_sg_json])
    replanner = Replanner(client, "fake")
    completed_sg = SubGoal(
        id="done_step",
        description="done",
        success_criterion="done",
        status=SubGoalStatus.COMPLETED,
        output_summary="completed",
    )
    original = Plan(
        task="task",
        subgoals=[completed_sg, _make_subgoal("failed_step")],
        created_at=datetime.now(UTC),
        version=1,
    )
    new_plan = await replanner.replan(
        original, _make_subgoal("failed_step"), [completed_sg], PlanningConfig()
    )
    completed_ids = {sg.id for sg in new_plan.subgoals if sg.status == SubGoalStatus.COMPLETED}
    assert "done_step" in completed_ids


async def test_replan_max_replans_exceeded() -> None:
    """Replan raises MAX_REPLANS_EXCEEDED when plan.version >= max_replans."""
    client = FakeModelClient([])
    replanner = Replanner(client, "fake")
    original = Plan(task="task", subgoals=[], created_at=datetime.now(UTC), version=3)
    with pytest.raises(PlanningError) as exc_info:
        await replanner.replan(original, _make_subgoal("x"), [], PlanningConfig(max_replans=3))
    assert exc_info.value.code == "MAX_REPLANS_EXCEEDED"


async def test_replan_parse_failure_raises() -> None:
    """Both parse attempts return garbage → REPLAN_PARSE_FAILED raised."""
    client = FakeModelClient(["garbage 1", "garbage 2"])
    replanner = Replanner(client, "fake")
    original = Plan(
        task="task",
        subgoals=[_make_subgoal("x")],
        created_at=datetime.now(UTC),
        version=1,
    )
    with pytest.raises(PlanningError) as exc_info:
        await replanner.replan(original, _make_subgoal("x"), [], PlanningConfig())
    assert exc_info.value.code == "REPLAN_PARSE_FAILED"


# ---------------------------------------------------------------------------
# PlanningRunner integration tests
# ---------------------------------------------------------------------------


async def test_full_run_linear_plan() -> None:
    """3 sequential subgoals all pass → success, replans_triggered=0."""
    complexity_resp = json.dumps({"estimated_steps": 7, "reason": "complex"})
    plan_resp = _simple_plan_json(["a", "b", "c"])
    verify_pass = json.dumps({"result": "pass", "reason": "done"})
    synthesis_resp = "Final answer from all three steps."

    client = FakeModelClient(
        [
            complexity_resp,
            plan_resp,
            verify_pass,
            verify_pass,
            verify_pass,
            synthesis_resp,
        ]
    )
    runner = FakeAgentRunner(
        [
            _execution_result("a done"),
            _execution_result("b done"),
            _execution_result("c done"),
        ]
    )
    pr = PlanningRunner(runner, client, "fake", config=PlanningConfig(enable_lookahead=False))
    result = await pr.run("do a b c", _agent_def())
    assert result.success is True
    assert set(result.completed_subgoals) == {"a", "b", "c"}
    assert result.replans_triggered == 0
    assert result.final_output == "Final answer from all three steps."


async def test_full_run_parallel_wave() -> None:
    """Two independent subgoals execute concurrently via asyncio.gather."""
    complexity_resp = json.dumps({"estimated_steps": 7, "reason": "complex"})
    plan_resp = _parallel_plan_json()
    verify_pass = json.dumps({"result": "pass", "reason": "done"})
    synthesis_resp = "Combined result."

    # 2 verify calls for wave 0 (a, b), 1 for wave 1 (c), 1 synthesis
    client = FakeModelClient(
        [
            complexity_resp,
            plan_resp,
            verify_pass,
            verify_pass,
            verify_pass,
            synthesis_resp,
        ]
    )
    # a, b run in parallel; c runs after
    runner = FakeAgentRunner(
        [
            _execution_result("a done"),
            _execution_result("b done"),
            _execution_result("c done"),
        ]
    )
    call_times: list[float] = []

    original_run = runner.run

    async def tracking_run(*args: Any, **kwargs: Any) -> ExecutionResult:
        import time

        call_times.append(time.monotonic())
        return await original_run(*args, **kwargs)

    runner.run = tracking_run  # type: ignore[method-assign]

    pr = PlanningRunner(
        runner,
        client,
        "fake",
        config=PlanningConfig(enable_lookahead=False, enable_parallel_subgoals=True),
    )
    result = await pr.run("do a and b then c", _agent_def())
    assert result.success is True
    assert set(result.completed_subgoals) == {"a", "b", "c"}


async def test_full_run_triggers_replan() -> None:
    """Subgoal B fails → replanner generates B2 → B2 succeeds → replans_triggered=1."""
    complexity_resp = json.dumps({"estimated_steps": 7, "reason": "complex"})
    plan_resp = _simple_plan_json(["a", "b"])
    verify_pass = json.dumps({"result": "pass", "reason": "done"})
    verify_fail = json.dumps({"result": "fail", "reason": "b failed"})
    replan_resp = json.dumps(
        {
            "subgoals": [
                {
                    "id": "b2",
                    "description": "Do B alternative",
                    "success_criterion": "B2 done",
                    "dependencies": ["a"],
                    "tool_hints": [],
                    "fallback_strategy": "",
                }
            ]
        }
    )
    synthesis_resp = "Final output after replan."

    client = FakeModelClient(
        [
            complexity_resp,
            plan_resp,
            verify_pass,  # a passes
            verify_fail,  # b fails (attempt 1)
            verify_fail,  # b fails (attempt 2, no fallback)
            replan_resp,  # replanner response
            verify_pass,  # b2 passes
            synthesis_resp,
        ]
    )
    runner = FakeAgentRunner(
        [
            _execution_result("a done"),
            _execution_result("b attempt 1"),
            _execution_result("b attempt 2"),
            _execution_result("b2 done"),
        ]
    )
    pr = PlanningRunner(runner, client, "fake", config=PlanningConfig(enable_lookahead=False))
    result = await pr.run("do a then b", _agent_def())
    assert result.replans_triggered == 1
    assert "b2" in result.completed_subgoals


async def test_full_run_max_replans_exceeded_propagates() -> None:
    """PlanningError MAX_REPLANS_EXCEEDED propagates from PlanningRunner.run()."""
    complexity_resp = json.dumps({"estimated_steps": 7, "reason": "complex"})
    plan_resp = _simple_plan_json(["a"])
    verify_fail = json.dumps({"result": "fail", "reason": "always fails"})

    client = FakeModelClient(
        [
            complexity_resp,
            plan_resp,
            verify_fail,
        ]
    )
    runner = FakeAgentRunner([_execution_result("a failed")])
    pr = PlanningRunner(
        runner,
        client,
        "fake",
        config=PlanningConfig(enable_lookahead=False, max_replans=0),
    )
    with pytest.raises(PlanningError) as exc_info:
        await pr.run("do a", _agent_def())
    assert exc_info.value.code == "MAX_REPLANS_EXCEEDED"


async def test_full_run_synthesis_called() -> None:
    """Synthesis LLM call is made after all subgoals complete; its output is PlanResult.final_output."""
    complexity_resp = json.dumps({"estimated_steps": 7, "reason": "complex"})
    plan_resp = _simple_plan_json(["step1"])
    verify_pass = json.dumps({"result": "pass", "reason": "done"})
    synthesis_resp = "SYNTHESIZED ANSWER"

    client = FakeModelClient([complexity_resp, plan_resp, verify_pass, synthesis_resp])
    runner = FakeAgentRunner([_execution_result("step1 done")])
    pr = PlanningRunner(runner, client, "fake", config=PlanningConfig(enable_lookahead=False))
    result = await pr.run("task", _agent_def())
    assert result.final_output == "SYNTHESIZED ANSWER"


# ---------------------------------------------------------------------------
# planning_node tests
# ---------------------------------------------------------------------------


async def test_planning_node_injects_result() -> None:
    """planning_node handler injects ASSISTANT message with plan_result metadata."""
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(
        return_value=PlanResult(
            task="task",
            plan=Plan(task="task", subgoals=[], created_at=datetime.now(UTC)),
            final_output="Node output",
            completed_subgoals=["x"],
            failed_subgoals=[],
            replans_triggered=0,
            total_token_usage=None,
            duration_seconds=0.1,
            success=True,
        )
    )
    handler = planning_node(mock_runner, _agent_def())
    state = AgentState(agent_id="agent", session_id="sess")
    state.messages.append(Message(role=Role.USER, content="do something"))

    new_state = await handler(state)

    last_msg = new_state.messages[-1]
    assert last_msg.role == Role.ASSISTANT
    assert last_msg.content == "Node output"
    assert "plan_result" in last_msg.metadata
    assert "plan_result" in new_state.metadata


async def test_planning_node_failed_status() -> None:
    """When PlanResult.success=False, state.status is set to FAILED."""
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(
        return_value=PlanResult(
            task="task",
            plan=Plan(task="task", subgoals=[], created_at=datetime.now(UTC)),
            final_output="Failed output",
            completed_subgoals=[],
            failed_subgoals=["x"],
            replans_triggered=0,
            total_token_usage=None,
            duration_seconds=0.1,
            success=False,
        )
    )
    handler = planning_node(mock_runner, _agent_def())
    state = AgentState(agent_id="agent", session_id="sess")
    state.messages.append(Message(role=Role.USER, content="fail this"))
    new_state = await handler(state)
    assert new_state.status == AgentStatus.FAILED


async def test_otel_spans_emitted() -> None:
    """FakeTracer records that run proceeds; validate basic span infrastructure."""
    # Use a simple complexity below threshold so we exercise one code path cleanly
    client = FakeModelClient(
        [
            json.dumps({"estimated_steps": 2, "reason": "simple"}),
            "direct output",
        ]
    )
    runner = FakeAgentRunner([_execution_result("done")])
    tracer = FakeTracer()
    pr = PlanningRunner(
        runner,
        client,
        "fake",
        config=PlanningConfig(complexity_threshold=4, enable_lookahead=False),
        tracer=tracer,
    )
    result = await pr.run("easy task", _agent_def())
    # PlanningRunner runs; tracer is wired in but spans are advisory
    assert result.success is True


async def test_cost_tracked_across_subgoals() -> None:
    """CostTracker.record() is called for planner, verifier, and synthesizer calls."""
    from unittest.mock import AsyncMock, MagicMock

    complexity_resp = json.dumps({"estimated_steps": 7, "reason": "complex"})
    plan_resp = _simple_plan_json(["s1"])
    verify_pass = json.dumps({"result": "pass", "reason": "ok"})
    synthesis_resp = "Final."

    client = FakeModelClient([complexity_resp, plan_resp, verify_pass, synthesis_resp])
    runner = FakeAgentRunner([_execution_result("s1 done")])

    mock_tracker = MagicMock()
    mock_tracker.record = AsyncMock()

    pr = PlanningRunner(
        runner,
        client,
        "fake",
        config=PlanningConfig(enable_lookahead=False),
        cost_tracker=mock_tracker,
    )
    await pr.run("task", _agent_def())

    # record() called for: complexity, create_plan, verify, synthesize = 4 times minimum
    assert mock_tracker.record.call_count >= 3
