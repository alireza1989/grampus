"""Pydantic types for the long-horizon planning subsystem."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from grampus.core.types import TokenUsage


class SubGoalStatus(StrEnum):
    """Execution status of an individual subgoal."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VerificationResult(StrEnum):
    """Outcome of postcondition verification for a subgoal."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


class SubGoal(BaseModel):
    """One step in a structured plan DAG.

    Args:
        id: Short snake_case slug, ≤ 20 chars.
        description: What this step should accomplish.
        success_criterion: Verifiable completion condition.
        dependencies: IDs of subgoals that must complete first.
        tool_hints: Suggested tool names (advisory, not enforced).
        fallback_strategy: Alternative approach if primary fails.
        max_retries: How many PARTIAL retries before declaring FAIL.
        status: Current execution status.
        output_summary: Filled after completion; 1-2 sentences.
        attempts: Total execution attempts so far.
        failure_reason: Last failure reason (filled on FAIL).
    """

    id: str
    description: str
    success_criterion: str
    dependencies: list[str] = Field(default_factory=list)
    tool_hints: list[str] = Field(default_factory=list)
    fallback_strategy: str = ""
    max_retries: int = 2
    status: SubGoalStatus = SubGoalStatus.PENDING
    output_summary: str = ""
    attempts: int = 0
    failure_reason: str = ""


class Plan(BaseModel):
    """A structured DAG of subgoals for a user task.

    Args:
        task: Original user task description.
        subgoals: Ordered list; DAG implied by dependencies field.
        total_estimated_steps: Planner's estimate of total tool calls.
        created_at: UTC timestamp of plan creation.
        version: Increments on each replan (starts at 1).
    """

    task: str
    subgoals: list[SubGoal]
    total_estimated_steps: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = 1


class PlanResult(BaseModel):
    """Final outcome of a PlanningRunner execution.

    Args:
        task: Original user task.
        plan: The final plan version executed.
        final_output: Synthesized answer from all completed subgoals.
        completed_subgoals: IDs of successfully completed subgoals.
        failed_subgoals: IDs of subgoals that could not be completed.
        replans_triggered: How many times replanning was triggered.
        total_token_usage: Accumulated token usage across all calls.
        duration_seconds: Wall-clock duration of the full run.
        success: True when at least one subgoal completed and no catastrophic failure.
    """

    task: str
    plan: Plan
    final_output: str
    completed_subgoals: list[str]
    failed_subgoals: list[str]
    replans_triggered: int
    total_token_usage: TokenUsage | None
    duration_seconds: float
    success: bool


class PlanningConfig(BaseModel):
    """Tuning parameters for PlanningRunner.

    Args:
        max_subgoals: Hard cap on subgoals per plan.
        max_replans: Maximum number of replan cycles allowed.
        complexity_threshold: Skip planning if estimated steps ≤ this value.
        enable_lookahead: Run FLARE-style path simulation before each subgoal.
        lookahead_paths: Candidate paths to generate per lookahead call.
        enable_parallel_subgoals: Run independent subgoals concurrently.
        cost_budget_usd: Hard cost cap across all planning calls.
        planner_model_tier: Model tier for plan generation.
        executor_model_tier: Model tier for subgoal execution.
        verifier_model_tier: Model tier for postcondition verification.
    """

    max_subgoals: int = 12
    max_replans: int = 3
    complexity_threshold: int = 4
    enable_lookahead: bool = True
    lookahead_paths: int = 2
    enable_parallel_subgoals: bool = True
    cost_budget_usd: float | None = None
    planner_model_tier: str = "powerful"
    executor_model_tier: str = "balanced"
    verifier_model_tier: str = "fast"

    model_config = {"arbitrary_types_allowed": True}


def _empty_token_usage(model: str = "unknown") -> TokenUsage:
    """Return a zero-value TokenUsage for accumulation."""
    return TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0, model=model)


def _add_token_usage(a: TokenUsage | None, b: TokenUsage | None) -> TokenUsage | None:
    """Accumulate two optional TokenUsage objects."""
    if a is None:
        return b
    if b is None:
        return a
    return TokenUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
        model=b.model,
    )


def build_completed_summary(completed: list[SubGoal]) -> str:
    """Format completed subgoal outputs as a concise summary string.

    Args:
        completed: List of completed SubGoal objects.

    Returns:
        Multi-line string with one '- id: summary' line per subgoal,
        or 'None yet.' when list is empty.
    """
    if not completed:
        return "None yet."
    return "\n".join(f"- {sg.id}: {sg.output_summary}" for sg in completed)


def extract_last_user_content(messages: list[Any]) -> str:
    """Return content of the last USER-role message.

    Args:
        messages: List of Message objects.

    Returns:
        Content string.

    Raises:
        ValueError: When no USER message is found.
    """
    from grampus.core.types import Role

    for msg in reversed(messages):
        if msg.role == Role.USER and msg.content is not None:
            return str(msg.content)
    raise ValueError("No USER message found in state")
