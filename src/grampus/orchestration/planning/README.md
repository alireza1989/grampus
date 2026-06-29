# `grampus/orchestration/planning/` — Long-Horizon Planning (ADR-014)

This sub-package implements `PlanningRunner`, a structured orchestration layer that wraps `AgentRunner` without modifying it. It decomposes long-horizon tasks into a SubGoal DAG, executes them in topological waves, applies adaptive replanning on failures, and confines each executor to scoped context — preventing error propagation and reducing token usage by ~82% vs. passing full history to every call.

Grounded in: "Why Reasoning Fails to Plan" (arXiv 2601.22311, Jan 2026), Task-Decoupled Planning (arXiv 2601.07577), ReAcTree (arXiv 2511.02424, AAMAS 2026), Google DeepMind Subgoal Framework (arXiv 2603.19685, Mar 2026), "Learning When to Plan" (arXiv 2509.03581).

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `PlanningRunner` | `runner.py` | Top-level orchestrator: complexity gate → plan → waves → replan → synthesize |
| `Planner` | `planner.py` | LLM call that produces a `Plan` (SubGoal DAG) from a task description |
| `SubGoalExecutor` | `executor.py` | Runs one SubGoal via `AgentRunner` with scoped context only |
| `PostconditionVerifier` | `verifier.py` | LLM call verifying a SubGoal's output meets its postcondition |
| `Replanner` | `replanner.py` | Regenerates only downstream unfinished SubGoals on failure (partial replan) |
| `LookaheadSimulator` | `lookahead.py` | Optional: generates N candidate execution paths per SubGoal (advisory only) |
| `Plan` | `types.py` | Validated SubGoal DAG with `build_waves()` (topological sort via Kahn's) |
| `SubGoal` | `types.py` | One node in the plan: id, description, dependencies, fallback_strategy, postcondition |
| `PlanningConfig` | `types.py` | Tuning knobs: complexity_threshold, max_retries, enable_lookahead, max_replan_attempts |
| `PlanResult` | `types.py` | Output: final_output, subgoal_outputs, plan_used, replanning_count, total_token_usage |
| `SubGoalStatus` | `types.py` | Enum: `PENDING, RUNNING, COMPLETED, FAILED, SKIPPED` |

---

## How planning works

```
PlanningRunner.run(task, session_id)
    │
    ▼
Complexity estimate (1 cheap LLM call)
    │  estimated_tool_calls ≤ complexity_threshold?
    │  Yes → delegate directly to AgentRunner (no planning overhead, ~40% of tasks)
    │  No  → proceed to full planning
    │
    ▼
Planner.create_plan(task)
    │  1 LLM call → SubGoal DAG (JSON)
    │  Validation: unique IDs, no missing deps, cycle detection (Kahn's) →
    │  PlanningError(code="CIRCULAR_DEPENDENCY") on cycle
    │
    ▼
Plan.build_waves() → list[list[SubGoal]] (topological sort)
    Wave 0: subgoals with no dependencies
    Wave 1: subgoals whose all deps completed, etc.
    │
    ▼
For each wave (sequential between waves):
    │  asyncio.gather(execute_subgoal(sg) for sg in wave)  ← parallel within wave
    │  │
    │  ▼
    │  SubGoalExecutor.run(subgoal, completed_summaries, agent_runner)
    │      Scoped context = global_task + one-line summaries of completed steps + subgoal desc
    │      Full conversation history is NEVER passed
    │      1 AgentRunner.run() call per subgoal
    │  │
    │  ▼
    │  PostconditionVerifier.verify(subgoal, output)  ← 1 LLM call (fast model)
    │      PASS → mark SubGoal.status = COMPLETED
    │      FAIL → try fallback_strategy once (if specified)
    │           → trigger Replanner if fallback also fails
    │
    ▼
Replanner.replan(plan, failed_subgoal, completed_outputs)
    │  Only regenerates downstream UNFINISHED subgoals (partial replan)
    │  Completed subgoals and their outputs are preserved
    │  Raises PlanningError if max_replan_attempts exceeded
    │
    ▼
Synthesis: 1 LLM call combining all subgoal outputs into final answer
    │
    ▼
PlanResult
```

---

## Usage

```python
from grampus.orchestration.planning.runner import PlanningRunner
from grampus.orchestration.planning.types import PlanningConfig

plan_runner = PlanningRunner(
    agent_runner=runner,
    model_client=client,
    model_id="claude-haiku-4-5-20251001",
    config=PlanningConfig(
        complexity_threshold=5,    # delegate to AgentRunner if ≤5 tool calls expected
        max_retries=2,             # retries per subgoal
        enable_lookahead=False,    # lookahead adds LLM calls; off by default
        max_replan_attempts=2,
    ),
    cost_tracker=ct,
)

result = await plan_runner.run(
    task="Research and write a comprehensive competitive analysis of the agentic AI market",
    session_id="sess-abc",
)

print(result.final_output)
print(f"Replanning occurred: {result.replanning_count} times")
```

### As a Graph node

```python
from grampus.orchestration.planning.types import planning_node
graph.add_node("plan_execute", planning_node(plan_runner), entry=True)
```

---

## Context scoping (why it reduces token usage by 82%)

Each `SubGoalExecutor.run()` call passes to `AgentRunner`:
- The **global task description** (one paragraph)
- **One-line summaries** of all completed subgoals (not full outputs)
- The **current subgoal description** only

The full conversation history from previous subgoals is never passed. This confines error propagation: a hallucination in SubGoal 2 cannot infect SubGoal 5's reasoning because SubGoal 5 only sees "SubGoal 2 completed: <one-line summary>".

---

## Hard invariants

- **`AgentRunner` is unchanged.** `PlanningRunner` wraps it. Any existing `AgentRunner` instance works as the subgoal executor without modification.
- **`Plan` validation runs at creation time** — cycle detection via Kahn's algorithm. A `PlanningError(code="CIRCULAR_DEPENDENCY")` is raised immediately if cycles are found, before any subgoal runs.
- **Partial replanning only** — completed subgoal outputs are never discarded on failure. Only `PENDING` and `FAILED` downstream nodes are regenerated.
- **`LookaheadSimulator` failures are silently swallowed** — parse failures and LLM errors are suppressed. Execution continues without the lookahead hint.
- **`PostconditionVerifier` makes one LLM call per subgoal.** With the fast model tier this is negligible relative to subgoal execution cost. However, on long plans (10+ subgoals) this adds up — consider using `model_id="claude-haiku-4-5-20251001"`.

---

## Dependency map

```
planning/ depends on:     core/ (types, errors, logging), orchestration/runner.py
planning/ is imported by: orchestration/graph.py (via planning_node())
planning/ must NOT import from: memory/ (injected via AgentRunner), tools/,
                               safety/, evaluation/
```

---

## ADR references

- **ADR-014** — Long-horizon planning: full design rationale and research basis
