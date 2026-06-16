# Long-Horizon Planning

Long-Horizon Planning (Phase E34) lets a Grampus agent break a complex, multi-step task into a structured DAG of subgoals, execute them — in parallel where possible — and automatically recover from failures without restarting from scratch. Use it when a task requires more than 4–5 tool calls, when subgoal dependencies matter for correctness, or when you need confident progress tracking across long-running work.

---

## When to use planning vs. ReAct

| | [AgentRunner (ReAct)](single-agent.md) | PlanningRunner |
|---|---|---|
| **Best for** | Conversational turns, short tasks | Multi-step research, pipelines, batch jobs |
| **Structure** | Greedy step-by-step | Structured DAG of verified subgoals |
| **Failure handling** | Hits `max_iterations` | Retry → fallback → partial replan |
| **Token cost** | Full history passed every call | Scoped context per subgoal (82% reduction) |
| **Parallelism** | Sequential tool calls | Independent subgoals run concurrently |
| **Overhead** | None | 1–2 extra LLM calls for complexity check + synthesis |

The runner automatically detects simple tasks and delegates directly to the underlying `AgentRunner` without planning overhead (see [Adaptive routing](#adaptive-routing)).

---

## Prerequisites

```bash
pip install "grampus-ai[anthropic]"   # or openai
```

No additional dependencies — planning uses only the LLM client you already configure.

---

## Minimal example

```python
import asyncio
from grampus.core.models.anthropic import AnthropicClient
from grampus.core.types import AgentDefinition
from grampus.orchestration import AgentRunner, PlanningRunner, PlanningConfig
from grampus.tools.executor import ToolExecutor

async def main():
    client = AnthropicClient(api_key="...")
    executor = ToolExecutor(registry=...)   # your tool registry
    agent_runner = AgentRunner(client, executor)

    planner = PlanningRunner(
        agent_runner=agent_runner,
        model_client=client,
        model_id="claude-opus-4-7",   # powerful model for planning
        config=PlanningConfig(
            complexity_threshold=4,       # skip planning for simple tasks
            max_subgoals=10,
            max_replans=3,
            enable_lookahead=True,        # FLARE-style path simulation
            enable_parallel_subgoals=True,
        ),
    )

    agent_def = AgentDefinition(
        name="research-agent",
        model="claude-sonnet-4-6",
        system_prompt="You are a research assistant.",
        max_iterations=8,
    )

    result = await planner.run(
        "Research the top 5 Python async frameworks, compare their performance benchmarks, "
        "and write a summary with a recommendation.",
        agent_def,
        tool_names=["web_search", "read_url", "write_file"],
    )

    print(result.final_output)
    print(f"Subgoals completed: {result.completed_subgoals}")
    print(f"Replans triggered:  {result.replans_triggered}")

asyncio.run(main())
```

---

## Architecture

```
User task
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Complexity gate (1 cheap LLM call)                 │
│  estimated_steps ≤ threshold → AgentRunner directly │
│  estimated_steps > threshold → full planning        │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Planner (powerful model)                           │
│  • Generates SubGoal DAG from task + tool list      │
│  • Validates: unique IDs, no cycles, valid deps     │
│  • Topological sort → parallel execution waves      │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │  Wave 0 (no deps)       │  Wave 1 (deps met)  ...
              │  SubGoal A  SubGoal B   │  SubGoal C
              │  (asyncio.gather)       │
              └────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  For each subgoal:      │
              │  1. LookaheadSimulator  │  ← FLARE path scoring
              │  2. AgentRunner (scoped │  ← TDP: only task +
              │     context only)       │    completed summaries
              │  3. PostconditionVerify │  ← pass / partial / fail
              │  4. Retry if partial    │
              │  5. Try fallback if fail│
              └────────────┬────────────┘
                           │ subgoal FAILED after all retries?
                           ▼
              ┌────────────────────────┐
              │  Replanner             │  ← partial replan only
              │  Preserves completed   │    (Google DeepMind design)
              │  Generates new subgoals│
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Synthesis (1 call)    │  ← combine all outputs
              └────────────┬───────────┘
                           │
                           ▼
                       PlanResult
```

---

## Adaptive routing

The planning layer first estimates task complexity with a single cheap LLM call. If the estimated tool-call count is at or below `PlanningConfig.complexity_threshold` (default `4`), the runner delegates directly to `AgentRunner.run()` without creating a plan. This eliminates planning overhead on simple conversational tasks.

```python
config = PlanningConfig(
    complexity_threshold=4,   # tasks needing ≤4 tool calls skip planning
)
```

Change `complexity_threshold` to tune the break-even point for your workload.

---

## Scoped context (TDP)

Each subgoal executor receives a fresh context containing only:

1. **Global task description** — one sentence for orientation
2. **Completed subgoal summaries** — one line per finished step: `- id: output_summary`
3. **Current subgoal** — description + verifiable success criterion

The full conversation history is **not** passed. This is the core of Task-Decoupled Planning (arXiv 2601.07577) and reduces token usage by ~82% on long plans while confining error propagation to the active subgoal.

---

## Lookahead path simulation

When `enable_lookahead=True` (the default), the runner generates `lookahead_paths` candidate execution approaches before committing to each subgoal. The approach with the highest estimated success score is injected as a hint into the subgoal executor's prompt.

This is a lightweight version of the FLARE trajectory simulation from "Why Reasoning Fails to Plan" (arXiv 2601.22311). Lookahead is advisory: if parsing fails for any reason, the executor proceeds without a hint — it never blocks execution.

```python
config = PlanningConfig(
    enable_lookahead=True,
    lookahead_paths=2,   # number of candidate paths per subgoal
)
```

---

## Retry and fallback logic

For each subgoal, the executor runs this control flow:

```
execute() → verify()
    PASS   → subgoal COMPLETED
    PARTIAL → retry (up to max_retries times)
    FAIL   → try fallback_strategy (one attempt)
           → if still fails: subgoal FAILED → trigger Replanner
```

The `fallback_strategy` field on a `SubGoal` is a plain-English description of an alternative approach the LLM should try if the primary strategy fails. The planner populates it automatically; you can also set it explicitly when constructing subgoals for tests or manual plans.

---

## Partial replanning

When a subgoal fails after all retries and its fallback, the `Replanner` is called. It receives:

- The original task
- All **completed** subgoals and their outputs (unchanged)
- The **failed** subgoal and its failure reason
- The **remaining** planned subgoals (now invalidated)

The replanner generates only the new downstream subgoals — completed work is preserved. This is based on the Google DeepMind Subgoal Framework (arXiv 2603.19685) which shows that partial replanning preserves completed work and reduces cost vs. full replan.

```python
config = PlanningConfig(
    max_replans=3,   # hard cap on replan cycles before raising PlanningError
)
```

If `max_replans` is reached, `PlanningError(code="MAX_REPLANS_EXCEEDED")` is raised.

---

## Using `planning_node` in a Graph

Wrap `PlanningRunner` as a graph node for composable multi-step pipelines:

```python
from grampus.orchestration import Graph, planning_node, human_node

handler = planning_node(
    planning_runner=planner,
    agent_def=agent_def,
    tool_names=["web_search", "write_file"],
    memory_context_key="memory_context",   # reads from state.metadata
)

async def route(state):
    plan = state.metadata.get("plan_result", {})
    return "review" if not plan.get("success") else "end"

graph = (
    Graph(graph_id="research-pipeline")
    .add_node("plan", handler, entry=True)
    .add_conditional_edge("plan", route, {"review": "review", "end": None})
    .add_node("review", human_node("Planning failed — please review."))
)
```

The node appends an ASSISTANT message with `final_output` and stores the full `PlanResult` dict in `state.metadata["plan_result"]`. State status is set to `COMPLETED` on success, `FAILED` if `PlanResult.success=False`.

---

## PlanResult fields

| Field | Type | Description |
|-------|------|-------------|
| `task` | `str` | Original user task |
| `plan` | `Plan` | Final plan version executed (may be a replan) |
| `final_output` | `str` | Synthesized answer from all completed subgoals |
| `completed_subgoals` | `list[str]` | IDs of successfully completed subgoals |
| `failed_subgoals` | `list[str]` | IDs of subgoals that could not be completed |
| `replans_triggered` | `int` | Number of replan cycles that occurred |
| `total_token_usage` | `TokenUsage \| None` | Accumulated token usage |
| `duration_seconds` | `float` | Wall-clock duration |
| `success` | `bool` | `True` when all subgoals completed without failures |

---

## Cost model

| Call type | When | Model tier |
|-----------|------|------------|
| Complexity estimate | Once per run | `fast` |
| Plan creation | Once per run (+ once per replan) | `powerful` |
| Lookahead | Once per subgoal (if enabled) | `fast` |
| Subgoal execution | Once+ per subgoal (via AgentRunner) | `balanced` |
| Verification | Once+ per subgoal | `fast` |
| Synthesis | Once per run | `balanced` |

For a 6-subgoal plan with no replanning and lookahead enabled: roughly 14 LLM calls total (1 complexity + 1 plan + 6 lookahead + 6 verify + 1 synthesis = 15, minus any subgoal internal calls that short-circuit).

Wire in a `CostTracker` to get a full accounting:

```python
from grampus.orchestration import CostTracker

tracker = CostTracker(agent_id="research-agent", session_id="s1", budget_usd=0.50)
planner = PlanningRunner(agent_runner, client, model_id, cost_tracker=tracker)
```

---

## Research basis

| Design decision | Source |
|-----------------|--------|
| Greedy step selection fails on long horizons | "Why Reasoning Fails to Plan", arXiv 2601.22311 (Jan 2026) |
| Scoped context reduces tokens 82% | Task-Decoupled Planning (TDP), arXiv 2601.07577 (Jan 2026) |
| Fallback before replanning doubles success rate | ReAcTree, arXiv 2511.02424 (AAMAS 2026) |
| Partial replan, preserve completed subgoals | Google DeepMind Subgoal Framework, arXiv 2603.19685 (Mar 2026) |
| Adaptive engagement avoids overhead on simple tasks | "Learning When to Plan", arXiv 2509.03581 |
| DAG structure enables parallel subgoal execution | TDP + ReAcTree |

See also the [Orchestration API reference](../reference/orchestration-api.md#long-horizon-planning) for the full type reference.
