# `grampus/orchestration/` — Orchestration Engine

This package owns agent execution: the ReAct loop (`AgentRunner`), graph-based workflow execution (`Graph`, `Crew`), model selection (`ModelRouter`), cost tracking, and four advanced orchestration primitives (debate, uncertainty quantification, long-horizon planning, artifact collaboration).

`AgentRunner` is the central hub that wires together every other framework layer — memory, tools, safety, observability, and all optional advanced features.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `AgentRunner` | `runner.py` | Main ReAct loop: input → memory → LLM → tools → memory → respond |
| `RunnerConfig` | `runner.py` | Tuning knobs: max_iterations, memory_top_k, enable_memory, react_pattern |
| `Graph` | `graph.py` | DAG-based workflow engine with checkpointing |
| `LLMNode` | `nodes.py` | Pre-built graph node: LLM call → AgentState update |
| `ToolNode` | `nodes.py` | Pre-built graph node: tool call |
| `ConditionalNode` | `nodes.py` | Branches based on `AgentState` field |
| `HumanNode` | `nodes.py` | Pause + wait for human input |
| `ModelRouter` | `model_router.py` | Routes steps to cheapest capable model tier |
| `CostTracker` | `cost_tracker.py` | Per-step token/cost tracking; budget enforcement |
| `Crew` | `crew.py` | Multi-agent orchestration (sequential, parallel, hierarchical) |
| `HandoffExecutor` | `handoff.py` | A2A agent-to-agent handoffs |
| `AgentStateSnapshot` | `snapshot.py` | Checkpoint/restore of `AgentState` across restarts |

**Advanced sub-packages:**
- [debate/](debate/) — Multi-agent debate with convergence detection (ADR-012)
- [uncertainty/](uncertainty/) — Dual-process uncertainty quantification (ADR-013)
- [planning/](planning/) — Long-horizon structured planning (ADR-014)
- [artifact/](artifact/) — Artifact-centric collaborative document creation (ADR-015)

---

## `AgentRunner` — the execution loop

### Minimal usage

```python
from grampus.orchestration.runner import AgentRunner, RunnerConfig
from grampus.core.types import AgentDefinition

agent_def = AgentDefinition(
    name="researcher",
    model="claude-haiku-4-5-20251001",
    system_prompt="You are a research assistant.",
    tools=registry.to_definitions(),
)

runner = AgentRunner(
    model_client=client,
    tool_executor=executor,
    config=RunnerConfig(max_iterations=10),
)

result = await runner.run(
    agent_def=agent_def,
    user_input="Summarize the latest news on agentic AI",
    session_id="sess-abc",
)
# result.output, result.tool_calls_made, result.token_usage, result.steps_taken
```

### With all optional layers

```python
runner = AgentRunner(
    model_client=client,
    tool_executor=executor,
    memory_manager=mm,              # Phase 3-5
    cost_tracker=ct,                # Phase 7b
    state_store=dapr_store,         # for Graph checkpoints
    safety_pipeline=safety,         # Phase 8
    tracer=tracer,                  # Phase 9
    event_log=event_log,            # Phase 9
    grampus_metrics=metrics,        # Phase 9
    uncertainty_monitor=uq,         # E33 / ADR-013
    reflexion_engine=reflexion,     # F1 / ADR-016
    skill_library=skill_lib,        # F1 / ADR-016
    user_memory_adapter=user_mem,   # F2 / ADR-017
    graph_builder=graph_builder,    # F3 / ADR-018
    causal_world_model=world_model, # F4 / ADR-019
    causal_tracer=causal_tracer,    # F4 / ADR-019
    plugin_manager=plugins,         # H49 / ADR-024
    version_router=ver_router,      # H50 / ADR-025
)
```

All optional parameters default to `None`. Adding any optional layer does not change the observable behavior for callers that don't use it — only additional hooks fire.

---

## ReAct loop (AgentRunner.run)

```
1. Load or create AgentState
2. [if user_memory_adapter] inject user context into system prompt
3. [if memory_manager] recall relevant memories, inject into messages
4. [if uncertainty_monitor] initialize belief state

LOOP (max_iterations):
    5. [if safety_pipeline] check_input(messages)
    6. [if plugin_manager] pre_llm_call hook
    7. model_client.complete(messages, tools) → ModelResponse
    8. [if event_log] emit LLM_CALL event
    9. [if cost_tracker] record tokens + check budget
    10. [if uncertainty_monitor] observe_llm_response → UncertaintyAction
        → ABORT → raise UncertaintyError
        → PAUSE → inject reflection prompt, continue
        → PROCEED → continue
    11. If no tool_calls in response → break (final answer)
    12. [if safety_pipeline] check_llm_output(response)
    13. [if plugin_manager] pre_tool_call hook
    14. Execute each tool call via ToolExecutor
    15. [if safety_pipeline] check_tool_result(result)
    16. [if event_log] emit TOOL_CALL event
    17. [if memory_manager] remember tool results
    18. [if graph_builder] append_event(tool_result)
    19. [if causal_world_model] infer relationships

POST-LOOP:
    20. [if memory_manager] remember final answer
    21. [if reflexion_engine] observe_success or observe_failure
    22. [if skill_library] observe_success
    23. [if user_memory_adapter] extract_from_session
    24. [if causal_tracer] on failure: diagnose root cause
    25. Return ExecutionResult
```

---

## Graph-based workflows

```python
from grampus.orchestration.graph import Graph
from grampus.orchestration.nodes import LLMNode, ToolNode, ConditionalNode

graph = (
    Graph(graph_id="research-pipeline", state_store=dapr_store)
    .add_node("plan", LLMNode(client, system="Make a plan"), entry=True)
    .add_node("search", ToolNode(executor, "web_search"))
    .add_node("summarize", LLMNode(client, system="Summarize the results"))
    .add_edge("plan", "search")
    .add_edge("search", "summarize")
)

final_state = await graph.run(initial_state)
```

**Checkpointing:** after each node completes, `AgentState` is persisted to Dapr. If the process crashes mid-graph, `graph.resume(session_id)` loads the last checkpoint and continues from that node.

---

## Multi-agent Crew

```python
from grampus.orchestration.crew import Crew

crew = Crew(
    agents=[researcher, writer, reviewer],
    pattern="sequential",   # or "parallel" or "hierarchical"
    shared_state_store=dapr_store,
    lock_store=dapr_lock_store,
)
result = await crew.run(task="Write a comprehensive report on agentic AI")
```

For collaborative document creation, see [artifact/](artifact/) — `ArtifactCrew` is the preferred pattern when agents need to produce structured outputs.

---

## Cost tracking and model routing

```python
from grampus.orchestration.model_router import ModelRouter, RoutingTier

router = ModelRouter(
    fast_model="claude-haiku-4-5-20251001",
    balanced_model="claude-sonnet-4-6",
    powerful_model="claude-opus-4-8",
)

model_id = router.select(tier=RoutingTier.FAST)   # for planning steps
model_id = router.select(tier=RoutingTier.POWERFUL) # for synthesis steps
```

```python
from grampus.orchestration.cost_tracker import CostTracker

tracker = CostTracker(budget_usd=5.0)
tracker.record(token_usage)
await tracker.check_budget()  # raises BudgetExceededError if over limit
```

---

## Hard invariants

- **`AgentRunner.run()` catches all exceptions from optional hooks** via `contextlib.suppress`. A broken `reflexion_engine` or `user_memory_adapter` never propagates to the caller. Domain errors (tool failures, LLM errors, budget exceeded) DO propagate.
- **`Graph` checkpoints to Dapr after every node completion.** If `state_store=None`, checkpointing is skipped (in-memory only). Production deployments must provide a `state_store`.
- **`CostTracker.check_budget()` raises `BudgetExceededError` — this IS a domain error and propagates.** The caller (usually the user's code) must catch it and decide whether to retry, escalate, or abort.
- **`max_iterations` is a hard ceiling, not a soft hint.** When the loop reaches `max_iterations`, `AgentRunner` returns the best partial result with `status=AgentStatus.COMPLETED` and `steps_taken=max_iterations`.
- **`AgentRunner` does not own the model client or tool executor** — they are injected. This enables testing with mock clients and ensures the runner has no implicit global state.

---

## Extension guide

### Adding a new pre-built Graph node

1. Create a class in `nodes.py` with `async def __call__(self, state: AgentState) -> AgentState`.
2. Register it with a `Graph` via `.add_node(name, MyNode(...))`.

### Adding a new optional hook to AgentRunner

1. Add a new optional parameter (`my_hook: MyHook | None = None`) to `AgentRunner.__init__`.
2. Add `if self._my_hook:` guards in the appropriate places in the loop.
3. Wrap the call in `contextlib.suppress(Exception)` if it should never crash the runner.
4. Emit a structlog event inside the block.

---

## Dependency map

```
orchestration/ depends on: core/, dapr/, memory/, tools/, safety/,
                           observability/, causal/, plugins/, versioning/
orchestration/ is imported by: cli/, evaluation/ (runner fixtures)
orchestration/ must NOT import from: evaluation/ (circular)
```

---

## ADR references

- **ADR-004** — Async-first architecture (all nodes are `async def`)
- **ADR-005** — Event sourcing (`AgentRunner` emits every action to `EventLog`)
- **ADR-009** — Code agents as primary pattern (via `CodeExecutor` in tools)
- **ADR-012** — Multi-agent debate (→ `debate/`)
- **ADR-013** — Dual-process uncertainty quantification (→ `uncertainty/`)
- **ADR-014** — Long-horizon planning (→ `planning/`)
- **ADR-015** — Artifact-centric collaboration (→ `artifact/`)
