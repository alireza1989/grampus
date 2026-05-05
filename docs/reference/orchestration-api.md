# Orchestration API Reference

## AgentRunner

The main agent execution loop implementing the ReAct (Reason+Act) pattern.

::: nexus.orchestration.runner.AgentRunner
    options:
      show_source: false
      members: [run, resume, cost_summary]

---

## RunnerConfig

```python
from nexus.orchestration.runner import RunnerConfig

config = RunnerConfig(
    max_iterations=10,      # max ReAct iterations before OrchestrationError
    memory_top_k=5,         # episodic/semantic results per recall query
    enable_memory=True,     # enable memory read/write during runs
    react_pattern=True,     # use ReAct loop (vs. single-shot)
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iterations` | `int` | `10` | Abort with `OrchestrationError` if exceeded |
| `memory_top_k` | `int` | `5` | Results per `MemoryManager.recall()` call |
| `enable_memory` | `bool` | `True` | Whether to read/write memory during the loop |
| `react_pattern` | `bool` | `True` | Use ReAct; set `False` for single-shot LLM calls |

---

## Crew

Multi-agent orchestration with sequential, parallel, and hierarchical patterns.

::: nexus.orchestration.crew.Crew
    options:
      show_source: false
      members: [run]

---

## CrewPattern

```python
from nexus.orchestration.crew import CrewPattern

CrewPattern.SEQUENTIAL    # agents run one after another; outputs accumulate
CrewPattern.PARALLEL      # agents run concurrently on the same input
CrewPattern.HIERARCHICAL  # first member (role="supervisor") delegates to workers
```

---

## CrewMember

```python
from nexus.orchestration.crew import CrewMember

member = CrewMember(
    agent_def=AgentDefinition(...),
    runner=AgentRunner(...),
    role="researcher",       # semantic label; "supervisor" triggers hierarchical delegation
)
```

---

## CrewResult

Returned by `Crew.run()`:

```python
@dataclass
class CrewResult:
    outputs: dict[str, str]    # agent_name → output string
    total_cost_usd: float
    duration_seconds: float
    pattern: CrewPattern
```

---

## Graph engine

::: nexus.orchestration.graph.Graph
    options:
      show_source: false
      members: [add_node, add_edge, run]

::: nexus.orchestration.nodes.llm_node
    options:
      show_source: false

::: nexus.orchestration.nodes.tool_node
    options:
      show_source: false

::: nexus.orchestration.nodes.conditional_node
    options:
      show_source: false

::: nexus.orchestration.nodes.human_node
    options:
      show_source: false

---

## Model router

::: nexus.orchestration.model_router.ModelRouter
    options:
      show_source: false
      members: [route, register_model]

### Model tiers

| Tier | Use case | Example models |
|------|----------|---------------|
| `fast` | Simple classification, routing | claude-haiku-4-5 |
| `balanced` | Standard agent tasks | claude-sonnet-4-6 |
| `powerful` | Complex reasoning, synthesis | claude-opus-4-7 |

---

## Cost tracker

::: nexus.orchestration.cost_tracker.CostTracker
    options:
      show_source: false
      members: [record, summary, check_budget]

### CostSummary

```python
@dataclass
class CostSummary:
    total_cost_usd: float
    total_tokens: int
    by_model: dict[str, float]      # model → USD
    by_step: list[StepCost]         # per-iteration breakdown
    budget_usd: float | None
    budget_remaining_usd: float | None
```

---

## ExecutionResult

Returned by `AgentRunner.run()`:

::: nexus.core.types.ExecutionResult
    options:
      show_source: false
      members: []

---

## AgentDefinition

Blueprint passed to `AgentRunner.run()`:

::: nexus.core.types.AgentDefinition
    options:
      show_source: false
      members: []

---

## AgentState

Mutable runtime state maintained by the runner:

::: nexus.core.types.AgentState
    options:
      show_source: false
      members: []
