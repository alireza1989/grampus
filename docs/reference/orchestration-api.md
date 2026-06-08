# Orchestration API Reference

## Multi-Agent Debate

### DebateOrchestrator

```python
from nexus.orchestration.debate import DebateOrchestrator, DebateConfig, DebaterConfig

orch = DebateOrchestrator(
    config=DebateConfig(...),
    cost_tracker=None,   # optional CostTracker
    tracer=None,         # optional NexusTracer or any span(name, **attrs) tracer
)
result = await orch.run("question text")
```

::: nexus.orchestration.debate.orchestrator.DebateOrchestrator
    options:
      show_source: false
      members: [run]

### DebateConfig

```python
from nexus.orchestration.debate import DebateConfig, AggregationStrategy

cfg = DebateConfig(
    debaters=[...],                        # min 2 DebaterConfig entries
    max_rounds=3,
    aggregation=AggregationStrategy.WEIGHTED_VOTE,
    convergence_threshold=0.8,             # stop early when this fraction agrees
    adaptive_routing=True,                 # skip debate if first model is confident
    routing_confidence_threshold=0.85,
    escalate_threshold=0.5,                # set escalate_to_human when below this
)
```

| Field | Type | Default |
|-------|------|---------|
| `debaters` | `list[DebaterConfig]` | required (≥ 2) |
| `max_rounds` | `int` | `3` |
| `aggregation` | `AggregationStrategy` | `WEIGHTED_VOTE` |
| `convergence_threshold` | `float` | `0.8` |
| `adaptive_routing` | `bool` | `True` |
| `routing_confidence_threshold` | `float` | `0.85` |
| `routing_model_client` | `ModelClient \| None` | `None` → debaters[0] |
| `routing_model_id` | `str` | `""` → debaters[0] |
| `judge_config` | `DebaterConfig \| None` | `None` |
| `cost_budget_usd` | `float \| None` | `None` |
| `escalate_threshold` | `float` | `0.5` |

### DebaterConfig

```python
from nexus.orchestration.debate import DebaterConfig

cfg = DebaterConfig(
    model_client=client,         # any Nexus ModelClient
    model_id="claude-sonnet-4-6",
    temperature=0.7,
    role_hint="You are a skeptical devil's advocate.",   # optional persona
    weight=1.0,                  # vote weight for WEIGHTED_VOTE aggregation
)
```

### AggregationStrategy

```python
from nexus.orchestration.debate import AggregationStrategy

AggregationStrategy.MAJORITY_VOTE   # largest Jaccard cluster, highest-confidence rep
AggregationStrategy.WEIGHTED_VOTE   # clusters scored by debater.weight × confidence
AggregationStrategy.JUDGE           # separate judge model synthesises all positions
```

### DebateResult

Returned by `DebateOrchestrator.run()`:

| Field | Type | Description |
|-------|------|-------------|
| `final_answer` | `str` | Aggregated winning answer |
| `final_reasoning` | `str` | Reasoning from the winning position |
| `confidence` | `float` | Aggregated confidence (0–1) |
| `escalate_to_human` | `bool` | `True` when `final_convergence_score < escalate_threshold` |
| `rounds` | `list[DebateRound]` | Full per-round transcript |
| `routing_decision` | `RoutingDecision` | `"debate"` or `"single_agent"` |
| `total_rounds_run` | `int` | Rounds actually completed |
| `converged` | `bool` | Whether early stopping triggered |
| `final_convergence_score` | `float` | Convergence score in the final round |
| `total_token_usage` | `TokenUsage` | Cumulative tokens across all rounds |
| `total_cost_usd` | `float` | Total spend |
| `duration_seconds` | `float` | Wall-clock time |

### debate_node

Graph node factory that wraps a `DebateOrchestrator`:

```python
from nexus.orchestration import debate_node, Graph, human_node

handler = debate_node(
    orchestrator,
    question_extractor=None,   # defaults to last USER message content
    on_escalate="human_review",  # metadata flag written when escalate_to_human=True
)

graph = (
    Graph(graph_id="qa")
    .add_node("debate", handler, entry=True)
    .add_conditional_edge("debate", route_fn, {"escalate": "human_review", "end": None})
    .add_node("human_review", human_node("Low confidence — please review."))
)
```

The ASSISTANT message appended by the node carries:

```python
message.metadata["debate_result"]      # full DebateResult serialised as dict
message.metadata["debate_confidence"]  # float
message.metadata["debate_escalate"]    # bool
message.metadata["debate_rounds"]      # int
message.metadata["debate_routing"]     # "debate" | "single_agent"
```

---

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
