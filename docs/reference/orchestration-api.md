# Orchestration API Reference

## Market-Based Allocation

### MarketAllocator

End-to-end allocation pipeline: capability filter → bid solicitation → scoring → award.

```python
from nexus.orchestration.market import (
    MarketAllocator, CapabilityRegistry, TaskBoard,
    BidScorer, ReputationTracker,
)

registry   = CapabilityRegistry(max_candidates=5)
board      = TaskBoard()
reputation = ReputationTracker()
scorer     = BidScorer(reputation)

allocator = MarketAllocator(
    registry=registry,
    board=board,
    scorer=scorer,
    reputation=reputation,
    model_client=client,   # any Nexus ModelClient; used for bid solicitation
    tracer=None,           # optional NexusTracer; emits market.allocate / market.award spans
)

result = await allocator.allocate(spec)          # AllocationResult
await allocator.report_outcome(outcome)          # updates board + reputation
```

::: nexus.orchestration.market.allocator.MarketAllocator
    options:
      show_source: false
      members: [allocate, report_outcome]

### CapabilityRegistry

Stores worker capability profiles with capability-first filtering (COALESCE, arXiv 2506.01900).

```python
from nexus.orchestration.market import CapabilityRegistry, CapabilityProfile, AgentTier

registry = CapabilityRegistry(max_candidates=5)   # default 5

profile = CapabilityProfile(
    agent_id="researcher",
    agent_name="Web Researcher",
    skill_tags=["web_search", "summarize"],
    model_tier=AgentTier.BALANCED,
    cost_per_step_usd=0.002,
    max_steps=10,
)
await registry.register(profile)
await registry.deregister("researcher")

capable = registry.filter_capable(
    required_skills=["web_search"],
    preferred_skills=["summarize"],
)   # → list[CapabilityProfile], ranked by preferred matches, capped at max_candidates
```

::: nexus.orchestration.market.registry.CapabilityRegistry
    options:
      show_source: false
      members: [register, deregister, filter_capable, load_all, list_agents]

### TaskBoard

Durable task and bid store. Backed by Dapr state when a `state_store` is provided.

```python
from nexus.orchestration.market import TaskBoard, TaskSpec, AllocationStatus

board = TaskBoard(state_store=None)   # in-memory; pass DaprStateStore for persistence

task_id = await board.post_task(spec)
await board.submit_bid(bid)
bids    = await board.get_bids_for_task(task_id)
await board.update_task_status(task_id, AllocationStatus.ALLOCATED)
await board.mark_outcome(outcome)    # → COMPLETED or FAILED
```

### ReputationTracker

UCB-based per-agent reputation (DRF, arXiv 2509.05764). Persists to Dapr state.

```python
from nexus.orchestration.market import ReputationTracker, TaskOutcome

tracker = ReputationTracker(state_store=None)

record       = await tracker.get("agent-id")           # ReputationRecord
record       = await tracker.update(outcome)            # → updated ReputationRecord
cal_factor   = await tracker.calibration_factor("agent-id")   # float
ucb          = await tracker.ucb_bonus("agent-id")     # float; decays with history
tracker.record_self_report("agent-id", 0.85)           # feed bid data for calibration
```

UCB formula: `sqrt(2 × ln(max(2, N)) / max(1, n_i))`  
New agents always receive a positive exploration bonus.

### BidScorer

Composite scoring with calibration discount.

```python
from nexus.orchestration.market import BidScorer, ReputationTracker

scorer = BidScorer(
    reputation_tracker=ReputationTracker(),
    alpha=0.35,   # reputation weight
    beta=0.45,    # calibrated success weight
    gamma=0.20,   # cost efficiency weight
    # alpha + beta + gamma must equal 1.0 (ValueError on mismatch)
)

score  = await scorer.score(bid, task_spec)       # BidScore
scores = await scorer.score_all(bids, task_spec)  # list[BidScore], sorted desc
```

Formula:
```
calibrated_success = clamp(raw_prob × calibration_factor, 0, 1)
cost_score         = 1 / (1 + estimated_cost / budget)
composite          = α×reputation + β×calibrated_success + γ×cost_score
final_score        = composite + ucb_bonus
```

Bids with `calibrated_success < min_success_threshold` receive `final_score = -1.0` (moral hazard guard).

### MarketCrew

`Crew` subclass with opt-in market allocation.

```python
from nexus.orchestration.market import MarketCrew

crew = MarketCrew(
    members=members,
    session_id="session-1",
    pattern=CrewPattern.SEQUENTIAL,    # used when use_market=False
    allocator=allocator,               # required when use_market=True
    use_market=True,                   # default False — zero overhead when off
)

# Market path: post → allocate → run → report
result = await crew.run_task_with_market(
    task_description="Summarise the latest AI papers.",
    required_skills=["web_search"],
    preferred_skills=["summarize"],
    budget_usd=0.05,
)

# Standard path (use_market=False): identical to Crew.run()
result = await crew.run(initial_input="Summarise the latest AI papers.")
```

::: nexus.orchestration.market.crew.MarketCrew
    options:
      show_source: false
      members: [run_task_with_market]

### market_node

Graph node factory that runs market allocation as a graph step.

```python
from nexus.orchestration.nodes import market_node

handler = market_node(
    allocator=allocator,
    required_skills=["web_search"],
    budget_usd=0.05,
    node_name="market_allocate",   # used in log messages
)
```

**Reads from state:**
- `state.metadata["task_description"]` — task description string

**Writes to state:**
- `state.metadata["market_winner"]` — winning `agent_id` string (or `None`)
- `state.metadata["market_result"]` — serialized `AllocationResult` dict
- `state.status = AgentStatus.FAILED` when allocation is REJECTED

### Types

#### CapabilityProfile

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | `str` | required | Unique identifier |
| `agent_name` | `str` | required | Human-readable name |
| `skill_tags` | `list[str]` | required | Capability labels used for filtering |
| `model_tier` | `AgentTier` | `BALANCED` | `fast` / `balanced` / `powerful` |
| `cost_per_step_usd` | `float` | `0.0` | Self-reported step cost (used in fallback bid) |
| `max_steps` | `int` | `20` | Maximum steps the agent will attempt |
| `latency_sla_ms` | `int \| None` | `None` | Optional latency SLA commitment |

#### TaskSpec

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_id` | `str` | required | Unique task identifier |
| `description` | `str` | required | Natural language task description |
| `required_skills` | `list[str]` | required | Must-have skills (hard filter) |
| `preferred_skills` | `list[str]` | `[]` | Skills used for ranking (soft filter) |
| `budget_usd` | `float \| None` | `None` | Hard cost cap; `None` = unlimited |
| `min_success_threshold` | `float` | `0.5` | Minimum calibrated probability to accept a bid |
| `deadline_ms` | `int \| None` | `None` | Optional wall-clock budget in milliseconds |
| `allow_partial` | `bool` | `False` | Whether PARTIAL outcome counts as success |

#### Bid

| Field | Type | Description |
|-------|------|-------------|
| `bid_id` | `str` | Unique bid ID (auto-generated UUID) |
| `task_id` | `str` | Task this bid is for |
| `agent_id` | `str` | Bidding agent |
| `self_reported_success_prob` | `float` | Agent's own estimate (0–1); will be discounted |
| `estimated_cost_usd` | `float` | Self-reported cost |
| `estimated_steps` | `int` | Estimated number of steps |
| `rationale` | `str` | One-sentence explanation (from bid solicitation prompt) |

#### BidScore

| Field | Type | Description |
|-------|------|-------------|
| `raw_success_prob` | `float` | Self-reported, before calibration |
| `calibrated_success_prob` | `float` | After `calibration_factor` discount |
| `reputation_score` | `float` | `success_rate` from ReputationTracker (0.5 for new agents) |
| `cost_score` | `float` | `1 / (1 + normalized_cost)` |
| `composite` | `float` | Weighted blend |
| `ucb_bonus` | `float` | UCB1 exploration bonus |
| `final_score` | `float` | `composite + ucb_bonus`; `-1.0` when below threshold |

#### AllocationResult

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | The allocated task |
| `status` | `AllocationStatus` | `ALLOCATED`, `REJECTED`, `BIDDING`, etc. |
| `winning_agent_id` | `str \| None` | Agent that won (None when REJECTED) |
| `winning_bid` | `Bid \| None` | The winning Bid |
| `winning_score` | `BidScore \| None` | The winning BidScore |
| `all_scores` | `list[BidScore]` | All computed scores, sorted descending |
| `capability_filtered_out` | `list[str]` | Agent IDs filtered before bid solicitation |
| `reject_reason` | `str \| None` | Human-readable reason when REJECTED |

#### AllocationStatus

```python
from nexus.orchestration.market import AllocationStatus

AllocationStatus.PENDING    # task posted, no bids yet
AllocationStatus.BIDDING    # bid solicitation in progress
AllocationStatus.ALLOCATED  # winner selected
AllocationStatus.REJECTED   # no capable bidders or all below threshold
AllocationStatus.COMPLETED  # task finished successfully
AllocationStatus.FAILED     # task execution failed
```

#### ReputationRecord

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_tasks` | `int` | `0` | Tasks completed (success or failure) |
| `successful_tasks` | `int` | `0` | Successful completions |
| `success_rate` | `float` | `0.0` | `successful / total` |
| `cost_accuracy` | `float` | `1.0` | EMA of `actual_cost / estimated_cost`; 1.0 = perfect |
| `calibration_factor` | `float` | `1.0` | EMA multiplier for future bid discounting |
| `ucb_confidence` | `float` | `1.0` | Current UCB exploration bonus |

#### TaskOutcome

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | The completed task |
| `agent_id` | `str` | The agent that executed it |
| `actual_success` | `bool` | Whether it succeeded |
| `actual_cost_usd` | `float` | Actual cost incurred |
| `actual_steps` | `int` | Actual steps taken |

See the [Market-Based Allocation guide](../guides/market-based-allocation.md) for full usage and research citations.

---

## Long-Horizon Planning

### PlanningRunner

Top-level orchestrator for structured multi-step task execution.

```python
from nexus.orchestration import PlanningRunner, PlanningConfig

planner = PlanningRunner(
    agent_runner=agent_runner,    # AgentRunner instance
    model_client=client,          # LLM client for planning calls
    model_id="claude-opus-4-7",   # model for planner/verifier/synthesizer
    config=PlanningConfig(
        complexity_threshold=4,
        max_subgoals=12,
        max_replans=3,
        enable_lookahead=True,
        enable_parallel_subgoals=True,
    ),
    cost_tracker=None,   # optional CostTracker
    tracer=None,         # optional NexusTracer or any span(name, **attrs) tracer
)
result = await planner.run(task, agent_def, tool_names=["web_search"], memory_context="")
```

::: nexus.orchestration.planning.runner.PlanningRunner
    options:
      show_source: false
      members: [run]

### PlanningConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_subgoals` | `int` | `12` | Hard cap on subgoals per plan |
| `max_replans` | `int` | `3` | Maximum replan cycles before `PlanningError` |
| `complexity_threshold` | `int` | `4` | Skip planning when estimated steps ≤ this |
| `enable_lookahead` | `bool` | `True` | FLARE-style path simulation before each subgoal |
| `lookahead_paths` | `int` | `2` | Candidate paths generated per lookahead call |
| `enable_parallel_subgoals` | `bool` | `True` | Run independent subgoals via `asyncio.gather` |
| `cost_budget_usd` | `float \| None` | `None` | Hard cost cap across all planning calls |
| `planner_model_tier` | `str` | `"powerful"` | Model tier for plan generation |
| `executor_model_tier` | `str` | `"balanced"` | Model tier for subgoal execution |
| `verifier_model_tier` | `str` | `"fast"` | Model tier for postcondition verification |

### Plan

```python
from nexus.orchestration import Plan, SubGoal

plan = Plan(
    task="original task",
    subgoals=[SubGoal(...)],
    total_estimated_steps=6,
    version=1,   # increments on each replan
)
```

| Field | Type | Description |
|-------|------|-------------|
| `task` | `str` | Original user task |
| `subgoals` | `list[SubGoal]` | Ordered list; DAG implied by `dependencies` |
| `total_estimated_steps` | `int` | Planner's estimate of total tool calls |
| `created_at` | `datetime` | UTC timestamp of plan creation |
| `version` | `int` | Increments on each replan (starts at 1) |

### SubGoal

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | required | Short snake_case slug, ≤ 20 chars |
| `description` | `str` | required | What this step should accomplish |
| `success_criterion` | `str` | required | Verifiable completion condition |
| `dependencies` | `list[str]` | `[]` | IDs of subgoals that must complete first |
| `tool_hints` | `list[str]` | `[]` | Suggested tool names (advisory) |
| `fallback_strategy` | `str` | `""` | Alternative approach if primary fails |
| `max_retries` | `int` | `2` | PARTIAL retries before declaring FAIL |
| `status` | `SubGoalStatus` | `PENDING` | Current execution status |
| `output_summary` | `str` | `""` | 1-2 sentence summary filled after completion |
| `attempts` | `int` | `0` | Total execution attempts so far |
| `failure_reason` | `str` | `""` | Last failure reason (filled on FAIL) |

### SubGoalStatus

```python
from nexus.orchestration import SubGoalStatus

SubGoalStatus.PENDING     # not yet started
SubGoalStatus.RUNNING     # currently executing
SubGoalStatus.COMPLETED   # success criterion met
SubGoalStatus.FAILED      # could not be completed after all retries
SubGoalStatus.SKIPPED     # skipped (e.g. dependency failed)
```

### VerificationResult

Returned by `PostconditionVerifier.verify()` after each subgoal execution:

```python
from nexus.orchestration import VerificationResult

VerificationResult.PASS      # criterion clearly met
VerificationResult.PARTIAL   # progress made; criterion not fully met (retry)
VerificationResult.FAIL      # criterion not met; retry unlikely to help
```

### PlanResult

Returned by `PlanningRunner.run()`:

| Field | Type | Description |
|-------|------|-------------|
| `task` | `str` | Original user task |
| `plan` | `Plan` | Final plan version executed |
| `final_output` | `str` | Synthesized answer from all completed subgoals |
| `completed_subgoals` | `list[str]` | IDs of successfully completed subgoals |
| `failed_subgoals` | `list[str]` | IDs of subgoals that could not be completed |
| `replans_triggered` | `int` | Number of replan cycles that occurred |
| `total_token_usage` | `TokenUsage \| None` | Accumulated token usage |
| `duration_seconds` | `float` | Wall-clock duration |
| `success` | `bool` | `True` when all subgoals completed |

### planning_node

Graph node factory wrapping a `PlanningRunner`:

```python
from nexus.orchestration import planning_node, Graph, human_node

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
    Graph(graph_id="pipeline")
    .add_node("plan", handler, entry=True)
    .add_conditional_edge("plan", route, {"review": "review", "end": None})
    .add_node("review", human_node("Planning failed — please review."))
)
```

The ASSISTANT message appended by the node carries:

```python
message.metadata["plan_result"]           # full PlanResult serialised as dict
message.metadata["replans_triggered"]     # int
message.metadata["subgoals_completed"]    # int
```

See the [Long-Horizon Planning guide](../guides/long-horizon-planning.md) for full usage and research citations.

---

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

## Uncertainty Quantification

### UncertaintyMonitor

Session-level uncertainty tracker implementing Dual-Process AUQ. Attach to `AgentRunner` via `uncertainty_monitor=monitor`.

```python
from nexus.orchestration import UncertaintyMonitor, UncertaintyPolicy, UncertaintyEstimator

policy = UncertaintyPolicy(
    low_threshold=0.80,
    medium_threshold=0.60,
    high_threshold=0.40,
    enable_p_true=True,
    enable_semantic_sampling=False,
    irreversible_tool_names=["send_email", "delete", "deploy"],
    inject_reflection_on_high=True,
)
monitor = UncertaintyMonitor(policy=policy)
runner = AgentRunner(client, executor, uncertainty_monitor=monitor)
```

::: nexus.orchestration.uncertainty.monitor.UncertaintyMonitor
    options:
      show_source: false
      members: [initialize, observe_llm_response, observe_tool_call, get_belief_state, summary_metadata, reset]

### UncertaintyPolicy

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `low_threshold` | `float` | `0.80` | Confidence floor for LOW → PROCEED |
| `medium_threshold` | `float` | `0.60` | Floor for MEDIUM → PROCEED_WITH_LOG |
| `high_threshold` | `float` | `0.40` | Floor for HIGH → PAUSE_FOR_HUMAN |
| `enable_p_true` | `bool` | `True` | Run P(True) follow-up call |
| `enable_semantic_sampling` | `bool` | `False` | Enable adaptive semantic entropy slow path |
| `irreversible_tool_names` | `list[str]` | `[]` | Tool name substrings triggering PAUSE at MEDIUM |
| `inject_reflection_on_high` | `bool` | `True` | Inject System-2 reflection before PAUSE |

### UncertaintyEstimator

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `verbalized_weight` | `float` | `0.4` | Fusion weight for verbalized signal |
| `p_true_weight` | `float` | `0.6` | Fusion weight for P(True) signal |
| `verbalized_calibration_bias` | `float` | `0.25` | ECE correction for verbalized (documented ECE ≥ 0.377) |
| `p_true_calibration_bias` | `float` | `0.10` | ECE correction for P(True) |
| `min_samples` | `int` | `2` | Adaptive entropy: start sample count |
| `max_samples` | `int` | `5` | Adaptive entropy: extend on disagreement |
| `early_stop_jaccard` | `float` | `0.60` | First-pair agreement threshold for early stop |
| `semantic_trigger_low` | `float` | `0.50` | Lower bound of sampling trigger zone |
| `semantic_trigger_high` | `float` | `0.72` | Upper bound of sampling trigger zone |

### UncertaintyLevel

```python
from nexus.orchestration import UncertaintyLevel

UncertaintyLevel.LOW       # ≥ low_threshold
UncertaintyLevel.MEDIUM    # ≥ medium_threshold
UncertaintyLevel.HIGH      # ≥ high_threshold
UncertaintyLevel.CRITICAL  # < high_threshold
```

### UncertaintyAction

```python
from nexus.orchestration import UncertaintyAction

UncertaintyAction.PROCEED            # run continues
UncertaintyAction.PROCEED_WITH_LOG   # run continues; warning logged
UncertaintyAction.PAUSE_FOR_HUMAN    # status=WAITING_FOR_HUMAN
UncertaintyAction.ABORT              # UncertaintyError raised
```

### StepUncertainty

Returned by `observe_llm_response()` and `observe_tool_call()`:

| Field | Type | Description |
|-------|------|-------------|
| `step_id` | `str` | Unique step identifier |
| `step_type` | `str` | `"llm_call"`, `"tool_call"`, `"memory_read"`, `"decision"` |
| `verbalized_confidence` | `float` | Raw extracted confidence (before calibration) |
| `p_true_confidence` | `float` | P(True) result; `-1.0` when not run |
| `fused_confidence` | `float` | Calibrated weighted fusion |
| `propagated_confidence` | `float` | After SAUP propagation through prior steps |
| `level` | `UncertaintyLevel` | Classified tier |
| `action` | `UncertaintyAction` | Control action taken |
| `samples_used` | `int` | Semantic entropy samples drawn (0 = not run) |
| `reflection_injected` | `bool` | Whether System-2 reflection was injected |

### uncertainty_guard_node

Graph node factory for explicit uncertainty checkpoints:

```python
from nexus.orchestration import uncertainty_guard_node, Graph, human_node

handler = uncertainty_guard_node(
    monitor,
    step_type="decision",          # SAUP weight lookup
    escalate_node="human_review",  # sets metadata["uncertainty_escalate"]=True on PAUSE
)

async def route(state):
    return "human_review" if state.metadata.get("uncertainty_escalate") else "continue"

graph = (
    Graph(graph_id="safe-qa")
    .add_node("llm", llm_handler, entry=True)
    .add_node("guard", handler)
    .add_conditional_edge("guard", route, {"human_review": "human_review", "continue": "end"})
    .add_node("human_review", human_node("Low confidence — please review."))
)
```

See the [Uncertainty Quantification guide](../guides/uncertainty.md) for full usage and research citations.

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
