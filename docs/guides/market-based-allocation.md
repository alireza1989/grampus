# Market-Based Task Allocation

## What you'll build

A `MarketCrew` where a supervisor posts tasks to a shared board and worker agents compete for them. The best-fit agent wins through a combination of capability filtering, calibration-discounted bid scoring, and UCB reputation tracking — not round-robin assignment.

---

## When to use it

| Situation | Recommended pattern |
|-----------|---------------------|
| Small crew, fixed roles, known task structure | Standard `Crew` (sequential / parallel / hierarchical) |
| Dynamic pool of workers with overlapping skills | `MarketCrew` with `use_market=True` |
| High-stakes tasks where the wrong agent is costly | Market allocation + `min_success_threshold=0.7` |
| Crew with agents that have very different costs | Market allocation with `budget_usd` set per task |

---

## Prerequisites

- Nexus installed with Anthropic support: `pip install "nexus-ai[anthropic]"`
- Dapr and Docker running locally
- `NEXUS_MODEL__ANTHROPIC_API_KEY` set

---

## Step 1 — Register worker agents

Each worker advertises what it can do via a `CapabilityProfile`.

```python
# market_crew.py
import asyncio
import os
from typing import Any

from nexus.core.models.anthropic import AnthropicClient
from nexus.core.types import AgentDefinition
from nexus.orchestration.crew import CrewMember
from nexus.orchestration.market import (
    CapabilityProfile,
    CapabilityRegistry,
    MarketAllocator,
    MarketCrew,
    ReputationTracker,
    TaskBoard,
    BidScorer,
)
from nexus.orchestration.runner import AgentRunner, RunnerConfig
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry


def make_client() -> AnthropicClient:
    return AnthropicClient(api_key=os.environ["NEXUS_MODEL__ANTHROPIC_API_KEY"])


# Worker 1: web researcher
researcher_profile = CapabilityProfile(
    agent_id="researcher",
    agent_name="Web Researcher",
    skill_tags=["web_search", "summarize", "citation"],
    cost_per_step_usd=0.002,
    max_steps=10,
)

# Worker 2: data analyst
analyst_profile = CapabilityProfile(
    agent_id="analyst",
    agent_name="Data Analyst",
    skill_tags=["sql", "data_analysis", "summarize"],
    cost_per_step_usd=0.003,
    max_steps=8,
)

# Worker 3: general-purpose writer (can do anything but not the best at specifics)
writer_profile = CapabilityProfile(
    agent_id="writer",
    agent_name="General Writer",
    skill_tags=["summarize", "drafting", "editing"],
    cost_per_step_usd=0.001,
    max_steps=5,
)
```

---

## Step 2 — Build the market infrastructure

```python
# continued from market_crew.py

async def build_market() -> MarketAllocator:
    registry = CapabilityRegistry(max_candidates=5)
    board = TaskBoard()
    reputation = ReputationTracker()
    scorer = BidScorer(reputation)   # α=0.35, β=0.45, γ=0.20 by default

    # Register all workers
    await registry.register(researcher_profile)
    await registry.register(analyst_profile)
    await registry.register(writer_profile)

    model_client = make_client()
    return MarketAllocator(
        registry=registry,
        board=board,
        scorer=scorer,
        reputation=reputation,
        model_client=model_client,
    )
```

---

## Step 3 — Build crew members

```python
# continued from market_crew.py

def make_crew_member(agent_id: str, system_prompt: str, tools: list[str]) -> CrewMember:
    registry = ToolRegistry()
    # (add tool registrations here as needed)
    executor = ToolExecutor(registry)
    runner = AgentRunner(
        model_client=make_client(),
        tool_executor=executor,
        config=RunnerConfig(max_iterations=8, enable_memory=False),
    )
    agent_def = AgentDefinition(
        name=agent_id,
        model="claude-sonnet-4-6",
        system_prompt=system_prompt,
        tools=tools,
        max_iterations=8,
    )
    return CrewMember(agent_def=agent_def, runner=runner, role="worker")


def make_members() -> list[CrewMember]:
    return [
        make_crew_member(
            "researcher",
            "You are a web researcher. Search for information and produce a detailed summary.",
            ["web_search"],
        ),
        make_crew_member(
            "analyst",
            "You are a data analyst. Query databases and produce structured analysis.",
            ["sql"],
        ),
        make_crew_member(
            "writer",
            "You are a writer. Draft, edit, and summarize documents.",
            [],
        ),
    ]
```

---

## Step 4 — Run tasks through the market

```python
# continued from market_crew.py

async def main() -> None:
    allocator = await build_market()
    members = make_members()

    crew = MarketCrew(
        members=members,
        session_id="market-session-001",
        allocator=allocator,
        use_market=True,
    )

    # Task 1: requires web_search — researcher will win
    result1 = await crew.run_task_with_market(
        task_description="Find the top 5 open-source agentic AI frameworks in 2025 and compare their GitHub stars.",
        required_skills=["web_search"],
        preferred_skills=["summarize"],
        budget_usd=0.05,
    )
    print("=== TASK 1 ===")
    print(result1.output)

    # Task 2: requires sql — analyst will win
    result2 = await crew.run_task_with_market(
        task_description="Query the sales database for total revenue by region in Q1 2025.",
        required_skills=["sql", "data_analysis"],
        budget_usd=0.03,
    )
    print("\n=== TASK 2 ===")
    print(result2.output)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## How allocation works

### 1. Capability-first filter (COALESCE)

Before any LLM calls are made, agents that lack **all** required skills are eliminated. Only capable agents receive bid solicitation prompts. This alone eliminates ~40% of unnecessary LLM calls on typical workloads.

```
Required: ["web_search"]

researcher  ["web_search", "summarize", "citation"]  ✓ capable
analyst     ["sql", "data_analysis", "summarize"]     ✗ filtered out
writer      ["summarize", "drafting", "editing"]      ✗ filtered out
```

### 2. Bid solicitation

Each capable agent receives a structured LLM prompt asking for its estimate:

```json
{
  "self_reported_success_prob": 0.85,
  "estimated_cost_usd": 0.018,
  "estimated_steps": 6,
  "rationale": "This is a standard web research task well within my capabilities."
}
```

Bid solicitation is concurrent — all capable agents are asked simultaneously.

### 3. Calibration discount (MarketBench)

LLMs systematically over-report success probability. The `ReputationTracker` corrects this:

```
calibrated_success = clamp(raw_success × calibration_factor, 0, 1)
```

A new agent starts with `calibration_factor = 1.0`. After several tasks where it claimed 0.9 probability but only succeeded 60% of the time, the factor drops to ~0.67, so a future bid of 0.9 becomes 0.60.

### 4. Composite scoring

```
composite = α × reputation + β × calibrated_success + γ × cost_score
final     = composite + ucb_bonus
```

Default weights: `α=0.35`, `β=0.45`, `γ=0.20`.

The **UCB bonus** gives new agents a chance — it decays as an agent accumulates history, so established reliable agents eventually dominate.

### 5. Award

The highest `final_score` wins, provided `calibrated_success ≥ min_success_threshold` (default 0.5). If no bid clears the threshold, allocation is **REJECTED** and `MarketAllocationError` is raised.

---

## Reputation over time

After each task completes, call `allocator.report_outcome()` (done automatically by `MarketCrew.run_task_with_market()`):

```python
from nexus.orchestration.market import TaskOutcome

outcome = TaskOutcome(
    task_id=result.task_id,
    agent_id="researcher",
    actual_success=True,
    actual_cost_usd=0.019,
    actual_steps=6,
)
await allocator.report_outcome(outcome)
```

The tracker updates:

| Metric | Update |
|--------|--------|
| `success_rate` | `successful / total` (rolling ratio) |
| `calibration_factor` | EMA(α=0.2) of `success_rate / mean_self_report` |
| `cost_accuracy` | EMA(α=0.2) of `actual_cost / estimated_cost` |
| `ucb_confidence` | `√(2 ln N / n_i)` — decays with agent task count |

---

## Tuning the scorer

Override the default weights when your priorities differ:

```python
from nexus.orchestration.market import BidScorer, ReputationTracker

scorer = BidScorer(
    ReputationTracker(),
    alpha=0.20,   # reputation
    beta=0.60,    # calibrated success ← up-weighted for quality focus
    gamma=0.20,   # cost
)
# alpha + beta + gamma must equal 1.0 (enforced at construction)
```

---

## Setting a success threshold

Tasks with high stakes should set a stricter threshold:

```python
result = await crew.run_task_with_market(
    task_description="File the quarterly compliance report...",
    required_skills=["compliance", "drafting"],
    budget_usd=0.10,
    min_success_threshold=0.75,  # passed through TaskSpec; rejects borderline bids
)
```

When the threshold is not met, `MarketAllocationError(code="MARKET_ALLOCATION_REJECTED")` is raised.

---

## Using market_node in the Graph engine

The market allocator integrates with the graph engine as a node:

```python
from nexus.orchestration.nodes import market_node, human_node
from nexus.orchestration.graph import Graph

handler = market_node(
    allocator=allocator,
    required_skills=["web_search"],
    budget_usd=0.05,
    node_name="route_to_researcher",
)

async def route(state):
    return "failed" if state.status.value == "failed" else "next"

graph = (
    Graph(graph_id="routed-pipeline")
    .add_node("allocate", handler, entry=True)
    .add_conditional_edge("allocate", route, {"failed": "human_review", "next": "execute"})
    .add_node("human_review", human_node("Market allocation failed — please assign manually."))
    .add_node("execute", llm_handler)
)
```

The node reads `state.metadata["task_description"]` and writes:
- `state.metadata["market_winner"]` — winning `agent_id` (or `None`)
- `state.metadata["market_result"]` — serialized `AllocationResult`

---

## Disabling the market

Set `use_market=False` (the default) to fall back to standard `Crew` execution with zero overhead. Useful during development or for small fixed-role crews:

```python
crew = MarketCrew(
    members=members,
    session_id="test-session",
    use_market=False,   # behaves identically to Crew
)
result = await crew.run(initial_input="do something")
```

---

## Error handling

```python
from nexus.core.errors import MarketAllocationError

try:
    result = await crew.run_task_with_market(
        task_description="...",
        required_skills=["rare_skill"],
    )
except MarketAllocationError as e:
    print(f"Allocation failed: {e}")
    print(f"Code:    {e.code}")     # MARKET_ALLOCATION_REJECTED or MARKET_WINNER_NOT_MEMBER
    print(f"Details: {e.details}")  # includes task_id and status
```

| Code | Cause |
|------|-------|
| `MARKET_ALLOCATION_REJECTED` | No capable agents found, or all bids below threshold |
| `MARKET_WINNER_NOT_MEMBER` | Winning agent from registry is not in the crew's member list |
| `MARKET_NO_MEMBERS` | `MarketCrew` was constructed with an empty member list |

---

## Research basis

| Insight | Source | Implementation |
|---------|--------|----------------|
| Capability-first filtering reduces cost 41.8% | COALESCE (arXiv 2506.01900, June 2026) | `CapabilityRegistry.filter_capable()` |
| LLMs over-report success; calibration required | MarketBench (arXiv 2604.23897, 2026) | `calibration_factor` in `BidScorer` |
| VCG matching beats round-robin (35% cost, 2.9× latency) | IEMAS (arXiv 2603.17302, March 2026) | `BidScorer` composite scoring |
| UCB exploration corrects repeated miscalibration | DRF (arXiv 2509.05764, 2025) | `ReputationTracker.ucb_bonus()` |
| Task board + capability certificates | Intelligent AI Delegation (arXiv 2602.11865, 2026) | `TaskBoard` + `CapabilityProfile` |
| Moral hazard / adverse selection failure modes | Strategic Self-Improvement (arXiv 2512.04988, 2025) | `min_success_threshold` guard |

---

## Next steps

- **[Multi-Agent Crew →](multi-agent-crew.md)** — Standard sequential/parallel/hierarchical patterns without market overhead
- **[Long-Horizon Planning →](long-horizon-planning.md)** — Structure complex multi-step tasks into a verified subgoal DAG
- **[Orchestration API →](../reference/orchestration-api.md#market-based-allocation)** — Full `MarketAllocator`, `MarketCrew`, and `market_node` reference
- **[Error Reference →](../reference/errors.md#marketallocationerror)** — `MarketAllocationError` codes and handling
