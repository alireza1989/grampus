# `grampus/orchestration/debate/` — Multi-Agent Debate (ADR-012)

This sub-package implements multi-agent debate as a first-class orchestration primitive. Multiple agents (optionally using different model families) argue toward a shared answer, producing substantially higher accuracy than single-agent calls on hard questions — without requiring weight updates or fine-tuning.

Grounded in: Du et al. ICML 2024, M3MAD-Bench ICLR 2025, CONSENSAGENT ACL 2025, "From Debate to Decision" April 2026, and arXiv 2504.05047 (adaptive routing).

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `DebateOrchestrator` | `orchestrator.py` | Main pipeline: routing → rounds → aggregation → act-or-escalate |
| `Debater` | `debater.py` | One debater agent: generates independent positions and responds to peers |
| `ConvergenceDetector` | `convergence.py` | Jaccard word-overlap clustering to detect when debaters agree |
| `DebateRouter` | `router.py` | Adaptive routing: bypass debate if a fast model reports high confidence |
| `DebateAggregator` | `aggregator.py` | Aggregates positions into a final answer (weighted voting, LLM judge, etc.) |
| `DebateConfig` | `types.py` | Full configuration: debaters, max_rounds, convergence threshold, aggregation strategy |
| `DebaterConfig` | `types.py` | Per-debater config: model_id, weight, system_prompt_suffix |
| `DebateResult` | `types.py` | Output: final_answer, convergence_score, rounds, escalate_to_human, token_usage |
| `DebateRound` | `types.py` | Per-round positions from all debaters |
| `RoutingDecision` | `types.py` | Router output: bypass=True with high-confidence answer, or proceed to full debate |

---

## How the debate pipeline works

```
DebateOrchestrator.run(question)
    │
    ▼
DebateRouter.check(question)
    │  Confidence ≥ adaptive_routing_threshold?
    │  Yes → return RoutingDecision(bypass=True, answer=...) ← ~40% of easy queries skip
    │  No  → proceed to full debate
    │
    ▼
Round 1: asyncio.gather(debater.respond_independently(question) for each debater)
    │  Each debater: independent answer, no peer influence
    │
    ▼
ConvergenceDetector.check(positions)
    │  Jaccard overlap ≥ convergence_threshold? → stop early
    │
    ▼
Rounds 2..max_rounds: asyncio.gather(debater.respond_to_peers(peers, question) for each)
    │  Sycophancy resistance: debater must restate own prior answer VERBATIM before evaluation
    │  Any position change requires citing specific logical evidence (CONSENSAGENT)
    │  Check convergence after each round
    │
    ▼
DebateAggregator.aggregate(all_rounds, weights)
    │
    ▼
Act-vs-escalate:
    final_convergence < escalate_threshold?
    │  Yes → DebateResult(escalate_to_human=True)  — caller uses human_node
    │  No  → DebateResult(escalate_to_human=False, final_answer=...)
```

---

## Usage

### Standalone

```python
from grampus.orchestration.debate.orchestrator import DebateOrchestrator
from grampus.orchestration.debate.types import DebateConfig, DebaterConfig

config = DebateConfig(
    debaters=[
        DebaterConfig(model_id="claude-haiku-4-5-20251001", weight=1.0),
        DebaterConfig(model_id="gpt-4o-mini", weight=0.8),
        DebaterConfig(model_id="claude-sonnet-4-6", weight=1.2),
    ],
    max_rounds=3,
    convergence_threshold=0.75,
    escalate_threshold=0.40,
    adaptive_routing=True,
    adaptive_routing_threshold=0.90,
    aggregation="weighted_vote",
)

orchestrator = DebateOrchestrator(config=config, cost_tracker=ct, tracer=tracer)
result = await orchestrator.run("Is this legal interpretation correct? ...")

if result.escalate_to_human:
    # Route to human_node in graph
    ...
else:
    print(result.final_answer)
    print(f"Convergence score: {result.convergence_score:.2f}")
```

### As a Graph node

```python
from grampus.orchestration.graph import Graph
from grampus.orchestration.debate.types import debate_node

graph = (
    Graph("legal-review")
    .add_node("intake", intake_handler, entry=True)
    .add_node("debate", debate_node(orchestrator))  # debate_node wraps DebateOrchestrator
    .add_node("finalize", finalize_handler)
    .add_conditional_edge(
        "debate",
        lambda state: "human" if state.metadata["escalate"] else "finalize",
    )
    .add_node("human", human_node)
)
```

---

## Key design decisions baked in

**Heterogeneous panels** (`model_id` per debater, not just temperature): different model families have orthogonal error modes — a question that confuses Claude may be trivial for GPT-4, and vice versa.

**Sycophancy resistance** (CONSENSAGENT, ACL 2025): Round 2+ prompts enforce verbatim restatement of the prior position. Debaters cannot silently shift positions to match the majority — they must cite specific logical evidence for any change.

**Adaptive routing** (~40% cost reduction): a fast routing model rates its own confidence before running the full debate. If confidence ≥ threshold, the routing answer is returned directly. This eliminates unnecessary debate panels on easy questions.

**Act-vs-escalate**: when the panel cannot converge above `escalate_threshold`, the result explicitly flags `escalate_to_human=True` rather than returning a low-confidence answer. This integrates with the existing `human_node` graph primitive.

---

## Hard invariants

- **All debaters within a round execute concurrently** via `asyncio.gather`. Debate latency is bounded by the slowest debater in the round, not the sum — it is no worse than a single LLM call per round.
- **`ConvergenceDetector` uses Jaccard word-overlap** — no embedding calls, no ML model. It is fast and deterministic. Do not replace it with a semantic similarity measure that would add latency.
- **`escalate_to_human=True` does not raise an exception** — it is a field in `DebateResult`. The caller must check this field and decide what to do. Use `human_node` in a Graph for automated escalation.
- **Zero new dependencies** — stdlib `json`, `asyncio`, `re`, `time` plus existing Pydantic and OTEL. Do not add dependencies to this sub-package without strong justification.

---

## Dependency map

```
debate/ depends on:     core/ (types, errors, logging), orchestration/runner.py (for Debater)
debate/ is imported by: orchestration/graph.py (via debate_node())
debate/ must NOT import from: memory/, tools/ (directly), safety/, evaluation/
```

---

## ADR references

- **ADR-012** — Multi-agent debate as a first-class orchestration primitive (full rationale)
