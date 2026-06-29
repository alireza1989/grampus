# `grampus/causal/` — Causal Analysis (F4)

This package implements a two-tier causal analysis layer described in ADR-019. Tier 1 (`CausalTracer`) diagnoses root causes from the existing event log — no LLM required, sub-second latency. Tier 2 (`CausalWorldModel` + `SimpleCausalInference`) builds a persistent SCM (Structural Causal Model) where the LLM labels relationships and pure-Python code does interventional inference.

Both tiers are **opt-in hooks** in `AgentRunner`. When `causal_tracer=None, causal_world_model=None` (the defaults), no behavioral change occurs.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `CausalTracer` | `tracer.py` | Reconstructs causal graph from event log; diagnoses root causes; never raises |
| `CausalWorldModel` | `world_model.py` | Persistent SCM per agent: LLM-labeled variable relationships + interventional queries |
| `SimpleCausalInference` | `inference.py` | Pure-Python backdoor adjustment over the WorldModelGraph |
| `CausalGraph` | `types.py` | Session-scoped causal graph with typed edges (SEQUENTIAL, DATA_DEPENDENCY, FAILURE_CASCADE) |
| `CausalEdge` | `types.py` | Directed edge: source, target, edge_type, confidence |
| `CausalNode` | `types.py` | One event node in the graph: event_id, event_type, agent_id, timestamp |
| `RootCauseAnalysis` | `types.py` | Output of `diagnose()`: root_cause, path, confidence, edge_types_on_path |
| `WorldModelGraph` | `types.py` | Persistent per-agent graph: variables + SCM edges |
| `SCMVariable` | `types.py` | A named causal variable with observed values |
| `SCMEdge` | `types.py` | A causal relationship: source → target with confidence |
| `InterventionResult` | `types.py` | Result of P(Y|do(X)): is_identifiable, point_estimate, confidence_interval |

---

## Tier 1 — CausalTracer (post-hoc, no LLM)

Reconstructs a causal graph from the existing `EventLog` using three edge types:

| Edge type | Detection rule |
|---|---|
| `SEQUENTIAL` | Event B follows Event A in the same session (chronological order) |
| `DATA_DEPENDENCY` | Event B's event data contains ≥ 20 characters of substring from Event A's data |
| `FAILURE_CASCADE` | Event B is a FAILURE event that occurs within 30 seconds of Event A |

Root cause score formula (from AgentTrace + CHIEF papers):
```
score = 0.6 × structural_score + 0.4 × positional_score

structural_score = number of failure-cascade outgoing edges / max_edges
positional_score = 1 - (event_position / total_events)
(earlier events score higher — they're more likely root causes)
```

```python
from grampus.causal.tracer import CausalTracer

tracer = CausalTracer(event_store=event_log)

# Diagnose a session failure
analysis = await tracer.diagnose(
    session_id="sess-abc",
    agent_id="agent-1",
)
# analysis.root_cause: the most likely root cause event
# analysis.path: causal chain from root to failure
# analysis.confidence: 0.0–1.0

# Full session graph (for visualization or further analysis)
graph = await tracer.trace_session(session_id="sess-abc", agent_id="agent-1")
```

**`diagnose()` and `trace_session()` NEVER raise** — all exceptions are suppressed. If the event log is empty or unreachable, they return empty/default results.

---

## Tier 2 — CausalWorldModel (persistent SCM, LLM labels)

```python
from grampus.causal.world_model import CausalWorldModel

world_model = CausalWorldModel(
    state_store=dapr_store,
    model_client=client,
    model_id="claude-haiku-4-5-20251001",
    agent_id="agent-1",
)

# AgentRunner calls this during execution to label relationships
await world_model.observe(
    cause_description="web_search returned empty results",
    effect_description="LLM produced hallucinated content",
)
# → 1 LLM call: "Is X a cause of Y? Confidence?"
# → Stored in WorldModelGraph (Dapr key: causal:causal_world_model:{agent_id})

# Answer interventional queries (P(Y|do(X)))
from grampus.causal.inference import SimpleCausalInference

inference = SimpleCausalInference(world_model.get_graph())
result = await inference.intervene(
    do_variable="use_cached_search_results",
    target_variable="hallucination_rate",
)
# result.is_identifiable: True if backdoor criterion is met
# result.point_estimate: estimated P(Y|do(X)) via backdoor adjustment
```

---

## WorldModelGraph storage

The `WorldModelGraph` is persisted as a single Dapr key per agent:
- Entity type: `causal_world_model`
- Entity id: `{agent_id}`
- Format: adjacency list as JSON (variables + SCM edges)

This follows the same pattern as `MemoryGraph` from F3. Revisit if graphs exceed 200 variables per agent (the `SimpleCausalInference` complexity bound from ADR-019).

---

## Tier 1 → Tier 2 integration

`CausalTracer.absorb_diagnosis()` converts structurally validated causal chains from failure diagnosis into `WorldModelGraph` edges, bypassing LLM extraction uncertainty:

```python
# After diagnosing a failure:
analysis = await tracer.diagnose(session_id=..., agent_id=...)

# Absorb confirmed causal chains into the world model
await world_model.absorb_diagnosis(analysis)
# This adds high-confidence SCM edges based on the structural evidence,
# not LLM inference — ground truth signals for the world model.
```

---

## Hard invariants

- **The LLM's job is only to LABEL causal relationships from text.** `SimpleCausalInference` does all causal inference (backdoor adjustment) via pure Python. This circumvents the Rung Collapse limitation (arXiv 2602.11675) — LLMs cannot reliably perform causal inference natively.
- **`CausalTracer._MAX_BACKWARD_DEPTH = 8`** — the BFS search for root causes goes at most 8 edges back. This prevents O(n²) traversal on very long session graphs.
- **`SimpleCausalInference` assumes a DAG.** `is_dag()` is checked before running `intervene()` on user-provided graphs. Cyclic world models return `is_identifiable=False` rather than raising.
- **`WorldModelGraph` nodes are not deduplicated automatically** — the LLM may create duplicate variable names. Call `world_model.deduplicate()` periodically or after `absorb_diagnosis()`.
- **`CausalTracer` requires `EventLog.get_events_for_session(session_id, agent_id) -> list[dict]`.** If this method is not yet present on the event store, it must be added before using `CausalTracer`.

---

## Dependency map

```
causal/ depends on:     core/ (errors, logging, types), dapr/ (world model persistence),
                        observability/ (event_log for CausalTracer)
causal/ is imported by: orchestration/runner.py
causal/ must NOT import from: memory/ (directly), tools/, safety/, evaluation/
```

---

## ADR references

- **ADR-019** — Two-tier causal analysis: full design rationale (AgentTrace, Rung Collapse, CHIEF, IJCAI 2025)
