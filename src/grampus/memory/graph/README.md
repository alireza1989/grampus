# `grampus/memory/graph/` — Graph Memory Consolidation (F3)

This sub-package implements the graph-structured memory consolidation layer described in ADR-018, based on GAM (arXiv 2604.12285). It builds a session-scoped event-progression graph and triggers LLM-based consolidation into the stable `MemoryGraph` only when a semantic shift is detected — preventing transient noise from polluting long-term knowledge.

This is an **optional layer** in `MemoryManager`. When `graph_consolidator=None` (the default), the memory system behaves identically to pre-F3.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `GraphBuilder` | `builder.py` | Maintains session-scoped `EventGraph`; detects semantic shift via cosine distance |
| `SemanticConsolidator` | `consolidator.py` | Triggered on semantic shift: LLM extracts relationships → updates `MemoryGraph` in Dapr |
| `GraphRetriever` | `retriever.py` | BFS graph traversal for related concept retrieval |
| `EventGraph` | `types.py` | In-memory session graph (discarded after consolidation, not persisted) |
| `EventNode` | `types.py` | One event in the session graph (event_id, type, content_summary, embedding) |
| `MemoryGraph` | `types.py` | Stable long-term concept graph persisted as one Dapr key per agent |
| `ConceptNode` | `types.py` | A named concept with frequency count and linked episodes |
| `ConceptEdge` | `types.py` | A typed relationship between two concept nodes |
| `SemanticShiftEvent` | `types.py` | Signal emitted when cosine distance exceeds the shift threshold |

---

## How it works

```
Session running
    │
    │ AgentRunner.run() → graph_builder.append_event(...)
    ▼
EventGraph (in-memory, session-scoped)
    │
    │ cosine_distance(current_embedding, last_consolidated) > 0.30?
    │   No  → continue accumulating events, no LLM call
    │   Yes → emit SemanticShiftEvent
    ▼
SemanticConsolidator.consolidate(session_id, agent_id)
    │
    │ 1 LLM call: "Extract concepts and relationships from these events"
    │ → list[ConceptNode], list[ConceptEdge]
    │
    ▼
MemoryGraph updated in Dapr (one key per agent: "graph:memory_graph:{agent_id}")
    │
    │ SemanticFacts with category="schematic" promoted for top-of-recall recall
    ▼
GraphRetriever.retrieve(query, agent_id, top_k=5)
    │
    │ BFS from seed concept → collect connected concepts within max_depth hops
    ▼
list[ConceptNode] — for injection into MemoryManager.recall()
```

---

## Semantic shift threshold

`GraphBuilder` uses a cosine distance threshold of **0.30** to decide when the session's focus has shifted enough to warrant consolidation. This value is from the GAM paper.

- **Too low** (e.g., 0.10) → consolidation fires too often, wasting LLM calls on minor topic shifts
- **Too high** (e.g., 0.60) → consolidation fires too rarely, long sessions lose knowledge between shifts

The threshold is configurable at construction:
```python
builder = GraphBuilder(embedding_service=embed_svc, shift_threshold=0.30)
```

---

## MemoryGraph storage

The stable `MemoryGraph` is stored as a **single Dapr JSON key** per agent:
- Key: `graph:memory_graph:{agent_id}`
- Format: adjacency list (concept nodes + edges as JSON)
- Revisit this approach if graphs exceed 10K nodes per agent (see ADR-018)

The session `EventGraph` is **in-memory only** — it is discarded after consolidation or session end. Never try to persist it.

---

## Schematic memory

Concepts that appear in ≥ 5 episodes with high frequency are tagged `category="schematic"` in `SemanticMemory`. These are surfaced at the top of all `recall()` results regardless of the query — they represent stable, high-frequency knowledge about the agent's domain.

This is implemented as tagged `SemanticFact` records, not a new storage layer (see ADR-018).

---

## Hard invariants

- **`GraphBuilder.append_event()` never raises** — all exceptions are suppressed with `contextlib.suppress`. If embedding fails, the event is stored without an embedding but the session graph continues.
- **`SemanticConsolidator` makes exactly 1 LLM call per consolidation trigger.** With semantic-shift gating this averages 1–3 calls per 30-minute session, not per event.
- **`MemoryGraph` is a DAG-approximation** — cycles can form if LLM extraction produces contradictory edges. `GraphRetriever` uses visited-set BFS so cycles don't cause infinite loops.
- **The `EventGraph` is session-local and never persisted.** Do not attempt to load it from Dapr — it does not exist there.

---

## Dependency map

```
graph/ depends on:      core/, memory/types.py, memory/embeddings.py,
                        memory/semantic.py, dapr/ (via state_store)
graph/ is imported by:  memory/manager.py, orchestration/runner.py
graph/ must NOT import from: orchestration/, tools/, safety/
```

---

## ADR references

- **ADR-018** — Graph consolidation + lifecycle tiers: full design rationale (GAM, MemOS, FluxMem)
