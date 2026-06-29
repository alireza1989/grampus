# `grampus/memory/lifecycle/` — Memory Lifecycle Tiers (F3)

This sub-package implements hot/warm/cold lifecycle management for memory records, based on MemOS (arXiv 2505.22101). It tracks access patterns and promotes/demotes records between tiers to ensure recently-used knowledge is retrieved fastest. Combined with `AdaptiveRetriever`, it routes queries to the optimal retrieval strategy per query type.

This is an **optional layer** in `MemoryManager`. When `lifecycle_manager=None, adaptive_router=None` (the defaults), the memory system behaves identically to pre-F3.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `LifecycleTierManager` | `tier_manager.py` | Tracks access counts, promotes/demotes records between HOT/WARM/COLD |
| `AdaptiveRetriever` | `adaptive_router.py` | Routes queries to sequential/graph/flat retrieval based on query type heuristics |
| `TierRecord` | `types.py` | Tracks access metadata per record: total access count, 7-day count, tier, timestamps |
| `MemoryTier` | `types.py` | Enum: `HOT, WARM, COLD` |
| `MemoryType` | `types.py` | Enum: `EPISODIC, SEMANTIC, PROCEDURAL, WORKING` |
| `LifecycleStats` | `types.py` | Summary of tier distribution across all records |

---

## Tier definitions (MemOS mapping to existing infrastructure)

| Tier | Infrastructure | Criterion |
|---|---|---|
| `HOT` | In-context (working memory) | ≥ 3 accesses in last 7 days; TTL 1 hour per session |
| `WARM` | Redis cache (Dapr cache store) | ≥ 1 access in last 7 days; TTL 7 days |
| `COLD` | PostgreSQL/Dapr state | All other records |

No new infrastructure is needed — these tiers map to the existing Dapr state and Redis cache components.

---

## How lifecycle management works

```
Any MemoryManager.recall() call
    │
    ├─ LifecycleTierManager.record_access(record_id, memory_type)
    │   → Update TierRecord in Dapr
    │   → Check promotion thresholds (lazy promotion on access)
    │       ≥ 3 accesses in 7 days → promote to HOT
    │       ≥ 1 access in 7 days → promote to WARM
    │
Session start (AgentRunner.run() pre-loop)
    │
    ├─ LifecycleTierManager.sweep(agent_id)  [via contextlib.suppress]
    │   → Scan all TierRecords for this agent
    │   → Demote stale HOT records (TTL expired) to WARM
    │   → Demote stale WARM records (no access in 7 days) to COLD
```

---

## Adaptive routing (FluxMem-inspired)

`AdaptiveRetriever` classifies each query and routes it to the best retrieval strategy using simple keyword heuristics — no ML model or embedding call required:

| Query type | Detection keywords | Retrieval strategy |
|---|---|---|
| `SEQUENTIAL` | "after", "then", "step", "first", "next", "sequence" | Time-ordered episode scan |
| `GRAPH` | Query length > 50 chars, or "because", "why", "caused", "led to" | GraphRetriever BFS |
| `FLAT` | Everything else | Standard hybrid retrieval (cosine + BM25) |

```python
from grampus.memory.lifecycle.adaptive_router import AdaptiveRetriever

router = AdaptiveRetriever(
    episodic_retriever=er,
    graph_retriever=gr,   # optional; None if graph/ not wired up
)
records = await router.retrieve(query="Why did the agent fail at step 3?", agent_id="a1")
# → routes to GRAPH strategy (causal keyword "why")
```

---

## Hard invariants

- **`LifecycleTierManager.record_access()` and `sweep()` never raise** — all operations are wrapped in `contextlib.suppress(Exception)`. A broken lifecycle manager never breaks retrieval.
- **Promotion is lazy (on access), demotion is periodic (on sweep).** HOT records are never demoted mid-session — only at session start via `sweep()`. This prevents records from being demoted while an agent is actively using them.
- **`TierRecord` is persisted as a separate Dapr key** per record: `lifecycle_tier:{record_id}`. It does not modify the original `EpisodicRecord` or `SemanticFact`.
- **`AdaptiveRetriever` falls back to flat retrieval on any error.** If `GraphRetriever` raises, the router returns results from the standard episodic retriever without surfacing the error to the caller.
- **`MemoryManager` passes `lifecycle_manager` results to `AdaptiveRetriever`.** The two components are designed to be used together — but each is independently injectable.

---

## Dependency map

```
lifecycle/ depends on:     core/, memory/types.py, memory/retriever.py,
                           memory/graph/retriever.py (optional),
                           dapr/ (via state_store)
lifecycle/ is imported by: memory/manager.py
lifecycle/ must NOT import from: orchestration/, tools/, safety/
```

---

## ADR references

- **ADR-018** — Graph consolidation + lifecycle tiers: full design rationale (GAM, MemOS, FluxMem)
