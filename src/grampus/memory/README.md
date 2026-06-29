# `grampus/memory/` — Memory System

This package implements the full multi-layer memory system for Grampus agents. It provides four core memory types (working, episodic, semantic, procedural), a security layer (provenance + trust + validation), and optional advanced layers (graph consolidation, lifecycle tiers, user modeling, self-improvement via reflexion).

The public interface for the rest of the framework is `MemoryManager` — callers should never reach into individual stores directly.

---

## Memory layer overview

| Layer | Scope | Persistence | Purpose |
|---|---|---|---|
| **Working** | In-session | Dapr (window + audit log) | Message buffer sent to LLM; auto-summarizes at token limit |
| **Episodic** | Cross-session | Dapr state + embeddings | Raw experiences with importance scores and semantic search |
| **Semantic** | Cross-session | Dapr state | Subject-predicate-object facts, extracted from episodes |
| **Procedural** | Cross-session | Dapr state | Reusable workflows, skills, and reflections |

**Advanced optional layers:**
- [reflexion/](reflexion/) — Post-failure verbal reflection + post-success skill extraction (ADR-016)
- [user/](user/) — Three-tier user modeling: episodes → facts → synthesized profile (ADR-017)
- [graph/](graph/) — Session event-progression graph + semantic-shift-triggered consolidation (ADR-018)
- [lifecycle/](lifecycle/) — Hot/warm/cold tier management for memory records (ADR-018)

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `MemoryManager` | `manager.py` | Unified interface: `remember()`, `recall()`, `forget()`, `consolidate()` |
| `WorkingMemory` | `working.py` | In-session message buffer with auto-summarization |
| `EpisodicMemory` | `episodic.py` | Cross-session CRUD + embedding-backed store |
| `EpisodicRetriever` | `retriever.py` | Hybrid retrieval: `α·recency + β·similarity + γ·importance` |
| `SemanticMemory` | `semantic.py` | Fact store with `(subject, predicate)` deduplication |
| `SemanticRetriever` | `semantic_retriever.py` | Fact retrieval by subject, predicate, or similarity |
| `ProceduralMemory` | `procedural.py` | Workflow, skill, and reflection store |
| `ProvenanceTracker` | `provenance.py` | SHA-256 content hash + source metadata on every write |
| `TrustScorer` | `trust.py` | Source-type trust levels with temporal decay |
| `MemoryValidator` | `validator.py` | Injection detection, size limits, rate limits |
| `MemoryAuditor` | `auditor.py` | Periodic integrity scan via content hash verification |
| `ConsolidationPipeline` | `consolidation.py` | Async background fact extraction from episodes |
| `EmbeddingService` | `embeddings.py` | Multi-provider embedding wrapper with Redis cache |
| `TokenCounter` | `token_counter.py` | tiktoken-based token counting per model family |
| `Summarizer` | `summarizer.py` | LLM-based conversation compression (truncate/summarize/hybrid) |

---

## How to use this package

### The standard path — `MemoryManager`

```python
from grampus.memory.manager import MemoryManager

# MemoryManager is constructed by AgentRunner from GrampusConfig.
# In tests, build it directly:
manager = MemoryManager(
    working_memory=wm,
    episodic_memory=em,
    episodic_retriever=er,
    semantic_memory=sm,
    semantic_retriever=sr,
    procedural_memory=pm,
    # Optional security:
    provenance_tracker=pt,
    validator=val,
    # Optional advanced layers:
    graph_consolidator=gc,       # F3
    lifecycle_manager=lm,        # F3
    adaptive_router=ar,          # F3
    plugin_manager=plugin_mgr,   # H49
)

# Write — goes through provenance + validation + store
await manager.remember(
    content="User prefers concise summaries.",
    source_type=SourceType.USER_INPUT,
    source_id="session-abc",
    memory_type="episodic",
)

# Read — hybrid retrieval, routed through AdaptiveRetriever if set
records = await manager.recall(
    query="What does the user prefer?",
    memory_types=["episodic", "semantic"],
    top_k=5,
)

# Delete — removes from store and index
await manager.forget(record_id="ep-123")

# Consolidate — extracts facts from recent episodes (async, non-blocking)
await manager.consolidate()
```

### Working memory directly (AgentRunner usage)

```python
from grampus.memory.working import WorkingMemory

wm = WorkingMemory(
    state_store=dapr_store,
    token_counter=token_counter,
    summarizer=summarizer,
    agent_id="agent-1",
    session_id="sess-1",
    max_tokens=100_000,
    threshold_fraction=0.8,  # summarize at 80% of limit
)

await wm.add(message)
window = await wm.get_window()      # compressed window sent to LLM
history = await wm.get_history()    # full uncompressed audit log
```

---

## Write path: security pipeline

Every `MemoryManager.remember()` call passes through this pipeline in order:

```
Input
  │
  ▼
ProvenanceTracker.create(content, source_type, source_id)
  → Provenance(content_hash_sha256, trust_level, source_type, ...)
  │
  ▼
MemoryValidator.validate(content, source_id)
  → Check 1: Injection patterns (13 compiled regex + heuristic)
  → Check 2: Size limit (default 10,000 bytes)
  → Check 3: Rate limit (default 60 writes/minute per source_id)
  → ValidationResult(allowed, reasons)
  │  ← raises MemorySecurityError if not allowed
  ▼
PluginManager.pre_memory_write(...)  ← optional H49 hook
  │  ← HookBlockedError → MemorySecurityError(code="PLUGIN_BLOCKED")
  ▼
EpisodicMemory.store(content, provenance=...)
                 or SemanticMemory.store(fact)
                 or ProceduralMemory.store(procedure)
```

---

## Retrieval score formula

`EpisodicRetriever` ranks all records using:

```
score = α·recency + β·similarity + γ·importance
recency = exp(-λ · age_in_days)

Default weights: α=0.4, β=0.4, γ=0.2, λ=0.01
Constraint: α + β + γ must equal 1.0 (validated at construction)
```

Records without embeddings receive `similarity_score=0.0` but are still ranked via recency and importance.

---

## SourceType trust levels

These are defined in `provenance.py` and determine the initial trust score:

| SourceType | Trust level |
|---|---|
| `SYSTEM` | 1.0 |
| `USER_INPUT` | 0.9 |
| `LLM_GENERATED` | 0.7 |
| `TOOL_RESULT` | 0.6 |
| `EXTERNAL_DATA` | 0.3 |

---

## Hard invariants

- **All writes go through `MemoryManager.remember()`** — never call individual store `.store()` methods from outside the memory package. The security pipeline is only enforced in `MemoryManager.remember()`.
- **`ProvenanceTracker.create()` computes a SHA-256 hash of the raw content string.** This hash is verified on read by `MemoryAuditor`. Never store memory without provenance.
- **`EpisodicRetriever` weights must sum to 1.0** — enforced with `ValueError` at construction. If you change weights, verify the constraint.
- **`WorkingMemory` maintains two Dapr keys per session**: `working:window:{session_id}` (compressed) and `working:history:{session_id}` (full audit log, never compressed). The history key is immutable once written.
- **`SemanticMemory.store()` deduplicates on `(subject, predicate)`.** If a fact with the same subject-predicate already exists, the higher-confidence version's `object_value` wins and `source_episode_ids` are merged. It does not raise on duplicate — it silently merges.
- **`ConsolidationPipeline` uses an LLM call.** Never call it in the hot path (inside `AgentRunner.run()`). Schedule it as a background task after session end.
- **`MemoryAuditor.verify(record)` compares `record.content` against `record.provenance.content_hash_sha256`.** A mismatch logs a warning and returns `False` — it does not auto-delete. Callers decide whether to delete tampered records.

---

## Extension guide

### Adding a new memory type

1. Define a Pydantic model in `types.py` (subclass `BaseModel`, include provenance fields).
2. Create `mytype.py` with a CRUD class backed by `DaprStateStore`.
3. Add a retriever class for it.
4. Register in `MemoryManager.__init__` as an optional parameter (default `None`).
5. Handle it in `remember()`, `recall()`, and `forget()` with `if self._mytype_memory:` guards.

### Swapping the embedding provider

The `EmbeddingService` in `embeddings.py` wraps an `EmbeddingProvider` ABC (ADR-023). To add a new provider:
1. Subclass `EmbeddingProvider` in `embedding_providers.py`.
2. Implement `async embed(text, input_type=None) -> list[float]` and `embed_batch(...)`.
3. Expose `.dimensions` property — used by pgvector setup to validate column width at startup.
4. Add the optional SDK as a new extras group in `pyproject.toml`.

---

## Dependency map

```
memory/ depends on:     core/, dapr/
memory/ is imported by: orchestration/ (AgentRunner, Graph), causal/,
                        evaluation/, cli/
memory/ must NOT import from: tools/, safety/ (circular), orchestration/
```

---

## ADR references

- **ADR-005** — Event sourcing; working memory's full audit log is part of this
- **ADR-006** — Memory write provenance as non-negotiable
- **ADR-016** — Reflexion + SkillLibrary (→ `reflexion/`)
- **ADR-017** — Three-tier user memory hierarchy (→ `user/`)
- **ADR-018** — Graph consolidation + lifecycle tiers (→ `graph/`, `lifecycle/`)
- **ADR-023** — Multi-provider embedding service with per-memory-type routing
