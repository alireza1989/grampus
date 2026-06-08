# Hierarchical Memory Architecture (F3)

## The Problem with Flat Memory at Scale

As agents accumulate experience, flat vector search over thousands of episodic records degrades in two distinct ways. First, **retrieval quality degrades**: when every interaction is stored as a separate vector, semantically distinct records with overlapping vocabulary crowd out the most relevant results. A record from six months ago about Python error handling can outrank a recent one about the same topic simply because it shares more tokens with the query. Flat cosine search has no way to distinguish "Python async" from "Python file I/O" — they're just neighboring vectors. GAM (arXiv 2604.12285) demonstrated that building a structured knowledge graph from interactions substantially improves reasoning accuracy on long-horizon tasks precisely because graph traversal respects conceptual relationships.

Second, **retrieval overhead scales with collection size**: treating every record equally means every query must score every record, even those accessed only once in 2023. MemOS (arXiv 2505.22101) measured a **35.24% token savings** in production by managing memory as a hot/warm/cold resource — frequently-accessed records stay in fast-retrieval tiers, while rarely-accessed ones are demoted to cold archive storage. Without lifecycle management, agents spend increasing amounts of context on stale, rarely-useful memories.

## The Knowledge Graph Layer

F3 introduces a two-level graph architecture on top of the existing four memory layers.

### Session EventGraph (transient)

During each agent session, a `GraphBuilder` maintains an in-memory `EventGraph` — a sequence of `EventNode` objects capturing what the agent did: LLM calls, tool calls, memory reads. The EventGraph is **never persisted to Dapr** — it lives only for the duration of the session.

The key insight from GAM is that consolidation should be **triggered by semantic shift, not by time**. After every event, the builder computes the cosine distance between the new event's embedding and the last consolidated state. If the distance exceeds a threshold (default 0.30), a `SemanticShiftEvent` is emitted.

**Example**: if an agent discusses Python async for 10 messages then suddenly switches to discussing database schema design, the cosine distance between "Python async" embeddings and "database schema" embeddings will be large. A semantic shift is detected — the Python async discussion is consolidated into the topic graph **before** the new topic begins, preventing cross-topic noise from contaminating stable knowledge.

### Topic-Associative-Network / MemoryGraph (persistent)

The `SemanticConsolidator` merges EventGraph events into a persistent `MemoryGraph` stored in Dapr. The MemoryGraph is an adjacency list of `ConceptNode` objects and `RelationshipEdge` objects:

```
MemoryGraph
├── nodes: {node_id: ConceptNode}   # "Python async", "PostgreSQL", "rate limiting"
└── edges: [RelationshipEdge]       # "Python async" precedes "database schema"
```

Merging uses cosine similarity: if a new extracted concept's embedding is ≥ 0.85 similar to an existing node, the existing node's frequency is incremented instead of creating a duplicate. Nodes with `frequency >= 5` are tagged as `category="schematic"` — core concepts that are always surfaced first in recall results.

**Graph version increments** on every consolidation, enabling audit trails.

## Hot/Warm/Cold Lifecycle Tiers

`LifecycleTierManager` tracks access patterns for every memory record and manages three tiers:

| Tier | Description | TTL | Promotion trigger |
|------|-------------|-----|-------------------|
| **HOT** | Active session working set | 1 hour | ≥3 accesses in 7 days (from WARM) |
| **WARM** | Redis cache: fast retrieval | 7 days | ≥1 access in 7 days (from COLD) |
| **COLD** | Dapr/Postgres permanent archive | permanent | default |

Promotion is **lazy** (runs on access). Demotion is **sweeper-based** (call `sweep()` at session start to clean up stale HOT records from the previous session).

```
COLD → WARM: 1 access in 7-day window
WARM → HOT: 3 accesses in 7-day window
HOT → WARM: last_accessed > 1 hour ago (sweep)
WARM → COLD: 0 accesses in 7 days AND last_accessed > 7 days ago (sweep)
```

## Adaptive Retrieval Routing

`AdaptiveRetriever` routes each query to the optimal retrieval path (FluxMem, arXiv 2602.14038) using lightweight keyword heuristics — no ML model, no embedding call just for routing:

| Query type | Example | Retrieval path |
|------------|---------|----------------|
| **SEQUENTIAL** | "what did we discuss last time?" | Most recent N episodic records |
| **GRAPH** | "how does authentication relate to authorization?" | GraphRetriever (BFS traversal) |
| **FLAT** | "what is the API endpoint?" | EpisodicRetriever + SemanticRetriever |

Keywords that trigger SEQUENTIAL: `last time`, `previously`, `earlier`, `before`, `recent`

Keywords that trigger GRAPH: `how`, `why`, `relationship`, `explain`, `cause`, `effect` (or queries longer than 80 characters)

When GRAPH retrieval returns an empty result (no graph built yet), the router falls back to FLAT automatically.

## Quick Start

```python
from nexus.memory.graph import GraphBuilder, SemanticConsolidator, GraphRetriever
from nexus.memory.lifecycle import LifecycleTierManager, AdaptiveRetriever

graph_builder = GraphBuilder(embedding_service=embedding_svc)
consolidator = SemanticConsolidator(
    state_store=dapr_store,
    embedding_service=embedding_svc,
    model_client=model_client,
)
graph_retriever = GraphRetriever(consolidator=consolidator, embedding_service=embedding_svc)
lifecycle_manager = LifecycleTierManager(state_store=dapr_store, agent_id="my-agent")
adaptive_router = AdaptiveRetriever(
    episodic_retriever=ep_retriever,
    semantic_retriever=sem_retriever,
    graph_retriever=graph_retriever,
    episodic_memory=ep_memory,
    tier_manager=lifecycle_manager,
)

memory_manager = MemoryManager(
    # ... existing params ...
    graph_consolidator=consolidator,
    lifecycle_manager=lifecycle_manager,
    adaptive_router=adaptive_router,
)
runner = AgentRunner(
    # ... existing params ...
    graph_builder=graph_builder,
)
```

When any of the three F3 params (`graph_consolidator`, `lifecycle_manager`, `adaptive_router`) are `None` (the default), `MemoryManager` and `AgentRunner` behave identically to pre-F3.

## Inspecting the Knowledge Graph

```python
graph = await consolidator.load_graph("my-agent")
print(f"Nodes: {len(graph.nodes)}, version: {graph.version}")
for node in graph.nodes.values():
    category = node.metadata.get("category", "regular")
    print(f"  {node.label} [{category}]: frequency={node.frequency}")

# Use GraphRetriever directly
result = await graph_retriever.query("my-agent", "how does X relate to Y?", top_k=5)
context_str = graph_retriever.format_as_context(result)
print(context_str)
# Knowledge graph context:
# - Python async: Asynchronous programming in Python [core concept]
# - asyncio: Python's async event loop library
```

## Implementation Notes

- `MemoryGraph` is stored as a single Dapr key per agent (adjacency list in JSON). Revisit if graphs exceed 10K nodes per agent.
- `SemanticConsolidator` makes 1 LLM call per consolidation trigger. With semantic-shift gating, this averages 1–3 calls per 30-minute session.
- Call `LifecycleTierManager.sweep()` at session start to demote stale HOT records from the previous session.
- Zero new required dependencies — stdlib `collections`, `math`, `json`, `uuid`, `datetime` plus existing Pydantic, Dapr, and model client.
