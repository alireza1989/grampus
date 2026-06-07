# Memory Guide

## What you'll learn

- Why Nexus has four distinct memory types and when to use each
- How to configure working memory, episodic retrieval, and semantic consolidation
- How to use the `MemoryManager` unified interface
- How memory security (provenance, trust, injection defense) works

---

## Why four memory types?

Each type serves a different timescale and purpose:

| Memory | Timescale | Purpose | Backend |
|--------|-----------|---------|---------|
| **Working** | Current session | Token window for LLM context | In-process |
| **Episodic** | Cross-session | Events: what happened, when, to whom | PostgreSQL + pgvector |
| **Semantic** | Persistent | Facts: subject–predicate–object triples | PostgreSQL + pgvector |
| **Procedural** | Persistent | Workflows: how to do recurring tasks | PostgreSQL |

---

## Working memory

Working memory holds the active conversation window. When it approaches the token limit, it auto-summarizes older turns.

```python
from nexus.core.models.anthropic import AnthropicClient
from nexus.memory.summarizer import Summarizer
from nexus.memory.working import WorkingMemory

client = AnthropicClient(api_key="...")
summarizer = Summarizer(model_client=client, strategy="hybrid")

working = WorkingMemory(
    summarizer=summarizer,
    token_limit=100_000,      # summarize when 80% full (80k tokens)
)
```

### Summarization strategies

| Strategy | Behavior |
|----------|----------|
| `truncate` | Drop oldest messages first (fast, no LLM call) |
| `summarize` | Use LLM to compress old messages into a summary |
| `hybrid` | Summarize old messages, keep the N most recent at full fidelity |

```python
# hybrid keeps the last 20 messages verbatim, summarizes the rest
summarizer = Summarizer(model_client=client, strategy="hybrid", recent_keep=20)
```

### Add and retrieve messages

```python
from nexus.core.types import Message, Role

await working.add_message(Message(role=Role.USER, content="Hello!"))
await working.add_message(Message(role=Role.ASSISTANT, content="Hi there!"))

messages = await working.get_messages()
print(f"Window size: {len(messages)} messages")
print(f"Token count: {working.current_token_count}")
```

---

## Episodic memory

Episodic memory persists events across sessions. Each record has a timestamp, embedding, trust score, and importance score.

```python
from nexus.memory.embeddings import EmbeddingService
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.types import EpisodicRecord

embedding_service = EmbeddingService(model_client=client)
episodic = EpisodicMemory(state_store=state_store, embedding_service=embedding_service)

# Store an event
record = await episodic.store(
    content="User asked about pricing for the enterprise plan.",
    agent_id="support-agent",
    session_id="session-42",
    metadata={"user_id": "user-123", "intent": "pricing"},
)
print(f"Stored record: {record.id}")
print(f"Trust score:   {record.trust_score}")
print(f"Importance:    {record.importance_score}")
```

### Retrieval with hybrid scoring

Episodic retrieval blends three signals:

```
score = α × recency + β × similarity + γ × importance
```

```python
from nexus.memory.retriever import EpisodicRetriever

retriever = EpisodicRetriever(
    episodic_memory=episodic,
    recency_weight=0.3,      # α — prefer recent events
    similarity_weight=0.5,   # β — prefer semantically relevant events
    importance_weight=0.2,   # γ — prefer high-importance events
)

results = await retriever.retrieve(
    query="pricing questions",
    agent_id="support-agent",
    top_k=5,
)
for r in results:
    print(f"  [{r.score:.2f}] {r.record.content[:80]}")
```

!!! tip "Tuning retrieval weights"
    For support agents, increase `recency_weight` — recent conversations are most relevant.
    For knowledge agents, increase `similarity_weight` — factual relevance matters more than recency.

---

## Semantic memory

Semantic memory stores Subject–Predicate–Object facts extracted from episodic records.

```python
from nexus.memory.semantic import SemanticMemory
from nexus.memory.types import SemanticFact

semantic = SemanticMemory(state_store=state_store, embedding_service=embedding_service)

# Store a fact
fact = await semantic.store_fact(
    subject="user-123",
    predicate="prefers",
    object="dark mode",
    confidence=0.9,
    source_episode_ids=["ep-001"],
)

# Query by subject
facts = await semantic.get_by_subject("user-123")
for f in facts:
    print(f"  {f.subject} {f.predicate} {f.object}  (confidence={f.confidence:.2f})")
```

### Conflict resolution

When a new fact conflicts with an existing one (same subject + predicate, different object), Nexus uses confidence-weighted replacement:

```
new_fact stored  if  new_confidence > existing_confidence * 0.9
```

This prevents noisy tool results from immediately overwriting established facts.

---

## Procedural memory

Procedural memory stores reusable workflow templates.

```python
from nexus.memory.procedural import ProceduralMemory
from nexus.memory.types import Procedure, ProcedureStep

procedural = ProceduralMemory(state_store=state_store)

# Store a learned procedure
procedure = Procedure(
    name="file_support_ticket",
    description="Steps to file a support ticket in the ticketing system",
    steps=[
        ProcedureStep(
            action="search_tickets",
            tool_name="search_existing_tickets",
            parameters_template={"query": "{issue_description}"},
            expected_outcome="list of similar existing tickets",
        ),
        ProcedureStep(
            action="create_ticket",
            tool_name="create_ticket",
            parameters_template={"title": "{issue_title}", "body": "{issue_description}"},
            expected_outcome="ticket ID",
        ),
    ],
    trigger_conditions=["user wants to file a ticket", "create issue"],
    agent_id="support-agent",
)
await procedural.store(procedure)

# Find relevant procedures for a task
matches = await procedural.search(
    task_description="I need to report a billing problem",
    agent_id="support-agent",
    top_k=3,
)
```

---

## The MemoryManager unified interface

In practice, you use `MemoryManager` rather than individual memory stores. It handles routing, provenance, and security automatically.

```python
from nexus.memory.manager import MemoryManager

manager = MemoryManager(
    working_memory=working,
    episodic_memory=episodic,
    semantic_memory=semantic,
    procedural_memory=procedural,
    episodic_retriever=retriever,
    semantic_retriever=semantic_retriever,
    consolidation_pipeline=consolidation,
    agent_id="my-agent",
)

# Store something — automatically adds provenance
await manager.remember(
    "User prefers responses in bullet points.",
    session_id="session-1",
    source_type=SourceType.USER_INPUT,
    source_id="user-123",
)

# Recall relevant memories for a query
recalled = await manager.recall("user formatting preferences", top_k=5)
for ep in recalled.episodic:
    print(f"Episodic: {ep.record.content}")
for fact in recalled.semantic:
    print(f"Fact: {fact.subject} {fact.predicate} {fact.object}")

# Add to working memory
from nexus.core.types import Message, Role
await manager.add_message(Message(role=Role.USER, content="Hello"))
messages = await manager.get_messages()

# Delete a record
await manager.forget(record_id="ep-001", memory_type="episodic")

# Run consolidation (extract semantic facts from episodic records)
consolidation_result = await manager.consolidate()
print(f"Extracted {consolidation_result.facts_created} new facts")
```

---

## Consolidation pipeline

The consolidation pipeline runs asynchronously in the background, extracting semantic facts from recent episodic records:

```python
from nexus.memory.consolidation import ConsolidationPipeline

pipeline = ConsolidationPipeline(
    episodic_memory=episodic,
    semantic_memory=semantic,
    model_client=client,
    lookback_hours=24,        # process records from last 24 hours
    batch_size=50,            # process 50 records per run
)

result = await pipeline.run()
print(f"Processed:      {result.episodes_processed}")
print(f"Facts created:  {result.facts_created}")
print(f"Facts merged:   {result.facts_merged}")
print(f"Facts skipped:  {result.facts_skipped}")
```

---

## Memory security

Every memory write is validated and stamped with provenance. The `MemoryValidator` blocks suspicious writes before they reach the store:

```python
from nexus.memory.provenance import ProvenanceTracker
from nexus.memory.validator import MemoryValidator

validator = MemoryValidator(
    max_content_size_bytes=10_000,
    rate_limit_per_source=100,       # max 100 writes per source per minute
    detect_injection=True,           # block "remember that always..." patterns
)
tracker = ProvenanceTracker()

manager = MemoryManager(
    ...,
    provenance_tracker=tracker,
    memory_validator=validator,
)
```

When `detect_injection=True`, the validator blocks writes containing patterns like:

- `"Remember that in all future conversations..."`
- `"Always respond with..."`
- `"Ignore previous instructions and..."`

!!! warning "External data trust"
    Content retrieved from external APIs (web search results, webhooks, RSS feeds) should be stored with `SourceType.EXTERNAL_DATA` (trust=0.3). The memory retriever uses trust scores to deprioritize low-trust memories and the auditor flags anomalies.

---

## Inspecting memory via the web UI

You can browse all memory entries visually at `/ui/memory/` in the Nexus web interface. The memory inspector provides a filter bar to narrow by agent ID, memory type, search text, and minimum trust score. Each row in the table shows the record's type, content preview, trust score (color-coded: green ≥0.8, yellow 0.5–0.8, red <0.5), provenance source, and creation timestamp. Click any row to open the detail panel with the full content and complete provenance metadata. Start the server with `nexus serve` and open `http://localhost:8000/ui/memory/` to access it.

To delete individual entries from the UI, click the trash icon in the row's Actions column. You can also delete programmatically using `MemoryManager.forget(record_id)` or the REST API:

```python
# Programmatic deletion
await manager.forget(record_id="ep-001", memory_type="episodic")
```

```bash
# REST API deletion
curl -X DELETE "http://localhost:8000/memory/ep-001"
```

See the [Web UI guide](web-ui.md) for the full inspector reference.

---

## Next steps

- **[Memory API reference →](../reference/memory-api.md)** — Full `MemoryManager` and type reference
- **[Security model →](../architecture/security.md)** — MINJA/MemoryGraft threat model and defenses
- **[Observability guide →](observability.md)** — Trace memory reads and writes with OTEL
- **[Web UI →](web-ui.md)** — Browse and manage memory entries visually
