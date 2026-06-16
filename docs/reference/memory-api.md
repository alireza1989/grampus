# Memory API Reference

## MemoryManager

The unified interface to all four memory types. This is what `AgentRunner` interacts with — you rarely need to call individual memory stores directly.

::: grampus.memory.manager.MemoryManager
    options:
      show_source: false
      members: [remember, recall, forget, consolidate, add_message, get_messages]

---

## Working memory

::: grampus.memory.working.WorkingMemory
    options:
      show_source: false
      members: [add_message, get_messages, current_token_count, clear]

---

## Episodic memory

::: grampus.memory.episodic.EpisodicMemory
    options:
      show_source: false
      members: [store, get, list_by_agent, delete]

::: grampus.memory.retriever.EpisodicRetriever
    options:
      show_source: false
      members: [retrieve]

---

## Semantic memory

::: grampus.memory.semantic.SemanticMemory
    options:
      show_source: false
      members: [store_fact, get_by_subject, get_by_predicate, search, delete_fact]

::: grampus.memory.semantic_retriever.SemanticRetriever
    options:
      show_source: false
      members: [retrieve]

---

## Procedural memory

::: grampus.memory.procedural.ProceduralMemory
    options:
      show_source: false
      members: [store, get, search, delete]

---

## Consolidation

::: grampus.memory.consolidation.ConsolidationPipeline
    options:
      show_source: false
      members: [run]

---

## Memory security

::: grampus.memory.provenance.ProvenanceTracker
    options:
      show_source: false
      members: [create_provenance, verify]

::: grampus.memory.validator.MemoryValidator
    options:
      show_source: false
      members: [validate]

::: grampus.memory.trust.TrustScorer
    options:
      show_source: false
      members: [score, decay]

::: grampus.memory.auditor.MemoryAuditor
    options:
      show_source: false
      members: [audit, report]

---

## Types

::: grampus.memory.types.EpisodicRecord
    options:
      show_source: false
      members: []

::: grampus.memory.types.SemanticFact
    options:
      show_source: false
      members: []

::: grampus.memory.types.Procedure
    options:
      show_source: false
      members: []

::: grampus.memory.types.ProcedureStep
    options:
      show_source: false
      members: []

---

## SourceType enum

The `SourceType` enum determines the default trust level assigned to a memory write:

| Value | Default trust | Description |
|-------|--------------|-------------|
| `SYSTEM` | 1.0 | Internal framework writes |
| `USER_INPUT` | 0.9 | Direct user messages |
| `LLM_GENERATED` | 0.7 | Agent's own reasoning output |
| `TOOL_RESULT` | 0.6 | Results from tool executions |
| `EXTERNAL_DATA` | 0.3 | Data from external APIs, web scraping, etc. |

```python
from grampus.memory.provenance import SourceType

await manager.remember(
    "API returned rate limit error.",
    session_id="s1",
    source_type=SourceType.TOOL_RESULT,
    source_id="http_client:call_xyz",
)
```

---

## MemoryRecallResult

Returned by `MemoryManager.recall()`:

```python
@dataclass
class MemoryRecallResult:
    episodic: list[RetrievedRecord]   # scored episodic records
    semantic: list[SemanticFact]      # matching semantic facts
    query: str                        # original query string
```

`RetrievedRecord` wraps an `EpisodicRecord` with its retrieval score:

```python
@dataclass
class RetrievedRecord:
    record: EpisodicRecord
    score: float    # combined recency × similarity × importance score
```
