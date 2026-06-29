# `grampus/memory/user/` — Persistent User Modeling (F2)

This sub-package implements the three-tier user memory hierarchy described in ADR-017. Agents remember individual users across sessions — their expertise, preferences, goals, and constraints — without requiring explicit user profile management.

This is an **opt-in hook** in `AgentRunner`. When `user_memory_adapter=None, user_id=None` (the defaults), the runner is behaviorally identical to the pre-F2 runner with zero overhead.

---

## Three-tier architecture

```
Tier 3 — UserEpisodes (raw interactions)
    Stored in EpisodicMemory — no new infrastructure.
    Each message exchange is a raw episode tagged with user_id.
    Tier 3 provides the source material for Tier 2 extraction.
    │
    │  FactExtractor (1 LLM call post-session)
    ▼
Tier 2 — UserFacts (extracted, temporally-grounded facts)
    Stored in UserMemoryStore (Dapr state).
    Facts have valid_from / valid_until — contradicted facts are expired,
    not overwritten. Deduplication by cosine similarity (threshold 0.90).
    │
    │  ProfileSynthesizer (1 LLM call every 10 new facts, configurable)
    ▼
Tier 1 — UserProfile (synthesized persona)
    Stored in UserMemoryStore (Dapr state).
    Fields: expertise_level (1–5), expertise_domains, communication_style,
    preferred_depth, active_goals, past_key_decisions, active_constraints.
    Rebuilt from scratch on each synthesis — no incremental updates.
```

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `UserMemoryAdapter` | `adapter.py` | Connects Tier 2/1 hooks into `AgentRunner` (pre-run inject + post-session extract) |
| `FactExtractor` | `extractor.py` | Post-session LLM call → `list[UserFact]` |
| `ProfileSynthesizer` | `synthesizer.py` | Fires every N new facts → rebuilt `UserProfile` |
| `UserMemoryStore` | `store.py` | Dapr-backed CRUD for `UserFact` and `UserProfile` |
| `UserFact` | `types.py` | Temporally-grounded fact (content, category, valid_from, valid_until, confidence) |
| `UserProfile` | `types.py` | Synthesized persona (expertise_level, domains, communication_style, ...) |
| `UserMemoryContext` | `types.py` | Returned by `get_context()`: profile + relevant_facts + formatted_context |
| `UserFactCategory` | `types.py` | Enum: `EXPERTISE, PREFERENCE, DECISION, CONTEXT, CONSTRAINT` |

---

## How to use this package

```python
from grampus.memory.user.store import UserMemoryStore
from grampus.memory.user.extractor import FactExtractor
from grampus.memory.user.synthesizer import ProfileSynthesizer
from grampus.memory.user.adapter import UserMemoryAdapter

store = UserMemoryStore(state_store=dapr_store)
extractor = FactExtractor(model_client=client, model_id="claude-haiku-4-5-20251001")
synthesizer = ProfileSynthesizer(model_client=client, model_id="claude-haiku-4-5-20251001")

adapter = UserMemoryAdapter(
    store=store,
    extractor=extractor,
    synthesizer=synthesizer,
    embedding_service=embed_svc,
    synthesis_threshold=10,   # rebuild profile every 10 new facts
)

runner = AgentRunner(
    ...
    user_memory_adapter=adapter,
)
# Pass user_id at call time:
result = await runner.run(agent_def, user_input, session_id="sess-1", user_id="user-42")
```

### Standalone context retrieval

```python
# Get context to inject into any agent:
ctx = await adapter.get_context(user_id="user-42", current_query="How do I debug this?")
# ctx.formatted_context: string ready to prepend to system prompt
# ctx.relevant_facts: cosine-similarity-selected UserFacts for this query
# ctx.profile: full UserProfile (expertise level, communication style, etc.)
```

---

## Fact deduplication

Before storing a new `UserFact`, `UserMemoryStore` embeds it and computes cosine similarity against all existing facts for that user. If similarity > 0.90 against any existing fact:
- The existing fact's `confidence` is updated via EMA (exponential moving average)
- No new record is created

This prevents the fact list from growing without bound on repeated similar interactions.

---

## Temporal validity

Every `UserFact` has `valid_from` and `valid_until`:
- `valid_until=None` means "still true" — this is the default
- When a new fact contradicts an existing one (same category + semantic overlap), `ProfileSynthesizer` expires the old fact by setting `valid_until=now`
- Expired facts are stored but filtered out from `get_context()` via `fact.is_valid`
- History of what was true and when is preserved for audit

---

## Hard invariants

- **`user_id=None` silently skips all hooks.** There is no implicit user tracking. Callers must pass `user_id` explicitly.
- **`UserMemoryAdapter` hooks are wrapped in `contextlib.suppress(Exception)`.** User memory extraction never crashes agent execution. Check logs for `user_memory_extract_failed` or `profile_synthesis_failed`.
- **`ProfileSynthesizer` only fires every `synthesis_threshold` new facts** (default 10). This prevents thrashing on rapid-fire short sessions. The threshold is per-user, tracked by `UserProfile.synthesis_fact_count`.
- **Context injection is selective** — `get_context()` embeds the current query and returns only the facts with highest cosine similarity. The full `UserFact` list is never injected wholesale into the prompt.
- **`UserFact` and `UserProfile` are scoped to `user_id`, not `agent_id`.** The same user model is shared across all agents that use the same `UserMemoryStore`. This is intentional — user preferences transfer across agents.

---

## Dependency map

```
user/ depends on:      core/, memory/types.py, memory/embeddings.py,
                       dapr/ (via state_store)
user/ is imported by:  orchestration/runner.py
user/ must NOT import from: tools/, safety/, orchestration/
```

---

## ADR references

- **ADR-017** — Three-tier user memory hierarchy: full design rationale and research basis (Beyond Dialogue Time, Bi-Mem, HMO)
