# Persistent User Modeling Guide

## Why User Modeling Matters

A generic AI assistant treats every conversation as a blank slate. It cannot remember that you are a
senior backend engineer who prefers concise, technical answers, or that you are currently mid-way
through a Kubernetes migration. This gap forces users to re-introduce themselves every session and
prevents agents from adapting their reasoning depth or vocabulary to the individual.

Grampus F2 introduces a three-tier user memory hierarchy backed by peer-reviewed research. Two key
findings shaped the design: **Bi-Mem** (arXiv 2601.06490) showed that extracting facts bottom-up
from conversations and then synthesizing a persona top-down — rather than doing either alone —
prevents hallucination amplification that occurs when flat profiles are clustered. **HMO** (arXiv
2604.01670) demonstrated that a persona-driven promotion/demotion mechanism outperforms static
flat profiles because the most-accessed facts bubble up while stale background context recedes.
Additionally, **Beyond Dialogue Time** (arXiv 2601.07468) established that every fact must carry
temporal validity metadata: a user's job title from 2024 should not be asserted in 2026.

The result: agents that remember expertise, preferences, decisions, and constraints across sessions,
inject only the contextually-relevant facts before each LLM call, and continuously improve their
model of each individual user.

---

## Quick Start

```python
from grampus.memory.user import (
    UserMemoryStore, FactExtractor, ProfileSynthesizer, UserMemoryAdapter
)

store = UserMemoryStore(state_store=dapr_store, embedding_service=embedding_svc)
adapter = UserMemoryAdapter(
    store=store,
    extractor=FactExtractor(
        store=store,
        episodic_memory=episodic_mem,
        embedding_service=embedding_svc,
    ),
    synthesizer=ProfileSynthesizer(store=store),
    embedding_service=embedding_svc,
)

runner = AgentRunner(
    model_client=model_client,
    tool_executor=executor,
    user_memory_adapter=adapter,
)

# Pass user_id on every run — same user_id = same memory across sessions
result = await runner.run(agent_def, task, session_id="s1", user_id="alice")
```

`user_id` is explicit — the caller decides which user identity to track. This means the same
`UserMemoryStore` can serve multiple users and multiple agents without key collisions.

---

## What Gets Extracted and How

After each session completes, `FactExtractor` reads the raw episodic records for that session,
assembles them into a conversation string, and calls a fast LLM (Haiku) to extract structured facts.

**Example conversation:**

> User: "I've been writing Go for about five years. My team refuses to use any libraries with
> GPL licenses. Can you help me find a fast HTTP router?"

**Facts extracted:**

| Content | Category | Confidence |
|---|---|---|
| Has five years of Go experience | expertise | 0.9 |
| Team policy forbids GPL-licensed libraries | constraint | 0.85 |
| Looking for a fast HTTP router | context | 0.8 |

**Profile synthesized** (after ≥ 10 new facts):

```json
{
  "expertise_level": 4,
  "expertise_domains": ["Go", "backend", "HTTP"],
  "communication_style": "balanced",
  "preferred_depth": "deep-dive",
  "active_goals": ["find fast HTTP router"],
  "active_constraints": ["no GPL libraries"]
}
```

---

## Temporal Validity

Every `UserFact` has `valid_from` and `valid_until` fields. A fact with `valid_until=None` is
currently true. When the FactExtractor detects a contradiction (e.g., "I switched from Python to
Rust"), it expires the old fact by setting `valid_until = now()` and stores the new one. The
original fact is preserved in the store for audit purposes.

**Manually expire a stale fact:**

```python
# Expire a specific fact (e.g., outdated job title)
await store.expire_fact(user_id="alice", fact_id="fact-abc123")
```

**Check fact validity:**

```python
fact = await store.get_fact("alice", "fact-abc123")
if fact and fact.is_valid:
    print("still current:", fact.content)
```

---

## Inspecting and Managing the User Model

```python
# List all currently-valid facts
facts = await store.get_valid_facts("alice")
for f in facts:
    print(f"[{f.category.value}] {f.content} (confidence={f.confidence:.2f})")

# Filter by category
expertise_facts = await store.get_valid_facts("alice", category=UserFactCategory.EXPERTISE)

# Load the synthesized profile
profile = await store.get_profile("alice")
if profile:
    print(f"Expertise level: {profile.expertise_level}/5")
    print(f"Domains: {profile.expertise_domains}")
    print(f"Communication style: {profile.communication_style}")

# List all facts including expired ones (for audit)
all_facts = await store.list_all_facts("alice")
```

---

## What the Agent Sees

Before each LLM call, `UserMemoryAdapter.get_context()` embeds the current task query, finds the
top-5 most similar facts by cosine similarity, and formats them as a system message prefix:

```
User profile for alice:
- Expertise: level 4/5 in Go, backend, HTTP
- Communication style: balanced, preferred depth: deep-dive
- Active goals: find fast HTTP router
- Constraints: no GPL libraries

Relevant facts about this user:
  [constraint] Team policy forbids GPL-licensed libraries
  [expertise] Has five years of Go experience
  [context] Looking for a fast HTTP router
```

Only facts relevant to the current query are injected — the full fact list is never passed to the
LLM wholesale. The cosine similarity gate prevents irrelevant background context from inflating the
prompt.

---

## Synthesis Tuning

```python
# Synthesize more aggressively (every 5 new facts instead of 10)
synthesizer = ProfileSynthesizer(store=store, synthesis_interval=5)

# Force an immediate re-synthesis regardless of threshold
result = await synthesizer.synthesize("alice", model_client, force=True)
print(f"Profile version: {result.new_version}")
```

---

## Deduplication Behaviour

- **Similarity > 0.90 (same category):** confidence is updated with EMA (α=0.3) rather than
  creating a duplicate record.
- **Similarity 0.50–0.90:** the LLM is asked whether the new fact contradicts the existing one.
  If yes, the existing fact is expired and the new one is stored.
- **Similarity < 0.50:** stored as a new, independent fact.

This means repeated information across sessions strengthens confidence rather than inflating the
fact count.
