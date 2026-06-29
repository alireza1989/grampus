# `grampus/memory/reflexion/` — Agent Self-Improvement (F1)

This sub-package implements the two tiers of agent self-improvement described in ADR-016: verbal self-reflection after failures (Reflexion, NeurIPS 2023) and reusable skill extraction after successes (SAGE, arXiv 2512.17102). Both tiers store their outputs as `Procedure` records in the existing `ProceduralMemory` — no new Dapr key namespaces or storage infrastructure.

Both tiers are **opt-in hooks** wired into `AgentRunner`. When `reflexion_engine=None, skill_library=None` (the defaults), the runner behaves identically to a plain ReAct loop with zero overhead.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `ReflexionEngine` | `engine.py` | Post-failure hook: generates + rates + stores verbal reflections |
| `SkillLibrary` | `skill_library.py` | Post-success hook: extracts + manages reusable skill procedures |
| `PromptOptimizer` | `optimizer.py` | Proposes and evaluates system prompt mutations via EvalSuite |
| `ReflexionHookResult` | `types.py` | Result of `observe_failure()`: reflection text + quality score |
| `SkillExtractionResult` | `types.py` | Result of `observe_success()`: skill name + steps or `extractable=False` |

---

## How these hooks wire into `AgentRunner`

```python
from grampus.memory.reflexion.engine import ReflexionEngine
from grampus.memory.reflexion.skill_library import SkillLibrary

engine = ReflexionEngine(
    procedural_memory=pm,
    embedding_service=embed_svc,
    max_reflections=50,
    quality_threshold=0.3,   # reflections below this stored but not surfaced
)

library = SkillLibrary(
    procedural_memory=pm,
    embedding_service=embed_svc,
)

runner = AgentRunner(
    ...
    reflexion_engine=engine,
    skill_library=library,
)
# AgentRunner calls these automatically in its post-run hooks.
# You never call engine.observe_failure() or library.observe_success() directly.
```

### Pre-run hint injection (how context is enriched)

Before every LLM call, `AgentRunner` retrieves relevant reflections and validated skills and injects them into the system prompt:

```python
# Both calls are wrapped in contextlib.suppress — they NEVER crash the runner
hints = await engine.get_hints(task_description, model_client, top_k=3)
skills = await library.get_approach_hints(task_description, top_k=3)
# → appended to system prompt as: "Past reflections:\n..." / "Known approaches:\n..."
```

---

## Skill lifecycle (SAGE)

Skills transition through these states automatically:

```
Extracted (validated=False)
        │
        │ ≥ 3 successful uses
        ▼
Validated (validated=True) ← surfaced as approach hints
        │
        │ ≥ 5 uses with success_rate < 0.4
        ▼
Demoted (validated=False)
        │
        │ ≥ 5 uses with success_rate < 0.2
        ▼
Deleted (removed from ProceduralMemory)
```

Skill records are `Procedure` entries with `procedure_type=ProcedureType.SKILL`.

---

## Reflection lifecycle (ME-ICPO)

```
Task fails
    │
    ▼
ReflexionEngine.observe_failure()
    ├─ LLM call 1: generate verbal reflection (2-4 sentences)
    ├─ LLM call 2: rate reflection quality (0.0–1.0)
    │   quality < 0.3 → stored with low confidence, NOT surfaced as hints
    │   quality ≥ 0.3 → stored and surfaced on similar future tasks
    └─ Stored as Procedure(procedure_type=REFLECTION) in ProceduralMemory
           │
           │ if max_reflections exceeded → oldest pruned
```

---

## Hard invariants

- **Both hooks are wrapped in `contextlib.suppress(Exception)` in `AgentRunner`.** A failure in reflexion or skill extraction never crashes the agent run. Check logs for `reflexion_failure` or `skill_extract_failure` events.
- **`quality_threshold=0.3` filters out low-signal reflections from hints.** All reflections are stored regardless of quality — only high-quality ones are surfaced to the LLM context. Never remove stored reflections based solely on quality score.
- **Skills and reflections are `Procedure` records in `ProceduralMemory`.** They use `procedure_type=ProcedureType.SKILL` or `ProcedureType.REFLECTION` to distinguish them. Queries should always filter by `procedure_type`.
- **`PromptOptimizer.optimize()` runs N+1 `EvalSuite.run()` calls** (1 baseline + N candidates). Never use it on production agents or with expensive models without cost controls.

---

## Dependency map

```
reflexion/ depends on:      core/, memory/types.py, memory/procedural.py,
                            memory/embeddings.py
reflexion/ is imported by:  orchestration/runner.py
reflexion/ must NOT import from: dapr/ (uses it via memory/procedural.py),
                                 tools/, safety/, orchestration/
```

---

## ADR references

- **ADR-016** — Dual-tier agent self-improvement design decisions (full rationale)
