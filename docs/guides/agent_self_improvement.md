# Agent Self-Improvement

Grampus agents can learn from their own execution history without any weight updates or retraining. Two mechanisms work in tandem: **ReflexionEngine** stores verbal lessons from failures, and **SkillLibrary** extracts reusable approaches from successes. Together they implement the dual-tier self-improvement system described in ADR-016.

## Why Static System Prompts Degrade

When an agent runs the same class of task repeatedly, a fixed system prompt has no way to incorporate what worked, what failed, or why. The agent re-makes the same mistakes in session two that it made in session one. Over many runs on similar tasks, this compounds: a SQL-querying agent that consistently forgets to handle NULLs will keep failing in exactly the same way forever.

Dual-tier improvement breaks this pattern. **Tier 1 (Reflexion, NeurIPS 2023)**: after each failure, the model is asked to verbalize what went wrong and what it should do differently. That reflection is stored in ProceduralMemory and injected into the next similar run's context. **Tier 2 (SAGE, arXiv 2512.17102)**: after each success, the execution trace is analyzed to extract a reusable, parameterized skill. Skills that are validated by repeated successful use are promoted and surfaced as approach hints for future similar tasks. The compounding effect: each run is at least as informed as the previous one.

## Quick Start

```python
from grampus.memory.reflexion import ReflexionEngine, SkillLibrary
from grampus.memory.procedural import ProceduralMemory
from grampus.memory.embeddings import EmbeddingService
from grampus.orchestration.runner import AgentRunner

dapr_store = ...       # your DaprStateStore
embedding_svc = EmbeddingService(state_store=dapr_store)

procedural_mem = ProceduralMemory(state_store=dapr_store, agent_id="my-agent")
reflexion_engine = ReflexionEngine(
    procedural_memory=procedural_mem,
    embedding_service=embedding_svc,
)
skill_library = SkillLibrary(
    procedural_memory=procedural_mem,
    embedding_service=embedding_svc,
)
runner = AgentRunner(
    model_client=model_client,
    tool_executor=executor,
    reflexion_engine=reflexion_engine,
    skill_library=skill_library,
)

# Every run() now automatically learns from failure and success
result = await runner.run(agent_def, task, session_id="s1")
```

Both parameters default to `None`. An `AgentRunner` without them is behaviorally identical to the pre-F1 runner.

## How Reflections Work

After a task failure, `ReflexionEngine.observe_failure()` makes two LLM calls:

1. **Reflection call** (temperature=0.3, max_tokens=300): "You just failed. What went wrong and what should you do differently?" The response is stored as a REFLECTION-type Procedure in ProceduralMemory.
2. **Quality rating call** (temperature=0.0, max_tokens=60): "Rate this reflection's quality on 0–1." Reflections rated below `quality_threshold` (default 0.3) are stored but not surfaced as hints — they're archived for audit without polluting the model's context window.

**Example lifecycle:**

1. Agent fails to write a SQL query that handles NULL values.
2. Reflection stored: `"Don't use LEFT JOIN when an INNER JOIN is sufficient; always check NULL handling for columns used in WHERE clauses."`
3. Next similar task: `get_relevant_reflections("write SQL query for user activity")` returns this reflection.
4. It's injected into the system message prefix before the first LLM call:
   ```
   Lessons from past failures:
   1. Don't use LEFT JOIN when an INNER JOIN is sufficient; always check NULL handling...
   ```
5. The agent starts the new task already aware of the previous mistake.

## How Skills Work

After a task completes with `status=COMPLETED`, `SkillLibrary.observe_success()` makes one LLM call (temperature=0.2, max_tokens=400) that analyzes the execution trace and attempts to extract a reusable parameterized skill:

```json
{
  "extractable": true,
  "name": "sql_query_with_null_handling",
  "description": "Query a database table while correctly handling NULL values in filter columns.",
  "steps": [
    {"action": "Identify nullable columns in the WHERE clause", "tool_name": null},
    {"action": "Use IS NULL / IS NOT NULL instead of = NULL", "tool_name": "execute_sql"},
    {"action": "Test the query with a sample that includes NULL rows", "tool_name": "execute_sql"}
  ],
  "domain_tags": ["database", "sql", "null-handling"]
}
```

If the model returns `{"extractable": false}`, the task was too one-off to generalize and nothing is stored.

### Skill Lifecycle (SAGE)

| State | Condition | Effect |
|-------|-----------|--------|
| **Unvalidated** | Just extracted | Stored but not surfaced as hints by default |
| **Validated** | ≥3 successful uses with success_rate ≥ 0.6 | Surfaced as approach hints in future tasks |
| **Demoted** | ≥5 uses with success_rate < 0.4 | `validated=False`; no longer surfaced |
| **Deleted** | ≥5 uses with success_rate < 0.2 | Removed from ProceduralMemory entirely |

Track outcomes explicitly when you know a skill was used:

```python
await skill_library.record_skill_outcome(procedure_id, success=True)
```

### Batch Tasks: SAGE Sequential Rollout

For a list of related tasks, use `run_sequential()` so skills validated on earlier tasks are available for later ones:

```python
tasks = [
    "Find the top 10 users by activity this month",
    "Find users who haven't logged in for 30 days",
    "Calculate the median session duration per user tier",
]
results = await skill_library.run_sequential(
    tasks, agent_def, runner, session_prefix="analytics-batch"
)
```

Skills extracted from task 1 are immediately available (though unvalidated) for task 2. If task 2 also succeeds using the same approach, the skill's success count increments. By task 3, the skill may already be validated and surfaced prominently.

## Prompt Optimization

`PromptOptimizer` closes the loop: given an `EvalSuite`, it proposes three candidate system prompt mutations, evaluates each, and registers the best as a new `PromptVersion` if it beats the baseline by `improvement_threshold` (default 0.05).

```python
from grampus.memory.reflexion import PromptOptimizer
from grampus.evaluation.suite import EvalSuite
from grampus.evaluation.prompt_versions import PromptVersionManager

prompt_mgr = PromptVersionManager(agent_id="my-agent")
prompt_mgr.register("1.0.0", agent_def.system_prompt or "")
prompt_mgr.activate("1.0.0")

eval_suite = EvalSuite(
    "sql-suite",
    agent_runner=runner,
    agent_def=agent_def,
)
eval_suite.add_cases([...])  # your EvalCase list

optimizer = PromptOptimizer(
    reflexion_engine=reflexion_engine,
    skill_library=skill_library,
    prompt_manager=prompt_mgr,
    eval_runner=eval_suite,
    model_client=model_client,
)
result = await optimizer.optimize(agent_def, runner)
if result.improved:
    print(f"Improved from {result.original_score:.2%} to {result.best_score:.2%}")
    print(f"Strategy: {result.best_strategy}, new version: {result.new_version}")
```

Three mutation strategies are tried in parallel:

1. **append_reflection** — appends the highest-confidence stored reflection as a system note.
2. **append_skill** — appends the highest-performing validated skill as an approach hint.
3. **rewrite_failures** — asks the LLM to rewrite the system prompt to directly address the failing eval cases.

`optimize()` never raises — on any error it returns `OptimizationResult(improved=False)`.

## Monitoring Stored Reflections and Skills

Query ProceduralMemory directly to inspect what has been learned:

```python
from grampus.memory.types import ProcedureType

# All reflections (including low-quality ones)
reflections = await procedural_mem.query_by_type(ProcedureType.REFLECTION)
high_quality = [r for r in reflections if r.confidence >= 0.3]

# All skills
skills = await procedural_mem.query_by_type(ProcedureType.SKILL)
validated = [s for s in skills if s.metadata.get("validated")]
unvalidated = [s for s in skills if not s.metadata.get("validated")]

print(f"Reflections: {len(reflections)} total, {len(high_quality)} high-quality")
print(f"Skills: {len(validated)} validated, {len(unvalidated)} unvalidated")
for skill in validated:
    total = skill.success_count + skill.failure_count
    rate = skill.success_count / total if total > 0 else 0
    print(f"  [{skill.name}] success_rate={rate:.0%} uses={total}")
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ReflexionEngine.max_reflections` | 50 | Oldest reflections pruned beyond this count |
| `ReflexionEngine.quality_threshold` | 0.3 | Minimum quality to surface a reflection as a hint |
| `SkillLibrary.min_extraction_quality` | 0.5 | Minimum model confidence to store a skill |
| `SkillLibrary.max_skills` | 100 | Lowest-performing skills pruned beyond this count |
| `PromptOptimizer.improvement_threshold` | 0.05 | Minimum pass_rate improvement to register a new version |
