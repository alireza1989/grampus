# `grampus/evaluation/` — Evaluation Framework

This package provides built-in quality measurement for agent behaviors: eval suites with typed assertions, streaming eval support, prompt version management, quality baseline tracking, and run storage. The red-team adversarial testing suite lives in [red_team/](red_team/).

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `EvalSuite` | `suite.py` | Runs a list of `EvalCase` against an agent; returns `SuiteResult` |
| `EvalCase` | `suite.py` | One test case: input, tags, and a list of `Assertion` objects |
| `CaseResult` | `suite.py` | Per-case outcome: passed, assertion_results, token_usage, latency_seconds |
| `SuiteResult` | `suite.py` | Aggregate: pass_rate, total_cost_usd, median_latency_seconds, per-case results |
| `StreamingEvalSuite` | `streaming.py` | Same as `EvalSuite` but evaluates streaming responses chunk-by-chunk |
| `Assertion` | `assertions.py` | Base class for all assertion types |
| `PromptVersionManager` | `prompt_versions.py` | Track system prompt versions, diff, A/B test, rollback |
| `QualityBaseline` | `baseline.py` | Establish score baseline; detect regression on subsequent runs |
| `EvalRunStore` | `run_store.py` | Persists `SuiteResult`s to Dapr for historical comparison |
| `EvalReporter` | `reporter.py` | Output reports to stdout, JSON, or Dapr pub/sub |

---

## Assertion types

```python
from grampus.evaluation.assertions import (
    Contains,
    NotContains,
    MatchesRegex,
    SemanticSimilarity,
    JsonSchemaValid,
    ToolWasCalled,
    ToolNotCalled,
    LLMJudge,           # uses a second LLM call to evaluate quality
)

case = EvalCase(
    id="research-case-1",
    name="Summarizes key facts",
    input="Summarize the causes of World War I",
    tags=["history", "summarization"],
    assertions=[
        Contains("assassination"),
        Contains("Franz Ferdinand"),
        NotContains("World War II"),
        MatchesRegex(r"\d{4}"),                     # contains a year
        SemanticSimilarity(
            reference="Assassination of Archduke Franz Ferdinand",
            threshold=0.6,
        ),
        ToolWasCalled("web_search"),
        LLMJudge(
            criterion="Is the summary factually accurate and concise?",
            threshold=0.7,
        ),
    ],
)
```

---

## Running an eval suite

```python
from grampus.evaluation.suite import EvalSuite

suite = EvalSuite(
    name="research-agent-v1",
    cases=[case1, case2, case3],
    agent_runner=runner,
    agent_def=agent_def,
)

result = await suite.run()
print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Total cost: ${result.total_cost_usd:.4f}")

# Detailed per-case output
for case_result in result.case_results:
    print(f"{case_result.case_id}: {'PASS' if case_result.passed else 'FAIL'}")
    for ar in case_result.assertion_results:
        if not ar.passed:
            print(f"  FAIL: {ar.assertion_type} — {ar.reason}")
```

---

## Prompt version management

```python
from grampus.evaluation.prompt_versions import PromptVersionManager

pvm = PromptVersionManager(state_store=dapr_store, agent_id="researcher")

# Save a version
v1_id = await pvm.save("You are a research assistant. Be thorough and cite sources.")

# Diff two versions
diff = await pvm.diff(v1_id, v2_id)
print(diff)   # unified diff

# A/B test: run two versions against the same eval suite and compare scores
ab_result = await pvm.ab_test(
    suite=suite,
    version_a_id=v1_id,
    version_b_id=v2_id,
)
print(f"Version A: {ab_result.score_a:.2f}, Version B: {ab_result.score_b:.2f}")

# Rollback to a previous version
await pvm.rollback(v1_id)
```

---

## Quality baselines

```python
from grampus.evaluation.baseline import QualityBaseline

baseline = QualityBaseline(
    state_store=dapr_store,
    agent_id="researcher",
    metric="pass_rate",
    regression_threshold=0.05,   # alert if pass_rate drops by more than 5%
)

# Establish a baseline from the first run
await baseline.establish(suite_result)

# Compare subsequent runs — raises if regression detected
await baseline.compare(new_suite_result)
```

---

## CLI integration

```bash
# Run eval suite from CLI
grampus eval my_suite.py

# Output options
grampus eval my_suite.py --format json --output results.json
grampus eval my_suite.py --baseline      # compare against stored baseline
```

---

## Hard invariants

- **`SemanticSimilarity` assertions use the same `EmbeddingService` as the memory system** — inject it when constructing `EvalSuite`. Do not create a separate embedding client.
- **`LLMJudge` assertions make an additional LLM call per assertion per case.** On large suites (50+ cases) with multiple `LLMJudge` assertions, cost can be substantial. Cap `max_tokens` on the judge model and use a cheap model tier (`claude-haiku-4-5-20251001`).
- **`SuiteResult.pass_rate` is `passed_count / total_cases`.** Cases that error (runner exception) count as FAIL, not as skipped. This is intentional — errors indicate regressions, not eval infrastructure issues.
- **`EvalRunStore` persists results to Dapr state** for historical comparison. Results are never deleted automatically. Clean up old results with `eval_run_store.delete_before(cutoff_date)`.
- **`PromptVersionManager` versions are content-addressed** (same content = same version ID). Saving the same system prompt twice produces the same ID and does not create a duplicate.

---

## Extension guide

### Adding a new assertion type

1. Subclass `Assertion` in `assertions.py`.
2. Implement `async def check(self, actual_output: str, execution_result: ExecutionResult) -> AssertionResult`.
3. Return `AssertionResult(passed=True/False, reason="...", assertion_type=self.__class__.__name__)`.
4. Add a test in `tests/evaluation/test_assertions.py`.

---

## Dependency map

```
evaluation/ depends on:      core/, dapr/ (EvalRunStore), memory/ (EmbeddingService for SemanticSimilarity)
evaluation/ is imported by:  cli/ (grampus eval command), memory/reflexion/ (PromptOptimizer),
                             versioning/ (VersionRouter.run_evals)
evaluation/ must NOT import from: orchestration/ (AgentRunner is injected, not imported),
                                  tools/, safety/
```

---

## ADR references

- **ADR-025** — `VersionRouter` uses `EvalSuite` for automated A/B promotion scoring
