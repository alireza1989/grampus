# Evaluation Guide

## What you'll learn

- Why eval-driven development is essential for agents
- Structure of `EvalCase` and `EvalSuite`
- All 16 assertion types with examples
- Prompt version management
- Quality baselines and regression detection
- CI integration with JUnit XML output

---

## Why eval-driven development?

Unit tests verify that code runs correctly. Eval suites verify that agents *behave* correctly. An agent can pass all its unit tests and still:

- Hallucinate tool arguments
- Fail to use a required tool
- Produce output that violates safety policies
- Cost 3× more than expected
- Regress when the system prompt is tweaked

Eval suites catch all of these.

---

## EvalCase structure

```python
from nexus.evaluation.suite import EvalCase
from nexus.evaluation.assertions import contains, tool_was_called, max_cost

case = EvalCase(
    name="research_uses_search",
    description="Research agent must call web_search before answering",
    input="What is the capital of Brazil?",
    tags=["smoke", "regression"],
    assertions=[
        tool_was_called("web_search"),
        contains("Brasília"),
        max_cost(0.05),
    ],
)
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique case identifier |
| `description` | `str` | Human-readable description |
| `input` | `str` | User message sent to the agent |
| `tags` | `list[str]` | Used for filtering: `smoke`, `regression`, `safety`, etc. |
| `assertions` | `list[Assertion]` | Checks run against the `ExecutionResult` |

---

## EvalSuite

```python
from nexus.evaluation.suite import EvalSuite

suite = EvalSuite(
    name="research-agent-suite",
    agent_runner=runner,
    agent_def=agent_def,
    session_id_prefix="eval",   # sessions: eval-0, eval-1, ...
    concurrency=4,              # run 4 cases in parallel
    tags=["smoke"],             # run only cases tagged "smoke" (None = run all)
)

suite.add_case(case)
suite.add_cases([case2, case3, case4])

# Chain style
suite.add_case(case5).add_case(case6)

result = await suite.run()
```

### Reading results

```python
print(f"Suite: {result.suite_name}")
print(f"Pass rate: {result.pass_rate:.0%}  ({result.passed}/{result.total_cases})")
print(f"Total cost: ${result.total_cost_usd:.4f}")
print(f"Avg duration: {result.avg_duration_seconds:.2f}s")

for case_result in result.case_results:
    status = "PASS" if case_result.passed else "FAIL"
    print(f"  [{status}] {case_result.case_name}")
    if not case_result.passed:
        for ar in case_result.assertion_results:
            if not ar.passed:
                print(f"       {ar.assertion_type}: {ar.detail}")
```

---

## All 16 assertion types

### Output content

**`contains(expected, *, case_sensitive=True)`**

```python
from nexus.evaluation.assertions import contains

contains("Brasília")                           # output must contain "Brasília"
contains("brasília", case_sensitive=False)     # case-insensitive match
```

**`not_contains(forbidden, *, case_sensitive=True)`**

```python
from nexus.evaluation.assertions import not_contains

not_contains("I don't know")     # agent must not say this
not_contains("Error")
```

**`matches_regex(pattern)`**

```python
from nexus.evaluation.assertions import matches_regex

matches_regex(r"\d{4}-\d{2}-\d{2}")     # output contains a date
matches_regex(r"https?://\S+")           # output contains a URL
```

**`output_length(*, min_chars=None, max_chars=None)`**

```python
from nexus.evaluation.assertions import output_length

output_length(min_chars=100)             # at least 100 chars
output_length(max_chars=2000)            # at most 2000 chars
output_length(min_chars=50, max_chars=500)
```

---

### Tool calls

**`tool_was_called(tool_name)`**

```python
from nexus.evaluation.assertions import tool_was_called

tool_was_called("web_search")       # web_search must have been called
```

**`tool_not_called(tool_name)`**

```python
from nexus.evaluation.assertions import tool_not_called

tool_not_called("delete_file")      # delete_file must NOT have been called
```

**`tool_call_count(*, min_calls=None, max_calls=None)`**

```python
from nexus.evaluation.assertions import tool_call_count

tool_call_count(min_calls=1)             # at least one tool call
tool_call_count(max_calls=5)             # no more than 5 tool calls
tool_call_count(min_calls=2, max_calls=4)
```

---

### Structured output

**`json_schema_valid(schema)`**

```python
from nexus.evaluation.assertions import json_schema_valid

json_schema_valid({
    "type": "object",
    "required": ["answer", "sources"],
    "properties": {
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
})
```

**`status_is(expected_status)`**

```python
from nexus.core.types import AgentStatus
from nexus.evaluation.assertions import status_is

status_is(AgentStatus.COMPLETED)    # agent must not have failed
```

---

### Budget and performance

**`max_cost(limit_usd)`**

```python
from nexus.evaluation.assertions import max_cost

max_cost(0.10)      # must cost less than $0.10
```

**`max_duration(limit_seconds)`**

```python
from nexus.evaluation.assertions import max_duration

max_duration(30.0)  # must complete in under 30 seconds
```

**`max_steps(limit)`**

```python
from nexus.evaluation.assertions import max_steps

max_steps(5)        # agent must complete in 5 or fewer iterations
```

---

### LLM-as-judge

**`semantic_similarity(expected, *, model_client, threshold=0.8)`**

Use this when exact string matching is too brittle:

```python
from nexus.evaluation.assertions import semantic_similarity

semantic_similarity(
    expected="Brasília is the capital of Brazil, founded in 1960.",
    model_client=client,
    threshold=0.8,     # 0.8 cosine similarity required
)
```

**`llm_judge(criteria, *, model_client, threshold=0.7)`**

Ask a second LLM to score the output against free-text criteria:

```python
from nexus.evaluation.assertions import llm_judge

llm_judge(
    criteria=(
        "The response must: (1) name the correct capital city, "
        "(2) cite at least one source, (3) be written in a professional tone."
    ),
    model_client=client,
    threshold=0.7,     # LLM judge must score >= 0.7/1.0
)
```

---

### Safety assertions

**`no_pii(pii_types=None)`**

```python
from nexus.evaluation.assertions import no_pii

no_pii()                              # no PII of any type in output
no_pii(pii_types=["email", "phone"])  # no email or phone in output
```

**`no_injection_patterns()`**

```python
from nexus.evaluation.assertions import no_injection_patterns

no_injection_patterns()    # output contains no prompt injection patterns
```

---

## Prompt version management

```python
from nexus.evaluation.prompt_versions import PromptVersionManager

manager = PromptVersionManager(state_store=state_store)

# Register a new version
await manager.register(
    agent_name="research-agent",
    version="v2",
    system_prompt="You are a research assistant. Always cite sources with URLs.",
    notes="Added URL citation requirement",
)

# List all versions
versions = await manager.list_versions("research-agent")
for v in versions:
    print(f"  {v.version}: {v.notes}")

# Diff two versions
diff = await manager.diff("research-agent", from_version="v1", to_version="v2")
print(diff)

# A/B test: run the same eval suite against both versions
result_v1 = await suite_v1.run()
result_v2 = await suite_v2.run()
print(f"v1 pass rate: {result_v1.pass_rate:.0%}")
print(f"v2 pass rate: {result_v2.pass_rate:.0%}")

# Pin the winning version
await manager.set_active(agent_name="research-agent", version="v2")

# Rollback if needed
await manager.set_active(agent_name="research-agent", version="v1")
```

---

## Quality baselines

Pin a baseline score and detect regressions automatically:

```python
from nexus.evaluation.baseline import QualityBaseline

baseline = QualityBaseline(state_store=state_store, agent_name="research-agent")

# Establish baseline after a known-good run
result = await suite.run()
await baseline.pin(suite_result=result)
print(f"Baseline pinned: {result.pass_rate:.0%}")

# On subsequent runs, compare against baseline
new_result = await suite.run()
comparison = await baseline.compare(new_result)

if comparison.regressed:
    print(f"REGRESSION: pass rate dropped from {comparison.baseline_pass_rate:.0%} "
          f"to {comparison.current_pass_rate:.0%}")
    print(f"Degraded cases: {comparison.degraded_cases}")
else:
    print(f"No regression detected ({new_result.pass_rate:.0%})")
```

---

## Reporters

=== "Text (default)"

    ```python
    from nexus.evaluation.reporter import TextReporter

    reporter = TextReporter()
    reporter.report(result)     # prints to stdout
    ```

=== "JSON"

    ```python
    from nexus.evaluation.reporter import JSONReporter

    reporter = JSONReporter()
    json_output = reporter.render(result)
    with open("eval_report.json", "w") as f:
        f.write(json_output)
    ```

=== "JUnit XML (CI)"

    ```python
    from nexus.evaluation.reporter import JUnitReporter

    reporter = JUnitReporter()
    xml_output = reporter.render(result)
    with open("eval_results.xml", "w") as f:
        f.write(xml_output)
    ```

---

## CI integration

Gate deployments on eval pass rate:

```bash
# Fail the pipeline if pass rate drops below 90%
nexus eval eval_suite.py --format junit --output results.xml --fail-under 0.9
echo $?   # 0 if passed, 1 if below threshold
```

In GitHub Actions:

```yaml
- name: Run eval suite
  run: nexus eval tests/eval_suite.py --format junit --output results.xml --fail-under 0.9

- name: Publish test results
  uses: EnricoMi/publish-unit-test-result-action@v2
  if: always()
  with:
    files: results.xml
```

---

## Next steps

- **[Evaluation API reference →](../reference/evaluation-api.md)** — Full `EvalSuite` and assertion reference
- **[Safety guide →](safety.md)** — Write injection and PII safety assertions
- **[Observability guide →](observability.md)** — Correlate eval runs with OTEL traces
