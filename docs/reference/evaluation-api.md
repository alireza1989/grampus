# Evaluation API Reference

## EvalSuite

Runs a collection of `EvalCase` objects against an `AgentRunner`.

::: nexus.evaluation.suite.EvalSuite
    options:
      show_source: false
      members: [add_case, add_cases, run, run_case]

---

## EvalCase

::: nexus.evaluation.suite.EvalCase
    options:
      show_source: false
      members: []

---

## Results

### SuiteResult

::: nexus.evaluation.suite.SuiteResult
    options:
      show_source: false
      members: []

### CaseResult

::: nexus.evaluation.suite.CaseResult
    options:
      show_source: false
      members: []

### AssertionResult

```python
@dataclass
class AssertionResult:
    passed: bool
    assertion_type: str    # e.g., "contains", "tool_was_called"
    detail: str            # human-readable description
    score: float           # 0.0–1.0 (1.0 = fully passed)
    expected: str | None
    actual: str | None
```

---

## Assertion factories

All assertion factories return `Assertion` objects (async callables).

### Output content

::: nexus.evaluation.assertions.contains
    options:
      show_source: false

::: nexus.evaluation.assertions.not_contains
    options:
      show_source: false

::: nexus.evaluation.assertions.matches_regex
    options:
      show_source: false

::: nexus.evaluation.assertions.output_length
    options:
      show_source: false

### Tool calls

::: nexus.evaluation.assertions.tool_was_called
    options:
      show_source: false

::: nexus.evaluation.assertions.tool_not_called
    options:
      show_source: false

::: nexus.evaluation.assertions.tool_call_count
    options:
      show_source: false

### Structured output

::: nexus.evaluation.assertions.json_schema_valid
    options:
      show_source: false

::: nexus.evaluation.assertions.status_is
    options:
      show_source: false

### Budget and performance

::: nexus.evaluation.assertions.max_cost
    options:
      show_source: false

::: nexus.evaluation.assertions.max_duration
    options:
      show_source: false

::: nexus.evaluation.assertions.max_steps
    options:
      show_source: false

### LLM-as-judge

::: nexus.evaluation.assertions.semantic_similarity
    options:
      show_source: false

::: nexus.evaluation.assertions.llm_judge
    options:
      show_source: false

### Safety

::: nexus.evaluation.assertions.no_pii
    options:
      show_source: false

::: nexus.evaluation.assertions.no_injection_patterns
    options:
      show_source: false

---

## Prompt version manager

::: nexus.evaluation.prompt_versions.PromptVersionManager
    options:
      show_source: false
      members: [register, list_versions, diff, set_active, get_active]

---

## Quality baseline

::: nexus.evaluation.baseline.QualityBaseline
    options:
      show_source: false
      members: [pin, compare]

### BaselineComparison

```python
@dataclass
class BaselineComparison:
    regressed: bool
    baseline_pass_rate: float
    current_pass_rate: float
    delta: float                      # current - baseline
    degraded_cases: list[str]         # case names that regressed
    improved_cases: list[str]         # case names that improved
```

---

## Reporters

::: nexus.evaluation.reporter.EvalReporter
    options:
      show_source: false
      members: [report, render]

---

## Writing a custom assertion

```python
from nexus.evaluation.assertions import AssertionResult
from nexus.core.types import ExecutionResult


class WordCountAssertion:
    """Assert the output contains between min_words and max_words words."""

    def __init__(self, min_words: int, max_words: int) -> None:
        self.min_words = min_words
        self.max_words = max_words

    async def __call__(self, result: ExecutionResult) -> AssertionResult:
        output = result.output or ""
        word_count = len(output.split())
        passed = self.min_words <= word_count <= self.max_words
        return AssertionResult(
            passed=passed,
            assertion_type="word_count",
            detail=f"Word count {word_count} {'within' if passed else 'outside'} [{self.min_words}, {self.max_words}]",
            score=1.0 if passed else 0.0,
            expected=f"{self.min_words}–{self.max_words} words",
            actual=f"{word_count} words",
        )


# Use in an EvalCase
case = EvalCase(
    name="medium_length_response",
    input="Explain photosynthesis briefly.",
    assertions=[WordCountAssertion(min_words=50, max_words=200)],
)
```
