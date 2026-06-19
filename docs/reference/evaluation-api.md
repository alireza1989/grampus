# Evaluation API Reference

## EvalSuite

Runs a collection of `EvalCase` objects against an `AgentRunner`.

::: grampus.evaluation.suite.EvalSuite
    options:
      show_source: false
      members: [add_case, add_cases, run, run_case]

---

## EvalCase

::: grampus.evaluation.suite.EvalCase
    options:
      show_source: false
      members: []

---

## Results

### SuiteResult

::: grampus.evaluation.suite.SuiteResult
    options:
      show_source: false
      members: []

### CaseResult

::: grampus.evaluation.suite.CaseResult
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

::: grampus.evaluation.assertions.contains
    options:
      show_source: false

::: grampus.evaluation.assertions.not_contains
    options:
      show_source: false

::: grampus.evaluation.assertions.matches_regex
    options:
      show_source: false

::: grampus.evaluation.assertions.output_length
    options:
      show_source: false

### Tool calls

::: grampus.evaluation.assertions.tool_was_called
    options:
      show_source: false

::: grampus.evaluation.assertions.tool_not_called
    options:
      show_source: false

::: grampus.evaluation.assertions.tool_call_count
    options:
      show_source: false

### Structured output

::: grampus.evaluation.assertions.json_schema_valid
    options:
      show_source: false

::: grampus.evaluation.assertions.status_is
    options:
      show_source: false

### Budget and performance

::: grampus.evaluation.assertions.max_cost
    options:
      show_source: false

::: grampus.evaluation.assertions.max_duration
    options:
      show_source: false

::: grampus.evaluation.assertions.max_steps
    options:
      show_source: false

### LLM-as-judge

::: grampus.evaluation.assertions.semantic_similarity
    options:
      show_source: false

::: grampus.evaluation.assertions.llm_judge
    options:
      show_source: false

### Safety

::: grampus.evaluation.assertions.no_pii
    options:
      show_source: false

::: grampus.evaluation.assertions.no_injection_patterns
    options:
      show_source: false

---

## Prompt version manager

::: grampus.evaluation.prompt_versions.PromptVersionManager
    options:
      show_source: false
      members: [register, list_versions, diff, set_active, get_active]

---

## Quality baseline

::: grampus.evaluation.baseline.QualityBaseline
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

::: grampus.evaluation.reporter.EvalReporter
    options:
      show_source: false
      members: [render, print, publish]

---

## Red-Team API

### Types

```python
from grampus.evaluation.red_team import (
    AttackCategory,   # prompt_injection | jailbreak | reasoning_hijack
                      # memory_poison | tool_misuse | excessive_agency
    AttackVariant,    # direct_injection | indirect_injection | roleplay_jailbreak
                      # encoding_jailbreak | logic_trap | memory_write_inject
                      # memory_read_poison | tool_loop | tool_chain_escape
                      # scope_escalation | implicit_permission
    OWASPCategory,    # ASI01_GOAL_HIJACK | ASI02_TOOL_MISUSE | ASI06_MEMORY_POISON | ...
    SecurityProperty, # task_alignment | action_alignment | source_authorization | data_isolation
    Severity,         # critical | high | medium | low | info
    AttackPayload,
    JudgeVerdict,
    AttackResult,
    RedTeamTargetConfig,
    RedTeamCampaignConfig,
)
```

### RedTeamTargetConfig

```python
class RedTeamTargetConfig(BaseModel):
    agent_name: str
    system_prompt: str
    available_tools: list[str] = []
    memory_enabled: bool = False
    crew_enabled: bool = False
    max_turns: int = 1             # 1–10; >1 enables multi-turn strategy attacks
```

### RedTeamCampaignConfig

```python
class RedTeamCampaignConfig(BaseModel):
    campaign_id: str
    target: RedTeamTargetConfig
    enabled_categories: list[AttackCategory]
    payloads_per_strategy: int = 5    # 1–50
    max_concurrent: int = 5           # 1–10
    stop_on_critical: bool = False
```

### AttackerAgent

::: grampus.evaluation.red_team.attacker.AttackerAgent
    options:
      show_source: false
      members: [generate_payloads, mutate_failed]

### RedTeamJudge

::: grampus.evaluation.red_team.judge.RedTeamJudge
    options:
      show_source: false
      members: [evaluate]

### RedTeamRunner

::: grampus.evaluation.red_team.runner.RedTeamRunner
    options:
      show_source: false
      members: [run]

### RedTeamReport

::: grampus.evaluation.red_team.report.RedTeamReport
    options:
      show_source: false
      members: [build, to_text, to_json]

### Writing a custom attack strategy

```python
from grampus.evaluation.red_team.strategies.base import BaseAttackStrategy
from grampus.evaluation.red_team.types import (
    AttackCategory, AttackPayload, AttackVariant, RedTeamTargetConfig,
)


class MyCustomStrategy(BaseAttackStrategy):
    @property
    def category(self) -> AttackCategory:
        return AttackCategory.EXCESSIVE_AGENCY

    @property
    def name(self) -> str:
        return "my_custom"

    async def generate(
        self, target: RedTeamTargetConfig, count: int = 5
    ) -> list[AttackPayload]:
        try:
            return [
                AttackPayload(
                    content=f"Custom attack payload {i}",
                    attack_category=self.category,
                    attack_variant=AttackVariant.SCOPE_ESCALATION,
                    strategy_name=self.name,
                )
                for i in range(count)
            ]
        except Exception:
            return []
```

Pass it to `AttackerAgent`:

```python
from grampus.evaluation.red_team.attacker import AttackerAgent
from grampus.evaluation.red_team.strategies import ALL_STRATEGIES

attacker = AttackerAgent(
    strategies=[S() for S in ALL_STRATEGIES] + [MyCustomStrategy()]
)
```

See the [Red-Teaming guide](../guides/red_teaming.md) for a full walkthrough.

---

## Writing a custom assertion

```python
from grampus.evaluation.assertions import AssertionResult
from grampus.core.types import ExecutionResult


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
