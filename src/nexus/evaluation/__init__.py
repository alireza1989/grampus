"""Nexus evaluation framework — structured testing for agent behaviors."""

from nexus.evaluation.assertions import (
    Assertion,
    AssertionResult,
    contains,
    json_schema_valid,
    llm_judge,
    matches_regex,
    max_cost,
    max_duration,
    max_steps,
    no_injection_patterns,
    no_pii,
    not_contains,
    output_length,
    semantic_similarity,
    status_is,
    tool_call_count,
    tool_not_called,
    tool_was_called,
)
from nexus.evaluation.baseline import (
    BaselineRun,
    QualityBaseline,
    RegressionReport,
)
from nexus.evaluation.prompt_versions import (
    PromptDiff,
    PromptVersion,
    PromptVersionManager,
)
from nexus.evaluation.reporter import (
    EvalReport,
    EvalReporter,
    ReportFormat,
)
from nexus.evaluation.suite import (
    CaseResult,
    EvalCase,
    EvalSuite,
    SuiteResult,
)

__all__ = [
    # assertions
    "Assertion",
    "AssertionResult",
    "contains",
    "not_contains",
    "matches_regex",
    "output_length",
    "tool_was_called",
    "tool_not_called",
    "tool_call_count",
    "json_schema_valid",
    "status_is",
    "max_cost",
    "max_duration",
    "max_steps",
    "semantic_similarity",
    "llm_judge",
    "no_pii",
    "no_injection_patterns",
    # suite
    "EvalCase",
    "CaseResult",
    "SuiteResult",
    "EvalSuite",
    # prompt versions
    "PromptVersion",
    "PromptDiff",
    "PromptVersionManager",
    # baseline
    "BaselineRun",
    "RegressionReport",
    "QualityBaseline",
    # reporter
    "EvalReport",
    "EvalReporter",
    "ReportFormat",
]
