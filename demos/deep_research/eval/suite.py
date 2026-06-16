"""Evaluation suite for the Deep Research Demo.

Usage:
    python demos/deep_research/eval_suite.py
    nexus eval demos/deep_research/eval_suite.py --format text
"""

from __future__ import annotations

from demos.deep_research.agent import create_agent_def, create_runner
from demos.deep_research.eval.baseline import get_baseline
from demos.deep_research.eval.cases import EVAL_CASES
from grampus.evaluation.suite import EvalSuite


def create_suite() -> EvalSuite:
    """Build and return the configured EvalSuite. Called by ``nexus eval``."""
    suite = EvalSuite(
        "deep-research-eval",
        agent_runner=create_runner(),
        agent_def=create_agent_def(),
        session_id_prefix="eval",
        concurrency=2,
    )
    suite.add_cases(EVAL_CASES)
    return suite


__all__ = ["create_suite", "get_baseline"]
