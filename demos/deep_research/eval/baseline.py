"""Quality baseline for the Deep Research Demo eval suite."""

from __future__ import annotations

from grampus.evaluation.baseline import QualityBaseline

_baseline = QualityBaseline(suite_name="deep-research-eval", regression_threshold=0.10)


def get_baseline() -> QualityBaseline:
    """Return the shared QualityBaseline instance. Called by ``nexus eval``."""
    return _baseline
