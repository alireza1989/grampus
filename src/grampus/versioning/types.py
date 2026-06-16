"""Versioning data models and pure functions."""

from __future__ import annotations

import difflib
import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from grampus.core.types import AgentDefinition
from grampus.versioning.metrics import VersionMetrics


class VersionStatus(StrEnum):
    """Lifecycle state of a version deployment."""

    DRAFT = "draft"
    CANARY = "canary"
    PRODUCTION = "production"
    RETIRED = "retired"


class AgentVersion(BaseModel):
    """Immutable snapshot of an AgentDefinition at a point in time."""

    model_config = ConfigDict(frozen=True)

    version_id: str
    agent_id: str
    version_tag: str
    definition: AgentDefinition
    status: VersionStatus = VersionStatus.DRAFT
    author: str = "unknown"
    description: str = ""
    parent_version_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)


class DeploymentRecord(BaseModel):
    """Record of a version being promoted to active deployment."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    version_id: str
    deployed_at: datetime
    deployed_by: str = "system"
    previous_version_id: str | None = None


class VersionDiff(BaseModel):
    """Structured difference between two agent versions."""

    model_config = ConfigDict(frozen=True)

    version_id_a: str
    version_id_b: str
    system_prompt_diff: str
    tools_added: list[str]
    tools_removed: list[str]
    config_changes: dict[str, tuple[Any, Any]]
    has_changes: bool


class SuccessMetric(StrEnum):
    """The metric used to determine A/B test winner."""

    eval_pass_rate = "eval_pass_rate"
    avg_cost_usd = "avg_cost_usd"
    avg_latency_seconds = "avg_latency_seconds"
    error_rate = "error_rate"


class ABTestConfig(BaseModel):
    """Configuration and state for an active A/B experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    agent_id: str
    control_version_id: str
    treatment_version_id: str
    traffic_split: float
    success_metric: SuccessMetric = SuccessMetric.eval_pass_rate
    auto_promote_threshold: float = 0.05
    min_samples: int = 100
    active: bool = True
    created_at: datetime
    ended_at: datetime | None = None
    winner_version_id: str | None = None


class ABTestResult(BaseModel):
    """Current evaluation snapshot of a running A/B experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    control_metrics: VersionMetrics
    treatment_metrics: VersionMetrics
    p_value: float | None
    significant: bool
    winner_version_id: str | None
    recommendation: str


def _sort_dict(obj: Any) -> Any:
    """Recursively sort dict keys and sort list-of-strings for canonical form."""
    if isinstance(obj, dict):
        return {k: _sort_dict(v) for k in sorted(obj) for v in [obj[k]]}
    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            return sorted(obj)
        return [_sort_dict(x) for x in obj]
    return obj


def compute_version_id(definition: AgentDefinition) -> str:
    """Return a deterministic SHA-256 hex digest of the agent definition.

    The hash is stable across process restarts and Python version upgrades
    because it uses json.dumps with sorted keys, not Python's hash().
    """
    raw = definition.model_dump(mode="json")
    canonical_dict = _sort_dict(raw)
    canonical = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def diff_versions(a: AgentVersion, b: AgentVersion) -> VersionDiff:
    """Compute a structured diff between two agent versions."""
    prompt_a = a.definition.system_prompt or ""
    prompt_b = b.definition.system_prompt or ""
    prompt_diff = "".join(
        difflib.unified_diff(
            prompt_a.splitlines(keepends=True),
            prompt_b.splitlines(keepends=True),
            fromfile=f"a/{a.version_id[:8]}",
            tofile=f"b/{b.version_id[:8]}",
        )
    )

    tools_a = set(a.definition.tools)
    tools_b = set(b.definition.tools)
    tools_added = sorted(tools_b - tools_a)
    tools_removed = sorted(tools_a - tools_b)

    config_fields = ("temperature", "model", "max_iterations", "memory_enabled", "cost_budget_usd")
    config_changes: dict[str, tuple[Any, Any]] = {}
    for field in config_fields:
        val_a = getattr(a.definition, field)
        val_b = getattr(b.definition, field)
        if val_a != val_b:
            config_changes[field] = (val_a, val_b)

    has_changes = bool(prompt_diff or tools_added or tools_removed or config_changes)

    return VersionDiff(
        version_id_a=a.version_id,
        version_id_b=b.version_id,
        system_prompt_diff=prompt_diff,
        tools_added=tools_added,
        tools_removed=tools_removed,
        config_changes=config_changes,
        has_changes=has_changes,
    )
