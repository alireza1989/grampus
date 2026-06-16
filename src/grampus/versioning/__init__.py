"""Agent versioning — lifecycle management, A/B testing, deployment, rollback."""

from grampus.versioning.ab_testing import ABTestManager
from grampus.versioning.manager import VersionManager
from grampus.versioning.metrics import QualityTracker, VersionMetrics
from grampus.versioning.router import VersionRouter
from grampus.versioning.stats import two_proportion_z_test, welch_t_test
from grampus.versioning.store import VersionStore
from grampus.versioning.types import (
    ABTestConfig,
    ABTestResult,
    AgentVersion,
    DeploymentRecord,
    SuccessMetric,
    VersionDiff,
    VersionStatus,
    compute_version_id,
    diff_versions,
)

__all__ = [
    "ABTestConfig",
    "ABTestManager",
    "ABTestResult",
    "AgentVersion",
    "DeploymentRecord",
    "QualityTracker",
    "SuccessMetric",
    "VersionDiff",
    "VersionManager",
    "VersionMetrics",
    "VersionRouter",
    "VersionStatus",
    "VersionStore",
    "compute_version_id",
    "diff_versions",
    "two_proportion_z_test",
    "welch_t_test",
]
