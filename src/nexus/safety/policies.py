"""YAML policy loader for the safety pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from nexus.core.errors import ConfigError
from nexus.core.logging import get_logger
from nexus.safety.action_guard import ActionPolicy, SafetyActionGuard
from nexus.safety.injection import DetectionLevel, PromptInjectionDetector
from nexus.safety.pii import PIIAction, PIIDetector, PIIType
from nexus.safety.pipeline import SafetyPipeline, SafetyPipelineConfig

_log = get_logger(__name__)


class NexusSafetyPolicy(BaseModel):
    """Top-level policy document loaded from YAML."""

    injection_detection_level: DetectionLevel = DetectionLevel.BALANCED
    pii_actions: dict[str, str] = Field(default_factory=dict)
    agent_policies: list[ActionPolicy] = Field(default_factory=list)
    pipeline_config: SafetyPipelineConfig = Field(default_factory=SafetyPipelineConfig)


class PolicyLoader:
    """Loads NexusSafetyPolicy from a YAML file or dict.

    Args:
        path: Path to YAML file. If None, returns default policy.
    """

    @staticmethod
    def load(path: str | None = None) -> NexusSafetyPolicy:
        """Load and validate policy. Returns default policy if path is None.

        Args:
            path: Filesystem path to a YAML policy file.

        Returns:
            A validated NexusSafetyPolicy instance.

        Raises:
            ConfigError: If the file is missing or contains invalid YAML/schema.
        """
        if path is None:
            return NexusSafetyPolicy()

        file_path = Path(path)
        if not file_path.exists():
            raise ConfigError(f"Policy file not found: {path}", code="POLICY_NOT_FOUND")

        try:
            import yaml  # type: ignore[import-untyped]

            raw = yaml.safe_load(file_path.read_text())
        except Exception as exc:
            raise ConfigError(
                f"Failed to parse policy YAML at {path}: {exc}",
                code="POLICY_PARSE_ERROR",
            ) from exc

        try:
            return NexusSafetyPolicy.model_validate(raw or {})
        except Exception as exc:
            raise ConfigError(
                f"Policy schema validation failed: {exc}",
                code="POLICY_SCHEMA_ERROR",
            ) from exc

    @staticmethod
    def build_pipeline(policy: NexusSafetyPolicy, *, agent_id: str) -> SafetyPipeline:
        """Construct a fully configured SafetyPipeline from a loaded policy.

        Args:
            policy: A validated NexusSafetyPolicy.
            agent_id: The agent this pipeline will guard.

        Returns:
            A ready-to-use SafetyPipeline.
        """
        injection_detector = PromptInjectionDetector(level=policy.injection_detection_level)
        pii_detector = _build_pii_detector(policy.pii_actions)
        action_guard = _find_agent_guard(policy.agent_policies, agent_id)

        return SafetyPipeline(
            injection_detector=injection_detector,
            pii_detector=pii_detector,
            action_guard=action_guard,
            config=policy.pipeline_config,
        )


def load_safety_policy(path: str) -> NexusSafetyPolicy:
    """Convenience wrapper: load a YAML policy and return NexusSafetyPolicy.

    Accepts both the full NexusSafetyPolicy field names AND the shorthand
    structure used in quickstart YAML files::

        injection:
          level: strict
        pii:
          action: redact
        action_guard:
          denied_tools: [rm]
          max_tool_calls_per_turn: 5

    Args:
        path: Filesystem path to the YAML policy file.

    Returns:
        Validated NexusSafetyPolicy.
    """
    from pathlib import Path

    import yaml

    raw: dict[str, object] = yaml.safe_load(Path(path).read_text()) or {}

    # Translate shorthand structure → NexusSafetyPolicy field names
    normalised: dict[str, object] = {}

    if "injection" in raw:
        inj = raw["injection"]
        if isinstance(inj, dict) and "level" in inj:
            normalised["injection_detection_level"] = inj["level"]

    if "pii" in raw:
        pii = raw["pii"]
        if isinstance(pii, dict) and "action" in pii:
            normalised["pii_actions"] = {t: pii["action"] for t in PIIType}

    if "action_guard" in raw:
        raw_ag = raw["action_guard"]
        if isinstance(raw_ag, dict):
            ag: dict[str, object] = dict(raw_ag)
            ag.setdefault("agent_id", "default")
            normalised["agent_policies"] = [ag]

    # Allow full NexusSafetyPolicy fields to pass through as-is
    for key in ("injection_detection_level", "pii_actions", "agent_policies", "pipeline_config"):
        if key in raw and key not in normalised:
            normalised[key] = raw[key]

    return NexusSafetyPolicy.model_validate(normalised)


def _build_pii_detector(pii_actions: dict[str, str]) -> PIIDetector:
    """Convert string-keyed pii_actions dict to PIIDetector."""
    actions: dict[PIIType, PIIAction] = {}
    for type_str, action_str in pii_actions.items():
        try:
            pii_type = PIIType(type_str)
            pii_action = PIIAction(action_str)
            actions[pii_type] = pii_action
        except ValueError:
            _log.warning("policy.unknown_pii_entry", key=type_str, value=action_str)

    if not actions:
        return PIIDetector()
    return PIIDetector(actions=actions)


def _find_agent_guard(
    agent_policies: list[ActionPolicy], agent_id: str
) -> SafetyActionGuard | None:
    """Find and build a guard for agent_id, or return None."""
    for ap in agent_policies:
        if ap.agent_id == agent_id:
            return SafetyActionGuard(ap)
    return None
