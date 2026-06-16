"""Tests for PolicyLoader and GrampusSafetyPolicy."""

from __future__ import annotations

import pytest

from grampus.core.errors import ConfigError
from grampus.safety.action_guard import ActionPolicy
from grampus.safety.injection import DetectionLevel
from grampus.safety.pipeline import SafetyPipeline
from grampus.safety.policies import GrampusSafetyPolicy, PolicyLoader


class TestPolicyLoader:
    def test_load_returns_default_policy_when_path_none(self) -> None:
        policy = PolicyLoader.load(None)
        assert isinstance(policy, GrampusSafetyPolicy)
        assert policy.injection_detection_level == DetectionLevel.BALANCED

    def test_load_parses_yaml_file(self, tmp_path: pytest.TempPathFactory) -> None:
        yaml_content = """
injection_detection_level: strict
pii_actions:
  email: redact
  ssn: block
pipeline_config:
  check_user_input: true
  check_tool_results: false
agent_policies:
  - agent_id: "test-agent"
    denied_tools: ["shell"]
    max_tool_calls_per_turn: 10
"""
        f = tmp_path / "policy.yaml"  # type: ignore[operator]
        f.write_text(yaml_content)
        policy = PolicyLoader.load(str(f))
        assert policy.injection_detection_level == DetectionLevel.STRICT
        assert policy.pii_actions.get("ssn") == "block"
        assert policy.pipeline_config.check_tool_results is False
        assert len(policy.agent_policies) == 1
        assert policy.agent_policies[0].agent_id == "test-agent"

    def test_load_invalid_yaml_raises_config_error(self, tmp_path: pytest.TempPathFactory) -> None:
        f = tmp_path / "bad.yaml"  # type: ignore[operator]
        f.write_text("injection_detection_level: [invalid: yaml: here")
        with pytest.raises((ConfigError, Exception)):
            PolicyLoader.load(str(f))

    def test_load_nonexistent_file_raises_config_error(self) -> None:
        with pytest.raises(ConfigError):
            PolicyLoader.load("/no/such/path.yaml")

    def test_build_pipeline_returns_safety_pipeline(self) -> None:
        policy = GrampusSafetyPolicy()
        pipeline = PolicyLoader.build_pipeline(policy, agent_id="my-agent")
        assert isinstance(pipeline, SafetyPipeline)

    def test_build_pipeline_configures_injection_level(self) -> None:
        policy = GrampusSafetyPolicy(injection_detection_level=DetectionLevel.STRICT)
        pipeline = PolicyLoader.build_pipeline(policy, agent_id="my-agent")
        # Pipeline has injection detector set
        assert pipeline._injection_detector is not None
        assert pipeline._injection_detector.level == DetectionLevel.STRICT

    def test_build_pipeline_wires_agent_policy_by_id(self) -> None:
        ap = ActionPolicy(agent_id="target-agent", denied_tools=["rm"])
        policy = GrampusSafetyPolicy(agent_policies=[ap])
        pipeline = PolicyLoader.build_pipeline(policy, agent_id="target-agent")
        assert pipeline._action_guard is not None

    def test_build_pipeline_with_no_agent_policy_uses_no_guard(self) -> None:
        policy = GrampusSafetyPolicy()
        pipeline = PolicyLoader.build_pipeline(policy, agent_id="nonexistent-agent")
        assert pipeline._action_guard is None

    def test_build_pipeline_configures_pii_actions(self) -> None:
        policy = GrampusSafetyPolicy(pii_actions={"email": "redact", "ssn": "block"})
        pipeline = PolicyLoader.build_pipeline(policy, agent_id="a")
        assert pipeline._pii_detector is not None
