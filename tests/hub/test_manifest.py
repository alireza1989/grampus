"""Tests for TemplateManifest and TemplateParameter."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from grampus.hub.manifest import TemplateManifest, TemplateParameter


class TestTemplateParameter:
    def test_required_defaults_false(self) -> None:
        p = TemplateParameter(name="x", description="desc")
        assert p.required is False

    def test_choices_none_by_default(self) -> None:
        p = TemplateParameter(name="x", description="desc")
        assert p.choices is None

    def test_with_choices(self) -> None:
        p = TemplateParameter(name="x", description="desc", choices=["a", "b"])
        assert p.choices == ["a", "b"]


class TestTemplateManifest:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "grampus-template.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    def test_manifest_from_yaml_valid(self, tmp_path: Path) -> None:
        yaml_path = self._write_yaml(
            tmp_path,
            """
            name: simple-agent
            version: "1.0.0"
            description: A simple agent starter
            parameters:
              - name: project_name
                description: Project name
                default: my_agent
            files:
              - agent.py
              - README.md
            """,
        )
        manifest = TemplateManifest.from_yaml(yaml_path)
        assert manifest.name == "simple-agent"
        assert manifest.version == "1.0.0"
        assert manifest.description == "A simple agent starter"
        assert len(manifest.parameters) == 1
        assert manifest.parameters[0].name == "project_name"
        assert manifest.files == ["agent.py", "README.md"]

    def test_manifest_from_yaml_missing_required_field_raises(self, tmp_path: Path) -> None:
        yaml_path = self._write_yaml(
            tmp_path,
            """
            version: "1.0.0"
            description: Missing name field
            """,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TemplateManifest.from_yaml(yaml_path)

    def test_manifest_from_yaml_defaults(self, tmp_path: Path) -> None:
        yaml_path = self._write_yaml(
            tmp_path,
            """
            name: test-template
            version: "0.1.0"
            description: Minimal template
            """,
        )
        manifest = TemplateManifest.from_yaml(yaml_path)
        assert manifest.category == "general"
        assert manifest.author == "community"
        assert manifest.tags == []
        assert manifest.parameters == []
        assert manifest.files == []
        assert manifest.entry_point == "agent.py"

    def test_manifest_get_parameter_found(self, tmp_path: Path) -> None:
        yaml_path = self._write_yaml(
            tmp_path,
            """
            name: test
            version: "1.0.0"
            description: test
            parameters:
              - name: model
                description: Model name
                default: claude-sonnet-4-6
            """,
        )
        manifest = TemplateManifest.from_yaml(yaml_path)
        param = manifest.get_parameter("model")
        assert param is not None
        assert param.name == "model"
        assert param.default == "claude-sonnet-4-6"

    def test_manifest_get_parameter_missing_returns_none(self, tmp_path: Path) -> None:
        yaml_path = self._write_yaml(
            tmp_path,
            """
            name: test
            version: "1.0.0"
            description: test
            """,
        )
        manifest = TemplateManifest.from_yaml(yaml_path)
        assert manifest.get_parameter("nonexistent") is None

    def test_manifest_always_safe_loads(self, tmp_path: Path) -> None:
        yaml_path = self._write_yaml(
            tmp_path,
            """
            name: test
            version: "1.0.0"
            description: test
            """,
        )
        # Verify yaml.safe_load is used (not bare yaml.load)
        with patch(
            "grampus.hub.manifest.yaml.safe_load", wraps=__import__("yaml").safe_load
        ) as mock_safe:
            TemplateManifest.from_yaml(yaml_path)
            mock_safe.assert_called_once()
