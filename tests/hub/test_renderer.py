"""Tests for template renderer (safe {{var}} substitution)."""

from __future__ import annotations

import pytest

from grampus.hub.manifest import TemplateManifest, TemplateParameter
from grampus.hub.renderer import collect_variables, render_content, render_filename


class TestRenderContent:
    def test_render_content_substitutes_variable(self) -> None:
        result = render_content("Hello {{name}}!", {"name": "world"})
        assert result == "Hello world!"

    def test_render_content_multiple_variables(self) -> None:
        result = render_content("{{a}} and {{b}}", {"a": "foo", "b": "bar"})
        assert result == "foo and bar"

    def test_render_content_unknown_variable_left_as_is(self) -> None:
        result = render_content("Hello {{unknown}}!", {"name": "world"})
        assert result == "Hello {{unknown}}!"

    def test_render_content_no_variables_unchanged(self) -> None:
        original = "No variables here, just plain text."
        assert render_content(original, {}) == original

    def test_render_content_does_not_execute_code(self) -> None:
        # Ensure Python expressions are not evaluated — treated as literal strings
        content = "{{__import__('os')}}"
        result = render_content(content, {})
        assert result == "{{__import__('os')}}"

    def test_render_content_does_not_execute_code_with_matching_key(self) -> None:
        # Even if someone passes the exact key, we never eval
        content = "{{__import__('os')}}"
        # The variable name is invalid per [a-zA-Z_][a-zA-Z0-9_]*, so no substitution
        result = render_content(content, {"__import__('os')": "injected"})
        assert result == "{{__import__('os')}}"

    def test_render_content_variable_repeated(self) -> None:
        result = render_content("{{x}} {{x}} {{x}}", {"x": "hi"})
        assert result == "hi hi hi"

    def test_render_content_partial_braces_unchanged(self) -> None:
        result = render_content("{single} and {{double}}", {"double": "ok"})
        assert result == "{single} and ok"


class TestRenderFilename:
    def test_render_filename_substitutes(self) -> None:
        assert (
            render_filename("{{project_name}}_agent.py", {"project_name": "demo"})
            == "demo_agent.py"
        )

    def test_render_filename_no_variables(self) -> None:
        assert render_filename("agent.py", {}) == "agent.py"

    def test_render_filename_unknown_left_as_is(self) -> None:
        assert render_filename("{{unknown}}.py", {}) == "{{unknown}}.py"


class TestCollectVariables:
    def _make_manifest(self, params: list[TemplateParameter]) -> TemplateManifest:
        return TemplateManifest(
            name="test",
            version="1.0.0",
            description="test",
            parameters=params,
        )

    def test_collect_variables_uses_defaults(self) -> None:
        manifest = self._make_manifest(
            [TemplateParameter(name="model", description="model", default="claude-sonnet-4-6")]
        )
        result = collect_variables(manifest, {})
        assert result["model"] == "claude-sonnet-4-6"

    def test_collect_variables_overrides_take_precedence(self) -> None:
        manifest = self._make_manifest(
            [TemplateParameter(name="model", description="model", default="claude-sonnet-4-6")]
        )
        result = collect_variables(manifest, {"model": "gpt-4o"})
        assert result["model"] == "gpt-4o"

    def test_collect_variables_required_missing_raises(self) -> None:
        manifest = self._make_manifest(
            [TemplateParameter(name="project_name", description="project name", required=True)]
        )
        with pytest.raises(ValueError, match="project_name"):
            collect_variables(manifest, {})

    def test_collect_variables_required_with_override_ok(self) -> None:
        manifest = self._make_manifest(
            [TemplateParameter(name="project_name", description="project name", required=True)]
        )
        result = collect_variables(manifest, {"project_name": "my_project"})
        assert result["project_name"] == "my_project"

    def test_collect_variables_choices_validated(self) -> None:
        manifest = self._make_manifest(
            [
                TemplateParameter(
                    name="tier",
                    description="model tier",
                    choices=["fast", "balanced", "powerful"],
                    required=True,
                )
            ]
        )
        with pytest.raises(ValueError, match="tier"):
            collect_variables(manifest, {"tier": "invalid_choice"})

    def test_collect_variables_choices_valid_passes(self) -> None:
        manifest = self._make_manifest(
            [
                TemplateParameter(
                    name="tier",
                    description="model tier",
                    choices=["fast", "balanced", "powerful"],
                )
            ]
        )
        result = collect_variables(manifest, {"tier": "fast"})
        assert result["tier"] == "fast"

    def test_collect_variables_optional_no_default_not_included(self) -> None:
        manifest = self._make_manifest(
            [TemplateParameter(name="optional_param", description="optional", required=False)]
        )
        result = collect_variables(manifest, {})
        # Optional with no default and no override — not included
        assert "optional_param" not in result
