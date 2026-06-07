"""Validate the flagship deep-research-crew built-in template."""

from __future__ import annotations

import re
from pathlib import Path

from nexus.hub.installer import TemplateInstaller
from nexus.hub.manifest import TemplateManifest
from nexus.hub.registry import TemplateRegistry
from nexus.hub.renderer import collect_variables


def _templates_dir() -> Path:
    """Locate the bundled templates directory."""
    import nexus.hub.registry as reg_module

    pkg_dir = Path(reg_module.__file__).parent.parent
    return pkg_dir / "templates"


def _deep_research_dir() -> Path:
    return _templates_dir() / "deep-research-crew"


class TestDeepResearchTemplate:
    def test_deep_research_manifest_valid(self) -> None:
        manifest_path = _deep_research_dir() / "nexus-template.yaml"
        assert manifest_path.exists(), f"Manifest not found at {manifest_path}"
        manifest = TemplateManifest.from_yaml(manifest_path)
        assert manifest.name == "deep-research-crew"
        assert manifest.version
        assert manifest.description
        assert manifest.category == "research"
        assert len(manifest.parameters) >= 3

    def test_deep_research_all_files_present(self) -> None:
        manifest_path = _deep_research_dir() / "nexus-template.yaml"
        manifest = TemplateManifest.from_yaml(manifest_path)
        for fname in manifest.files:
            fpath = _deep_research_dir() / fname
            assert fpath.exists(), f"File declared in manifest not found: {fname}"

    def test_deep_research_no_undeclared_variables(self) -> None:
        manifest_path = _deep_research_dir() / "nexus-template.yaml"
        manifest = TemplateManifest.from_yaml(manifest_path)
        declared = {p.name for p in manifest.parameters}

        pattern = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")
        undeclared: list[str] = []
        for fname in manifest.files:
            fpath = _deep_research_dir() / fname
            if fpath.suffix in (".py", ".yaml", ".yml", ".md", ".txt"):
                content = fpath.read_text()
                for match in pattern.finditer(content):
                    var_name = match.group(1)
                    if var_name not in declared:
                        undeclared.append(f"{fname}: {{{{{var_name}}}}}")

        assert not undeclared, "Undeclared variables found:\n" + "\n".join(undeclared)

    def test_deep_research_renders_cleanly(self, tmp_path: Path) -> None:
        manifest_path = _deep_research_dir() / "nexus-template.yaml"
        manifest = TemplateManifest.from_yaml(manifest_path)
        variables = collect_variables(manifest, {})
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        result = installer.install_local(
            _deep_research_dir(), tmp_path / "output", variables=variables
        )
        assert result.template_name == "deep-research-crew"
        # No unrendered placeholders in output
        pattern = re.compile(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}")
        for fname in result.files_written:
            fpath = tmp_path / "output" / fname
            if fpath.suffix in (".py", ".yaml", ".yml", ".md"):
                content = fpath.read_text()
                remaining = pattern.findall(content)
                assert not remaining, f"Unrendered placeholders in {fname}: {remaining}"

    def test_deep_research_entry_point_exists_after_render(self, tmp_path: Path) -> None:
        manifest_path = _deep_research_dir() / "nexus-template.yaml"
        manifest = TemplateManifest.from_yaml(manifest_path)
        variables = collect_variables(manifest, {})
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        result = installer.install_local(
            _deep_research_dir(), tmp_path / "output", variables=variables
        )
        assert result.entry_point.exists()
        assert result.entry_point.name == manifest.entry_point
