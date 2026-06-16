"""Tests for TemplateInstaller."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from grampus.hub.installer import TemplateInstaller
from grampus.hub.registry import TemplateRegistry


def _create_simple_template_dir(base: Path) -> Path:
    """Create a minimal template directory for testing."""
    tdir = base / "simple-agent"
    tdir.mkdir(parents=True)
    (tdir / "grampus-template.yaml").write_text(
        textwrap.dedent(
            """
            name: simple-agent
            version: "1.0.0"
            description: Simple agent starter
            parameters:
              - name: project_name
                description: Project name
                default: my_agent
              - name: model
                description: Model to use
                default: claude-sonnet-4-6
            files:
              - agent.py
              - README.md
            entry_point: agent.py
            """
        ).strip()
    )
    (tdir / "agent.py").write_text("# Agent for {{project_name}}\nmodel = '{{model}}'\n")
    (tdir / "README.md").write_text("# {{project_name}}\nUsing {{model}}.\n")
    return tdir


class TestInstallLocalTemplate:
    def test_install_local_template(self, tmp_path: Path) -> None:
        template_dir = _create_simple_template_dir(tmp_path / "templates")
        output_dir = tmp_path / "output"
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        result = installer.install_local(template_dir, output_dir)
        assert result.template_name == "simple-agent"
        assert (output_dir / "agent.py").exists()
        assert (output_dir / "README.md").exists()

    def test_install_renders_variables(self, tmp_path: Path) -> None:
        template_dir = _create_simple_template_dir(tmp_path / "templates")
        output_dir = tmp_path / "output"
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        installer.install_local(
            template_dir, output_dir, variables={"project_name": "my_proj", "model": "gpt-4o"}
        )
        content = (output_dir / "agent.py").read_text()
        assert "my_proj" in content
        assert "gpt-4o" in content
        assert "{{project_name}}" not in content

    def test_install_creates_output_dir(self, tmp_path: Path) -> None:
        template_dir = _create_simple_template_dir(tmp_path / "templates")
        output_dir = tmp_path / "new" / "nested" / "dir"
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        installer.install_local(template_dir, output_dir)
        assert output_dir.is_dir()

    def test_install_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        template_dir = _create_simple_template_dir(tmp_path / "templates")
        output_dir = tmp_path / "output"
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        result = installer.install_local(template_dir, output_dir, dry_run=True)
        assert not output_dir.exists()
        assert len(result.files_written) == 2

    def test_install_existing_dir_raises(self, tmp_path: Path) -> None:
        template_dir = _create_simple_template_dir(tmp_path / "templates")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "existing_file.txt").write_text("exists")
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        with pytest.raises(ValueError, match="already exists"):
            installer.install_local(template_dir, output_dir)

    def test_install_force_overwrites(self, tmp_path: Path) -> None:
        template_dir = _create_simple_template_dir(tmp_path / "templates")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "existing_file.txt").write_text("exists")
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry, force=True)
        result = installer.install_local(template_dir, output_dir)
        assert result.template_name == "simple-agent"

    def test_install_result_has_entry_point(self, tmp_path: Path) -> None:
        template_dir = _create_simple_template_dir(tmp_path / "templates")
        output_dir = tmp_path / "output"
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        result = installer.install_local(template_dir, output_dir)
        assert result.entry_point == output_dir / "agent.py"

    def test_install_required_param_missing_raises(self, tmp_path: Path) -> None:
        tdir = tmp_path / "templates" / "required-template"
        tdir.mkdir(parents=True)
        (tdir / "grampus-template.yaml").write_text(
            textwrap.dedent(
                """
                name: required-template
                version: "1.0.0"
                description: Template with required param
                parameters:
                  - name: must_provide
                    description: You must set this
                    required: true
                files:
                  - agent.py
                """
            ).strip()
        )
        (tdir / "agent.py").write_text("# {{must_provide}}")
        output_dir = tmp_path / "output"
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        with pytest.raises(ValueError, match="must_provide"):
            installer.install_local(tdir, output_dir)


class TestSecurityChecks:
    def test_install_skips_path_traversal_files(self, tmp_path: Path) -> None:
        tdir = tmp_path / "templates" / "attack-template"
        tdir.mkdir(parents=True)
        (tdir / "grampus-template.yaml").write_text(
            textwrap.dedent(
                """
                name: attack-template
                version: "1.0.0"
                description: Security test
                files:
                  - agent.py
                  - ../../../etc/passwd
                """
            ).strip()
        )
        (tdir / "agent.py").write_text("safe content")
        output_dir = tmp_path / "output"
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        result = installer.install_local(tdir, output_dir)
        # Only agent.py should be written; path traversal skipped
        assert "agent.py" in result.files_written
        assert "../../../etc/passwd" not in result.files_written
        assert not (output_dir.parent.parent.parent / "etc" / "passwd").exists()

    def test_install_skips_git_directory(self, tmp_path: Path) -> None:
        tdir = tmp_path / "templates" / "git-template"
        tdir.mkdir(parents=True)
        (tdir / "grampus-template.yaml").write_text(
            textwrap.dedent(
                """
                name: git-template
                version: "1.0.0"
                description: Git test
                files:
                  - agent.py
                  - .git/config
                """
            ).strip()
        )
        (tdir / "agent.py").write_text("safe content")
        git_dir = tdir / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")
        output_dir = tmp_path / "output"
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        result = installer.install_local(tdir, output_dir)
        assert ".git/config" not in result.files_written
        assert not (output_dir / ".git").exists()


class TestBuiltinInstall:
    def test_install_builtin_template(self, tmp_path: Path) -> None:
        registry = TemplateRegistry(_http_client=None)
        installer = TemplateInstaller(registry=registry)
        output_dir = tmp_path / "output"
        result = installer.install("simple-agent", output_dir)
        assert result.template_name == "simple-agent"
        assert (output_dir / "agent.py").exists()
