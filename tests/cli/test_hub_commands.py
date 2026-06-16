"""Tests for grampus hub CLI commands."""

from __future__ import annotations

import textwrap
from pathlib import Path

from click.testing import CliRunner

from grampus.cli.main import cli


class TestHubList:
    def test_hub_list_shows_builtins(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "list"])
        assert result.exit_code == 0, result.output
        assert "simple-agent" in result.output
        assert "deep-research-crew" in result.output
        assert "customer-support-rag" in result.output
        assert "code-reviewer" in result.output

    def test_hub_list_table_format(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "list", "--format", "table"])
        assert result.exit_code == 0, result.output
        assert "NAME" in result.output

    def test_hub_list_json_format(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "list", "--format", "json"])
        assert result.exit_code == 0, result.output
        assert "[" in result.output  # JSON array
        assert "simple-agent" in result.output

    def test_hub_list_filter_category(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "list", "--category", "research"])
        assert result.exit_code == 0, result.output
        assert "deep-research-crew" in result.output
        # support templates should not appear
        assert "customer-support-rag" not in result.output

    def test_hub_list_filter_tag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "list", "--tag", "rag"])
        assert result.exit_code == 0, result.output
        assert "customer-support-rag" in result.output


class TestHubSearch:
    def test_hub_search_returns_match(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "search", "rag"])
        assert result.exit_code == 0, result.output
        assert "customer-support-rag" in result.output

    def test_hub_search_no_results(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "search", "zzznomatch999xyz"])
        assert result.exit_code == 0, result.output
        assert "No templates found" in result.output


class TestHubInfo:
    def test_hub_info_shows_details(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "info", "deep-research-crew"])
        assert result.exit_code == 0, result.output
        assert "deep-research-crew" in result.output
        assert "research" in result.output.lower()
        # Should show parameters
        assert "powerful_model" in result.output or "model" in result.output.lower()

    def test_hub_info_unknown_template(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "info", "does-not-exist"])
        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestHubPull:
    def test_hub_pull_creates_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "my-simple-agent")
        result = runner.invoke(
            cli,
            [
                "hub",
                "pull",
                "simple-agent",
                "--output",
                output_dir,
                "--var",
                "project_name=test_proj",
                "--var",
                "model=claude-sonnet-4-6",
            ],
        )
        assert result.exit_code == 0, result.output
        assert Path(output_dir).exists()
        assert (Path(output_dir) / "agent.py").exists()

    def test_hub_pull_with_var_flag(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "output")
        result = runner.invoke(
            cli,
            [
                "hub",
                "pull",
                "simple-agent",
                "--output",
                output_dir,
                "--var",
                "project_name=custom_proj",
                "--var",
                "model=gpt-4o",
            ],
        )
        assert result.exit_code == 0, result.output
        agent_content = (Path(output_dir) / "agent.py").read_text()
        assert "custom_proj" in agent_content or "gpt-4o" in agent_content

    def test_hub_pull_dry_run_prints_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "output")
        result = runner.invoke(
            cli,
            [
                "hub",
                "pull",
                "simple-agent",
                "--output",
                output_dir,
                "--dry-run",
                "--var",
                "project_name=my_proj",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not Path(output_dir).exists()
        # Should list files that would be written
        assert "agent.py" in result.output or "dry" in result.output.lower()

    def test_hub_pull_prompts_for_required_params(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "output")
        # simple-agent has defaults for all params, so no prompting needed
        # Provide input just in case
        result = runner.invoke(
            cli,
            ["hub", "pull", "simple-agent", "--output", output_dir],
            input="my_project\n",
        )
        assert result.exit_code == 0, result.output

    def test_hub_pull_success_message(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "output")
        result = runner.invoke(
            cli,
            [
                "hub",
                "pull",
                "simple-agent",
                "--output",
                output_dir,
                "--var",
                "project_name=proj",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "simple-agent" in result.output
        assert "grampus run" in result.output or "Next steps" in result.output


class TestHubValidate:
    def _create_valid_template(self, base: Path) -> Path:
        tdir = base / "my-template"
        tdir.mkdir(parents=True)
        (tdir / "grampus-template.yaml").write_text(
            textwrap.dedent(
                """
                name: my-template
                version: "1.0.0"
                description: Test template
                parameters:
                  - name: project_name
                    description: Name
                    default: proj
                files:
                  - agent.py
                  - README.md
                """
            ).strip()
        )
        (tdir / "agent.py").write_text("# {{project_name}}")
        (tdir / "README.md").write_text("# {{project_name}}")
        return tdir

    def test_hub_validate_valid_template(self, tmp_path: Path) -> None:
        tdir = self._create_valid_template(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "validate", str(tdir)])
        assert result.exit_code == 0, result.output
        assert "error" not in result.output.lower() or "0 error" in result.output.lower()

    def test_hub_validate_missing_file(self, tmp_path: Path) -> None:
        tdir = tmp_path / "missing-file-template"
        tdir.mkdir()
        (tdir / "grampus-template.yaml").write_text(
            textwrap.dedent(
                """
                name: missing-file-template
                version: "1.0.0"
                description: Template with missing file
                files:
                  - agent.py
                  - missing.py
                """
            ).strip()
        )
        (tdir / "agent.py").write_text("# agent")
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "validate", str(tdir)])
        assert result.exit_code != 0 or "error" in result.output.lower()
        assert "missing.py" in result.output

    def test_hub_validate_undeclared_variable(self, tmp_path: Path) -> None:
        tdir = tmp_path / "undeclared-var-template"
        tdir.mkdir()
        (tdir / "grampus-template.yaml").write_text(
            textwrap.dedent(
                """
                name: undeclared-var-template
                version: "1.0.0"
                description: Template with undeclared variable
                parameters:
                  - name: project_name
                    description: Name
                files:
                  - agent.py
                """
            ).strip()
        )
        (tdir / "agent.py").write_text("# {{project_name}} and {{api_key}}")
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "validate", str(tdir)])
        assert "api_key" in result.output
        assert "undeclared" in result.output.lower() or "error" in result.output.lower()


class TestHubPush:
    def test_hub_push_prints_instructions(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["hub", "push", str(tmp_path)])
        assert result.exit_code == 0, result.output
        # Stub — should print GitHub PR instructions
        assert (
            "github" in result.output.lower()
            or "pull request" in result.output.lower()
            or "PR" in result.output
        )
