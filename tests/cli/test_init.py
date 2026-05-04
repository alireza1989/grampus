"""Tests for nexus init command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from nexus.cli.main import cli


class TestInitCommand:
    """Tests for the nexus init command."""

    def test_init_creates_project_directory(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--name", "my-agent", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "my-agent").is_dir()

    def test_init_creates_nexus_yaml(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "my-agent", "--output-dir", str(tmp_path)])
        assert (tmp_path / "my-agent" / "nexus.yaml").exists()

    def test_init_creates_agent_py(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "my-agent", "--output-dir", str(tmp_path)])
        assert (tmp_path / "my-agent" / "agent.py").exists()

    def test_init_creates_docker_compose(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "my-agent", "--output-dir", str(tmp_path)])
        assert (tmp_path / "my-agent" / "docker-compose.yml").exists()

    def test_init_creates_dapr_components(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "my-agent", "--output-dir", str(tmp_path)])
        dapr_dir = tmp_path / "my-agent" / "dapr" / "components"
        assert dapr_dir.is_dir()
        assert (dapr_dir / "statestore-postgres.yaml").exists()
        assert (dapr_dir / "statestore-redis.yaml").exists()
        assert (dapr_dir / "pubsub-redis.yaml").exists()

    def test_init_simple_template_default(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--name", "my-agent", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "my-agent" / "agent.py").exists()

    def test_init_crew_template_creates_crew_py(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["init", "--name", "my-agent", "--template", "crew", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "my-agent" / "crew.py").exists()

    def test_init_rag_template_creates_rag_tools_py(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["init", "--name", "my-agent", "--template", "rag", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "my-agent" / "rag_tools.py").exists()

    def test_init_custom_name_used_in_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "cool-bot", "--output-dir", str(tmp_path)])
        nexus_yaml = (tmp_path / "cool-bot" / "nexus.yaml").read_text()
        assert "cool-bot" in nexus_yaml

    def test_init_prompts_on_existing_non_empty_dir(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "my-agent"
        project_dir.mkdir()
        (project_dir / "existing.txt").write_text("existing content")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["init", "--name", "my-agent", "--output-dir", str(tmp_path)],
            input="n\n",
        )
        assert "Overwrite" in result.output or "overwrite" in result.output.lower()

    def test_init_overwrite_yes_proceeds(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "my-agent"
        project_dir.mkdir()
        (project_dir / "existing.txt").write_text("existing content")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["init", "--name", "my-agent", "--output-dir", str(tmp_path)],
            input="y\n",
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "my-agent" / "nexus.yaml").exists()

    def test_init_overwrite_no_aborts(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "my-agent"
        project_dir.mkdir()
        (project_dir / "existing.txt").write_text("existing content")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["init", "--name", "my-agent", "--output-dir", str(tmp_path)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert not (tmp_path / "my-agent" / "nexus.yaml").exists()

    def test_init_prints_created_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--name", "my-agent", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "nexus.yaml" in result.output
        assert "agent.py" in result.output
