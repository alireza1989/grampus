"""nexus init — scaffold a new Nexus agent project."""

from __future__ import annotations

from pathlib import Path

import click

from nexus.core.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Embedded templates (no external template files)
# ---------------------------------------------------------------------------

_NEXUS_YAML = """\
agent:
  name: "{name}-agent"
  model: "claude-sonnet-4-6"
  system_prompt: "You are a helpful assistant."
  max_iterations: 10
  memory_enabled: true
  cost_budget_usd: 1.0

dapr:
  app_id: "{name}"
  app_port: 8000

memory:
  working_window_tokens: 4000
  episodic_enabled: true
  semantic_enabled: true
"""

_AGENT_PY = """\
\"\"\"Simple Nexus agent entrypoint.\"\"\"
from nexus.core.types import AgentDefinition
from nexus.orchestration.runner import AgentRunner, RunnerConfig


def create_agent_def() -> AgentDefinition:
    \"\"\"Return the agent definition loaded from nexus.yaml.\"\"\"
    return AgentDefinition(
        name="{name}-agent",
        model="claude-sonnet-4-6",
        system_prompt="You are a helpful assistant.",
        max_iterations=10,
        memory_enabled=True,
        cost_budget_usd=1.0,
    )


def create_runner() -> AgentRunner:
    \"\"\"Wire up and return an AgentRunner.\"\"\"
    # TODO: inject model_client, tool_executor, memory_manager, etc.
    raise NotImplementedError("Implement create_runner() to wire up your agent.")
"""

_CREW_PY = """\
\"\"\"Multi-agent crew example.\"\"\"
from nexus.core.types import AgentDefinition
from nexus.orchestration.crew import Crew


def create_crew() -> Crew:
    \"\"\"Create a 3-agent crew for collaborative task execution.\"\"\"
    researcher = AgentDefinition(
        name="researcher",
        model="claude-sonnet-4-6",
        system_prompt="You are a research specialist.",
    )
    writer = AgentDefinition(
        name="writer",
        model="claude-sonnet-4-6",
        system_prompt="You are a technical writer.",
    )
    reviewer = AgentDefinition(
        name="reviewer",
        model="claude-sonnet-4-6",
        system_prompt="You are a quality reviewer.",
    )
    # TODO: inject runners and wire up the crew
    raise NotImplementedError("Wire up runners before using this crew.")
"""

_RAG_TOOLS_PY = """\
\"\"\"RAG (Retrieval-Augmented Generation) tool stubs.\"\"\"
from nexus.tools.registry import ToolRegistry

registry = ToolRegistry()


@registry.tool(name="retrieve_documents", description="Retrieve relevant documents for a query.")
async def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    \"\"\"Retrieve documents from the vector store.

    Args:
        query: Search query.
        top_k: Number of documents to return.

    Returns:
        List of document dicts with 'content' and 'source' keys.
    \"\"\"
    # TODO: implement vector search against your document store
    raise NotImplementedError("Implement document retrieval logic.")
"""

_DOCKER_COMPOSE = """\
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: nexus
      POSTGRES_PASSWORD: nexus
      POSTGRES_DB: nexus
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  placement:
    image: daprio/dapr:1.13.0
    command: ["./placement", "--port", "50006"]
    ports:
      - "50006:50006"

volumes:
  postgres_data:
"""

_DAPR_CONFIG = """\
apiVersion: dapr.io/v1alpha1
kind: Configuration
metadata:
  name: nexus-config
spec:
  tracing:
    samplingRate: "1"
    zipkin:
      endpointAddress: "http://localhost:9411/api/v2/spans"
"""

_STATESTORE_POSTGRES = """\
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: statestore
spec:
  type: state.postgresql
  version: v1
  metadata:
    - name: connectionString
      value: "host=localhost user=nexus password=nexus dbname=nexus port=5432 sslmode=disable"
    - name: schema
      value: "public"
    - name: tableName
      value: "nexus_state"
"""

_STATESTORE_REDIS = """\
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: cache
spec:
  type: state.redis
  version: v1
  metadata:
    - name: redisHost
      value: "localhost:6379"
    - name: redisPassword
      value: ""
"""

_PUBSUB_REDIS = """\
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: pubsub
spec:
  type: pubsub.redis
  version: v1
  metadata:
    - name: redisHost
      value: "localhost:6379"
    - name: redisPassword
      value: ""
"""

_ENV_EXAMPLE = """\
# Copy to .env and fill in your API keys
NEXUS_MODEL__ANTHROPIC_API_KEY=your-anthropic-api-key
NEXUS_MODEL__OPENAI_API_KEY=your-openai-api-key
NEXUS_MODEL__DEFAULT_MODEL=claude-sonnet-4-6
"""

_PYPROJECT_TOML = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.1.0"
description = "A Nexus agent project"
requires-python = ">=3.12"

dependencies = [
    "nexus-ai>=0.1.0",
]
"""

_README = """\
# {name}

A Nexus agent project.

## Quick start

```bash
uv sync
cp .env.example .env  # add your API keys
docker compose up -d  # start PostgreSQL, Redis, Dapr
nexus run agent.py --input "Hello, agent!"
```

## Running evals

```bash
nexus eval eval_suite.py
```
"""

# ---------------------------------------------------------------------------
# File manifest
# ---------------------------------------------------------------------------

_SIMPLE_FILES: list[tuple[str, str]] = [
    ("nexus.yaml", _NEXUS_YAML),
    ("agent.py", _AGENT_PY),
    ("docker-compose.yml", _DOCKER_COMPOSE),
    ("dapr/config.yaml", _DAPR_CONFIG),
    ("dapr/components/statestore-postgres.yaml", _STATESTORE_POSTGRES),
    ("dapr/components/statestore-redis.yaml", _STATESTORE_REDIS),
    ("dapr/components/pubsub-redis.yaml", _PUBSUB_REDIS),
    (".env.example", _ENV_EXAMPLE),
    ("pyproject.toml", _PYPROJECT_TOML),
    ("README.md", _README),
]

_CREW_EXTRA: list[tuple[str, str]] = [("crew.py", _CREW_PY)]
_RAG_EXTRA: list[tuple[str, str]] = [("rag_tools.py", _RAG_TOOLS_PY)]


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command("init")
@click.option("--name", default="nexus-agent", show_default=True, help="Project name.")
@click.option(
    "--template",
    type=click.Choice(["simple", "crew", "rag"]),
    default="simple",
    show_default=True,
    help="Project template.",
)
@click.option(
    "--output-dir",
    default=".",
    show_default=True,
    help="Parent directory for the new project.",
)
def init(name: str, template: str, output_dir: str) -> None:
    """Scaffold a new Nexus agent project."""
    project_dir = Path(output_dir) / name

    if not _confirm_overwrite(project_dir):
        return

    files = _collect_files(name, template)
    _write_files(project_dir, files, name)
    _print_summary(name, project_dir, files)


def _confirm_overwrite(project_dir: Path) -> bool:
    """Return True if it is safe to proceed (dir empty or user confirmed)."""
    if project_dir.exists() and any(project_dir.iterdir()):
        return click.confirm(
            f"Directory '{project_dir}' already exists and is not empty. Overwrite?",
            default=False,
        )
    return True


def _collect_files(name: str, template: str) -> list[tuple[str, str]]:
    """Build the list of (relative_path, content) pairs for the chosen template."""
    files = list(_SIMPLE_FILES)
    if template == "crew":
        files.extend(_CREW_EXTRA)
    elif template == "rag":
        files.extend(_RAG_EXTRA)
    return files


def _write_files(project_dir: Path, files: list[tuple[str, str]], name: str) -> None:
    """Create all template files under *project_dir*."""
    for rel_path, content in files:
        target = project_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.format(name=name))


def _print_summary(name: str, project_dir: Path, files: list[tuple[str, str]]) -> None:
    """Print a success message listing every created file."""
    click.echo(f"\nCreated project '{name}' in {project_dir}/\n")
    for rel_path, _ in files:
        click.echo(f"  {rel_path}")
    click.echo(
        f"\nNext steps:\n  cd {project_dir.name}\n  cp .env.example .env\n  docker compose up -d\n"
    )
