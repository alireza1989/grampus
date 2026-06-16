"""RAG Q&A agent — interactive or single-shot retrieval-augmented generation."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import click

from demos.rag.config import RAGConfig
from demos.rag.ingest import _build_embedding_service
from demos.rag.rag_store import RAGStore
from demos.rag.rag_tool import make_retrieve_tool
from grampus.core.logging import get_logger
from grampus.core.types import AgentDefinition

_log = get_logger(__name__)

_SYSTEM_PROMPT = """You are a knowledgeable assistant with access to a document knowledge base.

When answering questions:
1. ALWAYS call retrieve_context first to search for relevant information.
2. Base your answer on the retrieved context, not prior knowledge.
3. Cite sources using [1], [2] notation — each number maps to a retrieved chunk.
4. If the retrieved context does not contain enough information to answer confidently,
   say so clearly rather than guessing.
5. Keep answers concise and factual.

The user is asking about documents that have been indexed in this knowledge base."""


async def run_agent(question: str, config: RAGConfig, *, stream: bool = True) -> str:
    """Run one question through the RAG pipeline and return the answer."""
    from grampus.core.models.anthropic import AnthropicClient
    from grampus.orchestration.runner import AgentRunner
    from grampus.tools.executor import ToolExecutor

    embedding_service = _build_embedding_service(config)
    store = await RAGStore.create(config.db_url, dimensions=embedding_service.dimensions)

    try:
        rag_registry, _ = make_retrieve_tool(store, embedding_service, config)

        agent_def = AgentDefinition(
            name="rag-agent",
            model=config.model_id,
            system_prompt=_SYSTEM_PROMPT,
            tools=["retrieve_context"],
            max_iterations=5,
            temperature=config.temperature,
            memory_enabled=False,
            cost_budget_usd=None,
        )

        api_key = config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        model_client = AnthropicClient(api_key=api_key)
        tool_executor = ToolExecutor(rag_registry)

        runner = AgentRunner(model_client, tool_executor)

        result = await runner.run(
            agent_def,
            question,
            session_id=str(uuid.uuid4()),
        )
        return result.output or "(no output)"
    finally:
        await store.close()


@click.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(),
    help="RAGConfig JSON/YAML path.",
)
@click.option("--namespace", default=None, help="Namespace to query (overrides config).")
@click.option("--query", "-q", default=None, help="Single question (non-interactive mode).")
@click.option(
    "--interactive/--no-interactive",
    "-i",
    default=True,
    help="Interactive Q&A loop.",
)
@click.option("--stream/--no-stream", default=True, help="Stream tokens as generated.")
def main(
    config_path: str | None,
    namespace: str | None,
    query: str | None,
    interactive: bool,
    stream: bool,
) -> None:
    """Ask questions about your ingested documents."""
    config = RAGConfig.from_file(config_path) if config_path else RAGConfig.from_env()
    if namespace:
        config = config.model_copy(update={"namespace": namespace})

    if query:
        answer = asyncio.run(run_agent(query, config, stream=stream))
        click.echo(answer)
        return

    if interactive:
        click.echo(f"RAG Agent ready. Namespace: '{config.namespace}'. Type 'exit' to quit.\n")
        while True:
            try:
                question = click.prompt("You")
            except (click.Abort, EOFError):
                break
            if question.lower() in ("exit", "quit", "q"):
                break
            click.echo("Agent: ", nl=False)
            answer = asyncio.run(run_agent(question, config, stream=stream))
            click.echo(answer)
            click.echo()
    else:
        click.echo("Provide --query or use --interactive.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
