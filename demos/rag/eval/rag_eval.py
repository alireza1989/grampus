"""RAGAS-style evaluation for the RAG pipeline.

Scores faithfulness (are answers grounded in retrieved context?) and
relevancy (does the answer address the question?) using LLM-as-judge.

Usage:
    python demos/rag/eval/rag_eval.py [--config rag_config.json] [--output results.json]
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import click

_SAMPLE_QA_PATH = Path(__file__).parent / "sample_qa.json"


def _build_faithfulness_prompt(context: str, answer: str) -> str:
    return f"""You are evaluating whether an AI answer is faithfully grounded in the provided context.

Context:
{context}

Answer:
{answer}

Rate faithfulness on a scale of 1-5:
1 = Answer contains claims not supported by context (hallucination)
3 = Answer is mostly grounded but has minor unsupported claims
5 = Every claim in the answer is directly supported by the context

Respond with ONLY: "Score: X/5" where X is your rating."""


def _build_relevancy_prompt(question: str, answer: str) -> str:
    return f"""You are evaluating whether an AI answer addresses the user's question.

Question: {question}

Answer: {answer}

Rate relevancy on a scale of 1-5:
1 = Answer does not address the question at all
3 = Answer partially addresses the question
5 = Answer completely and directly addresses the question

Respond with ONLY: "Score: X/5" where X is your rating."""


def _parse_score(text: str) -> float:
    """Extract numeric score from LLM judge response."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        return min(5.0, float(m.group(1)))
    return 3.0


class RAGEvaluator:
    """Run faithfulness and relevancy evaluation over a Q&A set."""

    def __init__(self, model_client: Any, rag_config: Any) -> None:
        self._model = model_client
        self._config = rag_config

    async def evaluate_pair(self, question: str, answer: str, context: str) -> dict[str, float]:
        """Score one Q&A pair. Returns faithfulness and relevancy scores (1-5)."""
        from nexus.core.types import Message, Role

        faith_prompt = _build_faithfulness_prompt(context, answer)
        rel_prompt = _build_relevancy_prompt(question, answer)

        faith_msgs = [Message(role=Role.USER, content=faith_prompt)]
        rel_msgs = [Message(role=Role.USER, content=rel_prompt)]

        faith_resp = await self._model.complete(
            messages=faith_msgs,
            model=self._config.model_id,
            max_tokens=20,
            temperature=0.0,
        )
        rel_resp = await self._model.complete(
            messages=rel_msgs,
            model=self._config.model_id,
            max_tokens=20,
            temperature=0.0,
        )

        return {
            "faithfulness": _parse_score(faith_resp.content or ""),
            "relevancy": _parse_score(rel_resp.content or ""),
        }

    async def run(self, qa_path: Path = _SAMPLE_QA_PATH) -> dict[str, Any]:
        """Evaluate all QA pairs. Returns summary dict with per-question scores."""
        from demos.rag.ingest import _build_embedding_service
        from demos.rag.rag_agent import run_agent
        from demos.rag.rag_store import RAGStore
        from demos.rag.rag_tool import make_retrieve_tool

        qa_pairs = json.loads(qa_path.read_text())
        results = []

        embedding_service = _build_embedding_service(self._config)
        store = await RAGStore.create(self._config.db_url, dimensions=embedding_service.dimensions)
        _, _ = make_retrieve_tool(store, embedding_service, self._config)

        try:
            for item in qa_pairs:
                question = item["question"]
                query_emb = await embedding_service.embed(question, input_type="search_query")
                chunks = await store.retrieve(
                    query_emb,
                    question,
                    namespace=self._config.namespace,
                    top_k=self._config.top_k,
                    rrf_k=self._config.rrf_k,
                    limit=self._config.max_context_chunks,
                )
                context = "\n\n".join(c.content for c in chunks)

                answer = await run_agent(question, self._config, stream=False)

                scores = await self.evaluate_pair(question, answer, context)
                results.append(
                    {
                        "id": item["id"],
                        "question": question,
                        "answer": answer,
                        "context_chunks": len(chunks),
                        **scores,
                    }
                )
        finally:
            await store.close()

        avg_faith = sum(r["faithfulness"] for r in results) / len(results) if results else 0.0
        avg_rel = sum(r["relevancy"] for r in results) / len(results) if results else 0.0
        passed = sum(1 for r in results if r["faithfulness"] >= 4 and r["relevancy"] >= 4)

        return {
            "total": len(results),
            "passed": passed,
            "avg_faithfulness": round(avg_faith, 2),
            "avg_relevancy": round(avg_rel, 2),
            "results": results,
        }


@click.command()
@click.option("--config", "config_path", default=None, type=click.Path())
@click.option("--output", default=None, type=click.Path(), help="Write results JSON to this file.")
@click.option(
    "--qa-file",
    default=None,
    type=click.Path(exists=True),
    help="Custom Q&A JSON file.",
)
def evaluate(config_path: str | None, output: str | None, qa_file: str | None) -> None:
    """Evaluate the RAG pipeline with faithfulness and relevancy scoring."""
    from demos.rag.config import RAGConfig
    from nexus.core.models.anthropic import AnthropicClient

    config = RAGConfig.from_file(config_path) if config_path else RAGConfig.from_env()
    api_key = config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    model_client = AnthropicClient(api_key=api_key)
    evaluator = RAGEvaluator(model_client, config)

    qa_path = Path(qa_file) if qa_file else _SAMPLE_QA_PATH
    summary = asyncio.run(evaluator.run(qa_path))

    click.echo("\nRAG Evaluation Results")
    click.echo("=" * 40)
    click.echo(f"Total questions:    {summary['total']}")
    click.echo(f"Passed (>=4/5 both): {summary['passed']}")
    click.echo(f"Avg faithfulness:   {summary['avg_faithfulness']}/5")
    click.echo(f"Avg relevancy:      {summary['avg_relevancy']}/5")
    click.echo()
    for r in summary["results"]:
        status = "+" if r["faithfulness"] >= 4 and r["relevancy"] >= 4 else "-"
        click.echo(
            f"  [{status}] [{r['id']}] F={r['faithfulness']}/5 "
            f"R={r['relevancy']}/5 -- {r['question'][:60]}"
        )

    if output:
        Path(output).write_text(json.dumps(summary, indent=2))
        click.echo(f"\nResults written to {output}")

    sys.exit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    evaluate()
