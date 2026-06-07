"""Deep Research Crew — multi-agent research pipeline.

Workflow:
  Planner -> [Search A + Search B] (parallel) -> Fact Checker
  -> Synthesizer -> Critic -> (loop back if gaps, max {{max_search_rounds}} rounds)
  -> Writer -> final report
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml

from nexus.core.config import NexusConfig
from nexus.core.types import AgentDefinition
from nexus.orchestration.runner import AgentRunner


def load_agents(config_path: Path) -> dict:  # type: ignore[type-arg]
    return yaml.safe_load(config_path.read_text())  # type: ignore[no-any-return]


def make_runner(agents_cfg: dict, role: str) -> AgentRunner:  # type: ignore[type-arg]
    cfg = agents_cfg[role]
    return AgentRunner(
        agent_def=AgentDefinition(
            name=cfg["name"],
            model=cfg["model"],
            system_prompt=cfg["system_prompt"],
            max_iterations=cfg.get("max_iterations", 5),
            temperature=cfg.get("temperature", 0.2),
            memory_enabled=False,
        ),
    )


async def run_research(question: str, config: NexusConfig) -> str:  # noqa: ARG001
    agents_cfg = load_agents(Path(__file__).parent / "agents.yaml")

    planner = make_runner(agents_cfg, "planner")
    searcher_a = make_runner(agents_cfg, "searcher")
    searcher_b = make_runner(agents_cfg, "searcher")
    fact_checker = make_runner(agents_cfg, "fact_checker")
    synthesizer = make_runner(agents_cfg, "synthesizer")
    critic = make_runner(agents_cfg, "critic")
    writer = make_runner(agents_cfg, "writer")

    # Step 1: Plan sub-queries
    plan_result = await planner.run(f"Research question: {question}\n\nDecompose into sub-queries.")
    try:
        sub_queries = json.loads(plan_result.output)
    except json.JSONDecodeError:
        sub_queries = [question]
    print(f"[Planner] Generated {len(sub_queries)} sub-queries")

    all_findings: list[str] = []
    max_rounds = int("{{max_search_rounds}}")
    final_synthesis = ""

    for round_num in range(1, max_rounds + 1):
        print(f"\n[Round {round_num}] Searching {len(sub_queries)} queries in parallel...")

        mid = len(sub_queries) // 2 or len(sub_queries)
        queries_a = sub_queries[:mid]
        queries_b = sub_queries[mid:]

        async def search_batch(runner: AgentRunner, queries: list[str]) -> str:
            results = []
            for q in queries:
                r = await runner.run(f"Search query: {q}\nMax sources: {{max_sources_per_query}}")
                results.append(f"Query: {q}\n{r.output}")
            return "\n\n---\n\n".join(results)

        if queries_b:
            searches = await asyncio.gather(
                search_batch(searcher_a, queries_a),
                search_batch(searcher_b, queries_b),
            )
        else:
            searches = [await search_batch(searcher_a, queries_a)]

        raw_findings = "\n\n===\n\n".join(str(s) for s in searches if s)

        # Step 3: Fact check
        print("[Fact Checker] Validating sources...")
        fc_result = await fact_checker.run(
            f"Original question: {question}\n\nSearch results to fact-check:\n{raw_findings}"
        )
        all_findings.append(fc_result.output)

        # Step 4: Synthesize
        print("[Synthesizer] Building synthesis...")
        synth_result = await synthesizer.run(
            f"Original question: {question}\n\nVerified findings:\n{chr(10).join(all_findings)}"
        )

        # Step 5: Critique
        print("[Critic] Reviewing synthesis...")
        crit_result = await critic.run(
            f"Original question: {question}\n\nCurrent synthesis:\n{synth_result.output}"
        )

        try:
            critique = json.loads(crit_result.output)
        except json.JSONDecodeError:
            critique = {"needs_refinement": False, "additional_queries": []}

        final_synthesis = synth_result.output

        if not critique.get("needs_refinement", False) or round_num == max_rounds:
            print(f"[Critic] Synthesis accepted after round {round_num}")
            break

        sub_queries = critique.get("additional_queries", [])[:5]
        print(f"[Critic] Found gaps, adding {len(sub_queries)} queries for round {round_num + 1}")

    # Step 6: Write final report
    print("\n[Writer] Drafting final report...")
    report_result = await writer.run(
        f"Original question: {question}\n\nResearch synthesis to format:\n{final_synthesis}"
    )
    return report_result.output


async def main() -> None:
    import sys

    question = " ".join(sys.argv[1:]) or input("Research question: ")
    config = NexusConfig()
    report = await run_research(question, config)
    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
