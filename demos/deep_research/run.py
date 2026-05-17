#!/usr/bin/env python
"""Convenience runner for the Deep Research Demo.

Runs the full 4-agent hierarchical crew (supervisor + researcher +
fact-checker + writer) and prints the final report with cost stats.

Usage:
    python demos/deep_research/run.py "quantum computing in drug discovery"
    ANTHROPIC_API_KEY=sk-... python demos/deep_research/run.py "your topic"
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when executed directly as a script
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


async def main(topic: str) -> None:
    # Import here to keep startup fast for --help
    from demos.deep_research.agent import AGENT_ID, build_crew

    print(f"\n{'=' * 60}")
    print("  Nexus Deep Research Demo")
    print(f"  Topic: {topic}")
    print(f"{'=' * 60}\n")

    crew, session_id = build_crew()
    start = time.perf_counter()

    print("[crew] Starting hierarchical 4-agent research workflow...")
    print("[crew] supervisor → researcher + fact-checker + writer\n")

    crew_result = await crew.run(topic)
    elapsed = time.perf_counter() - start

    # Supervisor's final output is the approved research report
    supervisor_output = crew_result.outputs.get(AGENT_ID, "")
    if not supervisor_output:
        # Fall back to any worker output
        supervisor_output = next(iter(crew_result.outputs.values()), "(no output)")

    print(f"\n{'=' * 60}")
    print("  RESEARCH COMPLETE")
    print(f"{'=' * 60}")
    print(supervisor_output)

    print(f"\n{'─' * 40}")
    print("  Run Statistics")
    print(f"{'─' * 40}")
    print(f"  Duration:     {elapsed:.1f}s")
    print(f"  Agents:       {len(crew_result.outputs)}")
    print(f"  Pattern:      {crew_result.pattern}")
    print(f"  Total cost:   ${crew_result.total_cost_usd:.4f}")

    per_agent = {k: v[:60] + "..." if len(v) > 60 else v for k, v in crew_result.outputs.items()}
    print("\n  Per-agent outputs:")
    for agent_name, snippet in per_agent.items():
        print(f"    {agent_name}: {snippet!r}")
    print()


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "quantum computing applications in drug discovery"
    asyncio.run(main(topic))
