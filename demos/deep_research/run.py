#!/usr/bin/env python
"""Convenience runner for the Deep Research Demo.

Runs the full 4-agent hierarchical crew (supervisor + researcher +
fact-checker + writer) and prints the final report with cost stats.

Usage:
    python demos/deep_research/run.py "quantum computing in drug discovery"
    python demos/deep_research/run.py --stream "quantum computing in drug discovery"
    python demos/deep_research/run.py --no-stream "your topic"
    ANTHROPIC_API_KEY=sk-... python demos/deep_research/run.py "your topic"
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
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


async def main_stream(topic: str) -> None:
    """Run a single supervisor AgentRunner in streaming mode."""
    from demos.deep_research.agent import AGENT_ID, create_agent_def, create_runner
    from nexus.core.types import StreamEventType

    print(f"\n{'=' * 60}")
    print("  Nexus Deep Research Demo  (streaming)")
    print(f"  Topic: {topic}")
    print(f"{'=' * 60}\n")

    runner = create_runner()
    agent_def = create_agent_def()
    session_id = f"research-stream-{uuid.uuid4().hex[:8]}"

    use_color = sys.stdout.isatty()
    current_agent = AGENT_ID
    start = time.perf_counter()

    async for event in runner.stream(agent_def, topic, session_id=session_id):
        if event.event_type == StreamEventType.AGENT_START:
            current_agent = event.message or AGENT_ID
            print(f"[{current_agent}] Research started\n")

        elif event.event_type == StreamEventType.ITERATION_START:
            if event.iteration > 1:
                print()
            print(f"[{current_agent}] Step {event.iteration}: ", end="", flush=True)

        elif event.event_type == StreamEventType.TOKEN:
            if event.chunk and event.chunk.delta:
                print(event.chunk.delta, end="", flush=True)

        elif event.event_type == StreamEventType.TOOL_CALL_START:
            if event.tool_call:
                args_str = str(event.tool_call.arguments)[:80]
                if use_color:
                    print(f"\n \033[33m⚡ {event.tool_call.name}({args_str})\033[0m")
                else:
                    print(f"\n ⚡ {event.tool_call.name}({args_str})")

        elif event.event_type == StreamEventType.TOOL_CALL_END:
            if event.tool_call and event.tool_result:
                result_preview = str(event.tool_result.output or "")[:60]
                if use_color:
                    print(f"\033[32m✓ {event.tool_call.name} → {result_preview}\033[0m")
                else:
                    print(f"✓ {event.tool_call.name} → {result_preview}")

        elif event.event_type == StreamEventType.AGENT_END:
            elapsed = time.perf_counter() - start
            print(f"\n\n{'─' * 40}")
            print("  Run Statistics")
            print(f"{'─' * 40}")
            print(f"  Duration:     {elapsed:.1f}s")
            if event.chunk and event.chunk.token_usage:
                usage = event.chunk.token_usage
                print(f"  Total tokens: {usage.total_tokens:,}")
                print(f"  Total cost:   ${usage.cost_usd:.4f}")
            print()

        elif event.event_type == StreamEventType.ERROR:
            print(f"\nError: {event.message}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    # Parse --stream / --no-stream from argv; default is --stream
    raw_args = sys.argv[1:]
    use_stream = "--no-stream" not in raw_args
    topic_words = [a for a in raw_args if not a.startswith("--")]
    topic = " ".join(topic_words) or "quantum computing applications in drug discovery"

    if use_stream:
        asyncio.run(main_stream(topic))
    else:
        asyncio.run(main(topic))
