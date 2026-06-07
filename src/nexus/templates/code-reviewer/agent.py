"""{{project_name}} — automated code review agent."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nexus.core.types import AgentDefinition
from nexus.orchestration.runner import AgentRunner

REPO_PATH = Path("{{repo_path}}")

SYSTEM_PROMPT = f"""You are an expert code reviewer.
Repository path: {REPO_PATH}

Your job:
1. Use file_read to read source files
2. Identify: bugs, security vulnerabilities, style issues, performance problems
3. Produce a structured report with: Summary, Issues (severity: critical/high/medium/low),
   Positive observations, and Recommendations

Be specific: cite file names and line numbers. Be constructive.
Do NOT modify any files — read only.
"""


async def main() -> None:
    agent = AgentRunner(
        agent_def=AgentDefinition(
            name="{{project_name}}",
            model="{{model}}",
            system_prompt=SYSTEM_PROMPT,
            max_iterations=20,
            temperature=0.1,
            memory_enabled=False,
        ),
    )

    import sys

    instruction = " ".join(sys.argv[1:]) or f"Review the code in {REPO_PATH}"
    result = await agent.run(instruction)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
