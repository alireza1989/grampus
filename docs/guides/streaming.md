# Streaming

Token streaming enables progressive display and real-time UX — users see the first word of a response within milliseconds instead of waiting for the complete generation. Grampus streams at two levels: agent execution output and eval assertions that measure streaming quality.

---

## Streaming agent output

Use `runner.stream()` to iterate over chunks as they arrive:

```python
import asyncio
import os

from grampus.core.models.anthropic import AnthropicClient
from grampus.core.types import AgentDefinition
from grampus.orchestration.runner import AgentRunner, RunnerConfig
from grampus.tools.executor import ToolExecutor
from grampus.tools.registry import ToolRegistry


async def main() -> None:
    client = AnthropicClient(api_key=os.environ["GRAMPUS_MODEL__ANTHROPIC_API_KEY"])
    runner = AgentRunner(
        model_client=client,
        tool_executor=ToolExecutor(ToolRegistry()),
        config=RunnerConfig(max_iterations=5, enable_memory=False),
    )
    agent_def = AgentDefinition(
        name="stream-agent",
        model="claude-sonnet-4-6",
        system_prompt="You are a helpful assistant.",
    )

    async for chunk in runner.stream("Explain the water cycle in detail.", agent_def):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
        if chunk.is_final:
            print()  # newline after stream ends
            print(f"Tokens: {chunk.token_usage.total_tokens}")
            print(f"Cost:   ${chunk.token_usage.cost_usd:.6f}")


asyncio.run(main())
```

### Chunk fields

| Field | Type | Description |
|-------|------|-------------|
| `delta` | `str \| None` | Incremental text fragment for this chunk |
| `is_final` | `bool` | `True` on the last chunk |
| `token_usage` | `TokenUsage \| None` | Populated on the final chunk only |
| `tool_call` | `ToolCall \| None` | Set if the LLM is emitting a tool call |
| `step` | `int` | Current ReAct iteration number |

---

## Streaming through the REST API

When the Grampus server is running (`grampus serve`), send streaming requests to `POST /stream`:

```python
import asyncio

import httpx


async def stream_agent() -> None:
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/stream",
            json={
                "agent_id": "my-agent",
                "input": "Explain the water cycle.",
                "session_id": "session-42",
            },
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    import json
                    chunk = json.loads(line[6:])
                    if chunk.get("delta"):
                        print(chunk["delta"], end="", flush=True)
                    if chunk.get("is_final"):
                        print()
                        print(f"Tokens: {chunk['token_usage']['total_tokens']}")


asyncio.run(stream_agent())
```

The endpoint uses Server-Sent Events (SSE) format. Each event is `data: <json>` followed by a blank line.

---

## Streaming eval assertions

Testing streaming requires specialized assertions that measure not just *what* was said but *how* the stream behaved. `StreamingEvalSuite` provides this.

```python
import asyncio

from grampus.evaluation.streaming import (
    StreamingEvalCase,
    StreamingEvalSuite,
    chunk_count_between,
    first_token_within,
    min_throughput,
    no_repetition,
    no_stall,
    stream_contains,
    stream_output_length,
    token_usage_reported,
)


async def main() -> None:
    suite = StreamingEvalSuite(runner=runner)

    suite.add_case(
        StreamingEvalCase(
            name="fast-response",
            user_message="What is 2 + 2?",
            assertions=[
                first_token_within(seconds=2.0),
                no_stall(max_gap_seconds=5.0),
                min_throughput(tokens_per_second=10.0),
                stream_contains("4"),
                no_repetition(window=20),
            ],
        )
    )

    suite.add_case(
        StreamingEvalCase(
            name="water-cycle-detail",
            user_message="Explain the water cycle in detail.",
            assertions=[
                first_token_within(seconds=3.0),
                stream_output_length(min_chars=200, max_chars=2000),
                no_stall(max_gap_seconds=8.0),
                chunk_count_between(min_count=5, max_count=200),
                token_usage_reported(),
            ],
        )
    )

    results = await suite.run()
    print(f"Pass rate: {results.pass_rate:.0%}  ({results.passed}/{results.total_cases})")
    for case_result in results.case_results:
        status = "PASS" if case_result.passed else "FAIL"
        print(f"  [{status}] {case_result.case_name}")
        if not case_result.passed:
            for ar in case_result.assertion_results:
                if not ar.passed:
                    print(f"       {ar.assertion_type}: {ar.detail}")


asyncio.run(main())
```

---

## Available streaming assertions

| Assertion | What it checks | Key parameter |
|-----------|---------------|---------------|
| `first_token_within(s)` | Time from request to first non-empty chunk | `seconds` |
| `stream_completes()` | Stream ends cleanly without an error | — |
| `no_stall(max_gap_seconds)` | No gap longer than N seconds between consecutive chunks | `max_gap_seconds` |
| `min_throughput(tps)` | Sustained tokens/second across the full stream | `tokens_per_second` |
| `stream_contains(text)` | Full concatenated output contains the given string | `text` |
| `stream_not_empty()` | At least one non-empty chunk was emitted | — |
| `stream_output_length(min, max)` | Total character count falls within range | `min_chars`, `max_chars` |
| `no_repetition(window)` | No phrase from a rolling window of N tokens appears again | `window` |
| `chunk_count_between(min, max)` | Number of chunks emitted falls within range | `min_count`, `max_count` |
| `token_usage_reported()` | The final chunk includes non-zero token usage data | — |

---

## See also

- **[Evaluation guide →](evaluation.md)** — Standard (non-streaming) eval assertions
- **[Model providers →](model-providers.md)** — All providers support streaming
- **[Observability guide →](observability.md)** — Trace streaming runs with OTEL spans
