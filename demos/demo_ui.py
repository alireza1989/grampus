"""
UI dev demo — no API key, no Dapr required.

Starts the Nexus server with the web UI pre-populated with sample memory
records so you can explore every page without a real LLM or database.

Run:
    uv run python demos/demo_ui.py

Then open:
    http://localhost:8000/ui/           — Dashboard
    http://localhost:8000/ui/memory/    — Memory Inspector
    http://localhost:8000/docs          — API docs
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import uvicorn
from pydantic import BaseModel

from nexus.core.types import AgentDefinition
from nexus.evaluation.run_store import EvalRunRecord, EvalRunStore
from nexus.observability.metrics import NexusMetrics
from nexus.orchestration.runner import AgentRunner
from nexus.server.app import create_app
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Minimal in-memory state store (no Dapr)
# ---------------------------------------------------------------------------


class _StateStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def save(self, namespace: str, key: str, value: Any) -> None:
        if isinstance(value, BaseModel):
            self._data[key] = value.model_dump_json()
        else:
            self._data[key] = json.dumps(value)

    async def get(self, namespace: str, key: str, model_class: type) -> tuple[Any, str]:
        raw = self._data.get(key)
        if raw is None:
            return None, ""
        return model_class.model_validate_json(raw), "etag-1"

    async def delete(self, namespace: str, key: str) -> None:
        self._data.pop(key, None)


# ---------------------------------------------------------------------------
# Sample memory records for the inspector
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {
        "id": "ep-aaaa-1111-2222-3333",
        "agent_id": "research-agent",
        "memory_type": "episodic",
        "content": "The user prefers concise bullet-point summaries over long prose. Always use markdown lists when presenting research results.",
        "trust_score": 0.92,
        "created_at": datetime(2026, 6, 9, 14, 23, tzinfo=UTC),
        "last_accessed": datetime(2026, 6, 9, 16, 5, tzinfo=UTC),
        "metadata": {"session_id": "sess-abc123", "importance_score": 0.85},
        "provenance": {
            "source_type": "USER_INPUT",
            "trust_level": 0.9,
            "content_hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        },
    },
    {
        "id": "sf-bbbb-4444-5555-6666",
        "agent_id": "research-agent",
        "memory_type": "semantic",
        "content": "research-agent knows Python is a dynamically-typed, interpreted programming language",
        "trust_score": 0.88,
        "created_at": datetime(2026, 6, 8, 10, 0, tzinfo=UTC),
        "last_accessed": None,
        "metadata": {"source_episode_ids": ["ep-aaaa-1111-2222-3333"]},
        "provenance": None,
    },
    {
        "id": "sf-cccc-7777-8888-9999",
        "agent_id": "support-agent",
        "memory_type": "semantic",
        "content": "support-agent knows Nexus framework uses Dapr for distributed state management",
        "trust_score": 0.95,
        "created_at": datetime(2026, 6, 7, 9, 30, tzinfo=UTC),
        "last_accessed": datetime(2026, 6, 9, 11, 0, tzinfo=UTC),
        "metadata": {"source_episode_ids": []},
        "provenance": None,
    },
    {
        "id": "pr-dddd-aaaa-bbbb-cccc",
        "agent_id": "research-agent",
        "memory_type": "procedural",
        "content": "web-search-and-summarize: Search for information, then summarize findings. fetch_url extract_text summarize_content",
        "trust_score": None,
        "created_at": datetime(2026, 6, 6, 8, 0, tzinfo=UTC),
        "last_accessed": datetime(2026, 6, 9, 14, 0, tzinfo=UTC),
        "metadata": {"success_count": 12, "failure_count": 1},
        "provenance": None,
    },
    {
        "id": "ep-eeee-dddd-eeee-ffff",
        "agent_id": "support-agent",
        "memory_type": "episodic",
        "content": "User reported that the /run endpoint returns a 422 when session_id contains spaces. Confirmed bug, filed as GH-447.",
        "trust_score": 0.35,
        "created_at": datetime(2026, 6, 5, 16, 45, tzinfo=UTC),
        "last_accessed": None,
        "metadata": {"session_id": "sess-support-7", "user_id": "user-42"},
        "provenance": {
            "source_type": "TOOL_RESULT",
            "trust_level": 0.6,
            "content_hash": "deadbeefdeadbeef1234567890abcdef",
        },
    },
]


# ---------------------------------------------------------------------------
# Mock MemoryManager backed by the sample records
# ---------------------------------------------------------------------------


def _build_mock_manager() -> Any:
    manager = MagicMock()

    async def list_records(
        *,
        agent_id: str | None = None,
        memory_type: str | None = None,
        query: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        records = list(_SAMPLE_RECORDS)
        if agent_id:
            records = [r for r in records if r.get("agent_id") == agent_id]
        if memory_type:
            records = [r for r in records if r.get("memory_type") == memory_type]
        if query:
            q = query.lower()
            records = [r for r in records if q in (r.get("content") or "").lower()]
        if min_trust:
            records = [
                r
                for r in records
                if r.get("trust_score") is None or (r.get("trust_score") or 0.0) >= min_trust
            ]
        records.sort(
            key=lambda r: r.get("created_at") or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        return records[offset : offset + limit]

    async def forget(record_id: str, *, memory_type: str) -> None:
        print(f"[demo] forget({record_id!r}, memory_type={memory_type!r}) — no-op in demo")

    manager.list_records = list_records
    manager.forget = forget
    return manager


# ---------------------------------------------------------------------------
# Build app
# ---------------------------------------------------------------------------


def _build_eval_store() -> EvalRunStore:
    store = EvalRunStore()

    def _case(name: str, passed: bool, atype: str = "contains") -> dict:
        return {
            "case_id": f"c-{name}",
            "case_name": name,
            "passed": passed,
            "assertion_results": [
                {
                    "passed": passed,
                    "assertion_type": atype,
                    "detail": "ok" if passed else "failed",
                    "score": 1.0 if passed else 0.0,
                }
            ],
            "duration_seconds": 0.12 if passed else 0.34,
            "error": None,
            "tags": [],
        }

    runs = [
        # Suite A — improving trend
        EvalRunRecord(
            suite_name="smoke-tests",
            run_at=datetime(2026, 5, 28, 10, 0, tzinfo=UTC),
            pass_rate=0.60,
            passed=6,
            failed=4,
            errors=0,
            total_cases=10,
            total_cost_usd=0.0082,
            avg_duration_seconds=0.21,
            case_results=[
                _case("test-health-check", True),
                _case("test-run-basic", False),
                _case("test-memory-recall", True),
                _case("test-tool-call", False),
            ],
        ),
        EvalRunRecord(
            suite_name="smoke-tests",
            run_at=datetime(2026, 5, 31, 10, 0, tzinfo=UTC),
            pass_rate=0.70,
            passed=7,
            failed=3,
            errors=0,
            total_cases=10,
            total_cost_usd=0.0091,
            avg_duration_seconds=0.19,
            case_results=[
                _case("test-health-check", True),
                _case("test-run-basic", True),
                _case("test-memory-recall", True),
                _case("test-tool-call", False),
            ],
        ),
        EvalRunRecord(
            suite_name="smoke-tests",
            run_at=datetime(2026, 6, 3, 10, 0, tzinfo=UTC),
            pass_rate=0.80,
            passed=8,
            failed=2,
            errors=0,
            total_cases=10,
            total_cost_usd=0.0074,
            avg_duration_seconds=0.18,
            case_results=[
                _case("test-health-check", True),
                _case("test-run-basic", True),
                _case("test-memory-recall", True),
                _case("test-tool-call", True),
            ],
        ),
        EvalRunRecord(
            suite_name="smoke-tests",
            run_at=datetime(2026, 6, 6, 10, 0, tzinfo=UTC),
            pass_rate=0.90,
            passed=9,
            failed=1,
            errors=0,
            total_cases=10,
            total_cost_usd=0.0068,
            avg_duration_seconds=0.17,
            case_results=[
                _case("test-health-check", True),
                _case("test-run-basic", True),
                _case("test-memory-recall", True),
                _case("test-tool-call", True),
            ],
        ),
        # Suite B — regression then recovery
        EvalRunRecord(
            suite_name="safety-checks",
            run_at=datetime(2026, 5, 30, 14, 0, tzinfo=UTC),
            pass_rate=1.00,
            passed=5,
            failed=0,
            errors=0,
            total_cases=5,
            total_cost_usd=0.0031,
            avg_duration_seconds=0.14,
            case_results=[
                _case("test-injection-block", True, "no_injection_patterns"),
                _case("test-pii-redact", True, "no_pii"),
            ],
        ),
        EvalRunRecord(
            suite_name="safety-checks",
            run_at=datetime(2026, 6, 2, 14, 0, tzinfo=UTC),
            pass_rate=0.60,
            passed=3,
            failed=2,
            errors=0,
            total_cases=5,
            total_cost_usd=0.0029,
            avg_duration_seconds=0.22,
            case_results=[
                _case("test-injection-block", True, "no_injection_patterns"),
                _case("test-pii-redact", False, "no_pii"),
            ],
        ),
        EvalRunRecord(
            suite_name="safety-checks",
            run_at=datetime(2026, 6, 6, 14, 0, tzinfo=UTC),
            pass_rate=1.00,
            passed=5,
            failed=0,
            errors=0,
            total_cases=5,
            total_cost_usd=0.0033,
            avg_duration_seconds=0.13,
            case_results=[
                _case("test-injection-block", True, "no_injection_patterns"),
                _case("test-pii-redact", True, "no_pii"),
            ],
        ),
        # Suite C — quality regression (latest run is worst)
        EvalRunRecord(
            suite_name="quality-baseline",
            run_at=datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
            pass_rate=0.85,
            passed=17,
            failed=3,
            errors=0,
            total_cases=20,
            total_cost_usd=0.0412,
            avg_duration_seconds=0.88,
            case_results=[
                _case("test-output-quality", True, "semantic_similarity"),
                _case("test-no-hallucination", False, "contains"),
                _case("test-tool-selection", True, "tool_was_called"),
            ],
        ),
        EvalRunRecord(
            suite_name="quality-baseline",
            run_at=datetime(2026, 6, 5, 9, 0, tzinfo=UTC),
            pass_rate=0.70,
            passed=14,
            failed=6,
            errors=0,
            total_cases=20,
            total_cost_usd=0.0398,
            avg_duration_seconds=0.91,
            case_results=[
                _case("test-output-quality", False, "semantic_similarity"),
                _case("test-no-hallucination", False, "contains"),
                _case("test-tool-selection", True, "tool_was_called"),
            ],
        ),
    ]
    for r in runs:
        store.append(r)
    return store


def build_app() -> Any:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    state_store = _StateStore()

    # Minimal no-LLM runner (UI demo doesn't need to run agents)
    runner = AgentRunner(
        MagicMock(),  # model client — unused in UI demo
        executor,
        state_store=state_store,
    )

    agent_def = AgentDefinition(name="ui-demo-agent", model="demo")

    # Populate metrics across two models for the cost breakdown page
    metrics = NexusMetrics(agent_id="ui-demo-agent")
    metrics.record_llm_call(
        model="claude-sonnet-4-6",
        input_tokens=1_200,
        output_tokens=430,
        cost_usd=0.0086,
        latency_ms=820,
    )
    metrics.record_llm_call(
        model="claude-sonnet-4-6",
        input_tokens=980,
        output_tokens=310,
        cost_usd=0.0061,
        latency_ms=640,
    )
    metrics.record_llm_call(
        model="claude-opus-4-7",
        input_tokens=3_400,
        output_tokens=890,
        cost_usd=0.0432,
        latency_ms=2_100,
    )
    metrics.record_llm_call(
        model="claude-haiku-4-5",
        input_tokens=620,
        output_tokens=180,
        cost_usd=0.0008,
        latency_ms=190,
    )
    metrics.record_llm_call(
        model="claude-haiku-4-5",
        input_tokens=410,
        output_tokens=120,
        cost_usd=0.0005,
        latency_ms=140,
    )
    metrics.record_tool_call(tool_name="web_search", success=True, latency_ms=210)
    metrics.record_tool_call(tool_name="fetch_url", success=True, latency_ms=390)
    metrics.record_error(error_type="TimeoutError")
    metrics.set_active_agents(2)

    app = create_app(
        runner,
        agent_def,
        memory_manager=_build_mock_manager(),
        nexus_metrics=metrics,
        eval_run_store=_build_eval_store(),
    )
    return app


if __name__ == "__main__":
    print("\n  Nexus UI demo starting…")
    print("  Dashboard   →  http://localhost:8000/ui/")
    print("  Memory      →  http://localhost:8000/ui/memory/")
    print("  Evals       →  http://localhost:8000/ui/evals/")
    print("  Cost        →  http://localhost:8000/ui/cost/")
    print("  API docs    →  http://localhost:8000/docs")
    print("  Press Ctrl-C to stop.\n")
    uvicorn.run(build_app(), host="127.0.0.1", port=8000, log_level="warning")
