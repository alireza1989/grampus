"""Tests for grampus.server.models request/response schemas."""

from __future__ import annotations

import json

from grampus.core.types import AgentStatus, TokenUsage


class TestRunRequest:
    def test_run_request_defaults(self) -> None:
        from grampus.server.models import RunRequest

        req = RunRequest(input="hello")
        assert req.session_id is None
        assert req.agent_name is None
        assert req.temperature is None
        assert req.max_iterations is None

    def test_run_request_with_all_fields(self) -> None:
        from grampus.server.models import RunRequest

        req = RunRequest(
            input="hello",
            session_id="sid-123",
            agent_name="my-agent",
            temperature=0.7,
            max_iterations=5,
        )
        assert req.session_id == "sid-123"
        assert req.temperature == 0.7
        assert req.max_iterations == 5


class TestRunResponse:
    def _usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model="m"
        )

    def test_run_response_serializes(self) -> None:
        from grampus.server.models import RunResponse

        resp = RunResponse(
            output="hi",
            session_id="s1",
            steps_taken=1,
            tool_calls_made=0,
            token_usage=self._usage(),
            duration_seconds=0.5,
            status=str(AgentStatus.COMPLETED),
        )
        data = json.loads(resp.model_dump_json())
        assert data["output"] == "hi"
        assert data["session_id"] == "s1"
        assert data["status"] == "completed"

    def test_run_response_roundtrip(self) -> None:
        from grampus.server.models import RunResponse

        resp = RunResponse(
            output=None,
            session_id="s2",
            steps_taken=2,
            tool_calls_made=3,
            token_usage=self._usage(),
            duration_seconds=1.2,
            status="failed",
        )
        data = json.loads(resp.model_dump_json())
        restored = RunResponse(**data)
        assert restored.session_id == "s2"
        assert restored.output is None


class TestStreamChunkResponse:
    def test_stream_chunk_response_token_event(self) -> None:
        from grampus.server.models import StreamChunkResponse

        chunk = StreamChunkResponse(event_type="token", delta="Hello")
        data = json.loads(chunk.model_dump_json())
        assert data["event_type"] == "token"
        assert data["delta"] == "Hello"
        assert data["tool_name"] is None

    def test_stream_chunk_defaults(self) -> None:
        from grampus.server.models import StreamChunkResponse

        chunk = StreamChunkResponse(event_type="agent_start")
        assert chunk.delta == ""
        assert chunk.message == ""
        assert chunk.token_usage is None


class TestHealthResponse:
    def test_health_response_fields(self) -> None:
        from grampus.server.models import HealthResponse

        h = HealthResponse(status="ok", version="0.1.0", agent_name="my-agent")
        assert h.status == "ok"
        assert h.version == "0.1.0"
        assert h.agent_name == "my-agent"


class TestMemoryModels:
    def test_memory_recall_request_defaults(self) -> None:
        from grampus.server.models import MemoryRecallRequest

        req = MemoryRecallRequest(query="what happened?")
        assert req.top_k == 5
        assert "episodic" in req.memory_types
        assert "semantic" in req.memory_types

    def test_memory_recall_response_empty(self) -> None:
        from grampus.server.models import MemoryRecallResponse

        resp = MemoryRecallResponse(query="q")
        assert resp.episodic == []
        assert resp.semantic == []
