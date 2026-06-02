"""Tests for the GET /metrics Prometheus endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nexus.core.types import AgentDefinition, AgentStatus, ExecutionResult, TokenUsage
from nexus.observability.metrics import NexusMetrics


def _make_result() -> ExecutionResult:
    return ExecutionResult(
        output="ok",
        messages=[],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="test"
        ),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


@pytest.fixture
def metrics() -> NexusMetrics:
    return NexusMetrics(agent_id="test-agent")


@pytest.fixture
def mock_runner() -> MagicMock:
    runner = MagicMock()
    runner.run = AsyncMock(return_value=_make_result())
    return runner


@pytest.fixture
def mock_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-sonnet-4-6")


@pytest.fixture
def app_with_metrics(
    mock_runner: MagicMock,
    mock_agent_def: AgentDefinition,
    metrics: NexusMetrics,
) -> object:
    from nexus.server.app import create_app

    return create_app(mock_runner, mock_agent_def, nexus_metrics=metrics)


@pytest.fixture
def app_no_metrics(
    mock_runner: MagicMock,
    mock_agent_def: AgentDefinition,
) -> object:
    from nexus.server.app import create_app

    return create_app(mock_runner, mock_agent_def, nexus_metrics=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_200(app_with_metrics: object) -> None:
    client = TestClient(app_with_metrics)
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_content_type_is_text_plain(app_with_metrics: object) -> None:
    client = TestClient(app_with_metrics)
    response = client.get("/metrics")
    assert response.headers["content-type"].startswith("text/plain")


def test_metrics_contains_nexus_token_counter(
    app_with_metrics: object, metrics: NexusMetrics
) -> None:
    metrics.record_llm_call(
        model="test-model",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.005,
        latency_ms=250.0,
    )
    client = TestClient(app_with_metrics)
    response = client.get("/metrics")
    assert "nexus_total_tokens" in response.text


def test_metrics_contains_help_lines(app_with_metrics: object, metrics: NexusMetrics) -> None:
    metrics.record_llm_call(
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=100.0,
    )
    client = TestClient(app_with_metrics)
    response = client.get("/metrics")
    assert "# HELP nexus_total_tokens" in response.text


def test_metrics_contains_type_lines(app_with_metrics: object, metrics: NexusMetrics) -> None:
    metrics.record_llm_call(
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=100.0,
    )
    client = TestClient(app_with_metrics)
    response = client.get("/metrics")
    assert "# TYPE nexus_total_tokens counter" in response.text


def test_metrics_no_collector_returns_comment(app_no_metrics: object) -> None:
    client = TestClient(app_no_metrics)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "# No metrics collector configured" in response.text


def test_metrics_histogram_buckets_present(app_with_metrics: object, metrics: NexusMetrics) -> None:
    metrics.record_llm_call(
        model="test-model",
        input_tokens=20,
        output_tokens=10,
        cost_usd=0.002,
        latency_ms=300.0,
    )
    client = TestClient(app_with_metrics)
    response = client.get("/metrics")
    assert "nexus_llm_latency_ms_bucket" in response.text


def test_metrics_agent_id_label_present(app_with_metrics: object, metrics: NexusMetrics) -> None:
    metrics.record_llm_call(
        model="test-model",
        input_tokens=5,
        output_tokens=3,
        cost_usd=0.0005,
        latency_ms=50.0,
    )
    client = TestClient(app_with_metrics)
    response = client.get("/metrics")
    assert 'agent_id="test-agent"' in response.text
