"""Tests for Phase C4: webhook trigger functionality."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from grampus.core.types import AgentDefinition, AgentStatus, ExecutionResult, TokenUsage
from grampus.server.webhook import WebhookConfig, WebhookRegistry, extract_input, verify_signature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _make_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=5, output_tokens=10, total_tokens=15, cost_usd=0.0005, model="test"
    )


def _make_result() -> ExecutionResult:
    return ExecutionResult(
        output="done",
        messages=[],
        tool_calls_made=0,
        token_usage=_make_usage(),
        duration_seconds=0.1,
        steps_taken=1,
        status=AgentStatus.COMPLETED,
    )


@pytest.fixture
def mock_runner() -> MagicMock:
    runner = MagicMock()
    runner.run = AsyncMock(return_value=_make_result())
    return runner


@pytest.fixture
def mock_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-sonnet-4-6")


@pytest.fixture
def client(mock_runner: MagicMock, mock_agent_def: AgentDefinition) -> TestClient:
    from grampus.server.app import create_app

    app = create_app(mock_runner, mock_agent_def)
    return TestClient(app)


# ---------------------------------------------------------------------------
# WebhookConfig / WebhookRegistry unit tests
# ---------------------------------------------------------------------------


def test_webhook_config_auto_generates_id() -> None:
    config = WebhookConfig()
    assert config.id != ""


def test_webhook_config_auto_generates_secret() -> None:
    config = WebhookConfig()
    assert config.secret != ""


def test_webhook_registry_register_and_get() -> None:
    reg = WebhookRegistry()
    config = WebhookConfig(name="hook1")
    reg.register(config)
    assert reg.get(config.id) is config


def test_webhook_registry_delete_found() -> None:
    reg = WebhookRegistry()
    config = WebhookConfig()
    reg.register(config)
    assert reg.delete(config.id) is True
    assert reg.get(config.id) is None


def test_webhook_registry_delete_not_found() -> None:
    reg = WebhookRegistry()
    assert reg.delete("missing") is False


def test_webhook_registry_list_all() -> None:
    reg = WebhookRegistry()
    reg.register(WebhookConfig())
    reg.register(WebhookConfig())
    assert len(reg.list_all()) == 2


# ---------------------------------------------------------------------------
# HMAC verification
# ---------------------------------------------------------------------------


def test_verify_signature_valid() -> None:
    body = b'{"event": "push"}'
    secret = "mysecret"
    sig = _sign(body, secret)
    assert verify_signature(body, secret, sig) is True


def test_verify_signature_invalid() -> None:
    body = b'{"event": "push"}'
    sig = _sign(body, "right-secret")
    assert verify_signature(body, "wrong-secret", sig) is False


def test_verify_signature_missing_header() -> None:
    assert verify_signature(b"body", "secret", None) is False


def test_verify_signature_no_secret_skips_check() -> None:
    assert verify_signature(b"anything", "", None) is True
    assert verify_signature(b"anything", "", "sha256=garbage") is True


# ---------------------------------------------------------------------------
# Input extraction
# ---------------------------------------------------------------------------


def test_extract_input_default_json() -> None:
    payload = {"key": "value"}
    config = WebhookConfig()
    result = extract_input(payload, config)
    assert result == json.dumps(payload, ensure_ascii=False)


def test_extract_input_field_simple() -> None:
    config = WebhookConfig(input_field="title")
    assert extract_input({"title": "hello"}, config) == "hello"


def test_extract_input_field_nested() -> None:
    config = WebhookConfig(input_field="pull_request.title")
    payload = {"pull_request": {"title": "Fix bug"}}
    assert extract_input(payload, config) == "Fix bug"


def test_extract_input_field_missing() -> None:
    config = WebhookConfig(input_field="missing.path")
    assert extract_input({"other": "data"}, config) == ""


def test_extract_input_template() -> None:
    config = WebhookConfig(input_template="PR: {{title}} by {{user.login}}")
    payload = {"title": "Fix", "user": {"login": "alice"}}
    assert extract_input(payload, config) == "PR: Fix by alice"


def test_extract_input_field_wins_over_template() -> None:
    config = WebhookConfig(input_field="title", input_template="Template: {{title}}")
    payload = {"title": "direct"}
    assert extract_input(payload, config) == "direct"


# ---------------------------------------------------------------------------
# REST API: register / list / delete
# ---------------------------------------------------------------------------


def test_register_webhook_returns_201(client: TestClient) -> None:
    resp = client.post("/webhooks", json={"name": "my-hook"})
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert "secret" in data


def test_register_webhook_auto_generates_secret(client: TestClient) -> None:
    resp = client.post("/webhooks", json={})
    assert resp.status_code == 201
    assert resp.json()["secret"] != ""


def test_register_webhook_uses_provided_secret(client: TestClient) -> None:
    resp = client.post("/webhooks", json={"secret": "mysecret"})
    assert resp.status_code == 201
    assert resp.json()["secret"] == "mysecret"


def test_list_webhooks_empty(client: TestClient) -> None:
    resp = client.get("/webhooks")
    assert resp.status_code == 200
    assert resp.json() == {"webhooks": [], "count": 0}


def test_list_webhooks_masks_secrets(client: TestClient) -> None:
    client.post("/webhooks", json={"name": "hook"})
    resp = client.get("/webhooks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["webhooks"][0]["secret"] == "***"


def test_delete_webhook_found(client: TestClient) -> None:
    create_resp = client.post("/webhooks", json={})
    wid = create_resp.json()["id"]
    resp = client.delete(f"/webhooks/{wid}")
    assert resp.status_code == 204


def test_delete_webhook_not_found(client: TestClient) -> None:
    resp = client.delete("/webhooks/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Trigger tests
# ---------------------------------------------------------------------------


def test_trigger_sync_no_secret(client: TestClient, mock_runner: MagicMock) -> None:
    create_resp = client.post("/webhooks", json={"secret": ""})
    wid = create_resp.json()["id"]
    resp = client.post(f"/webhooks/{wid}/trigger", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "output" in data
    assert "session_id" in data


def test_trigger_sync_valid_signature(client: TestClient) -> None:
    secret = "s3cr3t"
    create_resp = client.post("/webhooks", json={"secret": secret})
    wid = create_resp.json()["id"]
    body = b"{}"
    sig = _sign(body, secret)
    resp = client.post(
        f"/webhooks/{wid}/trigger",
        content=body,
        headers={"Content-Type": "application/json", "X-Grampus-Signature": sig},
    )
    assert resp.status_code == 200


def test_trigger_sync_invalid_signature(client: TestClient) -> None:
    create_resp = client.post("/webhooks", json={"secret": "s3cr3t"})
    wid = create_resp.json()["id"]
    resp = client.post(
        f"/webhooks/{wid}/trigger",
        json={},
        headers={"X-Grampus-Signature": "sha256=deadbeef"},
    )
    assert resp.status_code == 401


def test_trigger_sync_missing_signature(client: TestClient) -> None:
    create_resp = client.post("/webhooks", json={"secret": "s3cr3t"})
    wid = create_resp.json()["id"]
    resp = client.post(f"/webhooks/{wid}/trigger", json={})
    assert resp.status_code == 401


def test_trigger_async_returns_202_accepted(client: TestClient) -> None:
    create_resp = client.post("/webhooks", json={"secret": "", "async_mode": True})
    wid = create_resp.json()["id"]
    with patch("grampus.server.routes.asyncio.create_task") as mock_task:
        resp = client.post(f"/webhooks/{wid}/trigger", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    assert "session_id" in data
    mock_task.assert_called_once()
    # Close the captured coroutine so Python doesn't emit a RuntimeWarning.
    mock_task.call_args[0][0].close()


def test_trigger_not_found(client: TestClient) -> None:
    resp = client.post("/webhooks/nonexistent/trigger", json={})
    assert resp.status_code == 404


def test_trigger_uses_input_field(client: TestClient, mock_runner: MagicMock) -> None:
    create_resp = client.post("/webhooks", json={"secret": "", "input_field": "text"})
    wid = create_resp.json()["id"]
    client.post(f"/webhooks/{wid}/trigger", json={"text": "hello world"})
    call_args = mock_runner.run.call_args
    assert call_args[0][1] == "hello world"


def test_trigger_default_json_input(client: TestClient, mock_runner: MagicMock) -> None:
    create_resp = client.post("/webhooks", json={"secret": ""})
    wid = create_resp.json()["id"]
    payload = {"key": "value"}
    client.post(f"/webhooks/{wid}/trigger", json=payload)
    call_args = mock_runner.run.call_args
    assert call_args[0][1] == json.dumps(payload, ensure_ascii=False)
