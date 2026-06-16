"""Tests for alert REST API endpoints (Phase D7)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from grampus.core.types import AgentDefinition
from grampus.observability.alerts import (
    AlertEvaluator,
    AlertEvent,
    AlertSeverity,
    ThresholdType,
)
from grampus.observability.notification import LogChannel, NotificationDispatcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-sonnet-4-6")


def _make_runner() -> MagicMock:
    runner = MagicMock()
    runner.list_pending_sessions = MagicMock(return_value=[])
    return runner


def _make_evaluator() -> AlertEvaluator:
    disp = NotificationDispatcher(channels=[LogChannel()])
    return AlertEvaluator(rules=[], dispatcher=disp)


def _make_client(evaluator: AlertEvaluator | None = None) -> TestClient:
    from grampus.server.app import create_app

    runner = _make_runner()
    agent_def = _make_agent_def()
    app = create_app(runner, agent_def, alert_evaluator=evaluator)
    return TestClient(app)


def _rule_payload(**kwargs) -> dict:
    base = {
        "name": "test-rule",
        "threshold_type": "per_session_usd",
        "threshold_usd": 0.30,
        "severity": "warning",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# POST /alerts/rules
# ---------------------------------------------------------------------------


class TestCreateAlertRule:
    def test_create_alert_rule(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        resp = client.post("/alerts/rules", json=_rule_payload())
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test-rule"
        assert "rule_id" in data
        assert data["threshold_usd"] == 0.30

    def test_create_rule_persisted_in_evaluator(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        client.post("/alerts/rules", json=_rule_payload(name="persisted-rule"))
        assert len(ev.list_rules()) == 1
        assert ev.list_rules()[0].name == "persisted-rule"

    def test_create_rule_with_agent_filter(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        resp = client.post("/alerts/rules", json=_rule_payload(agent_id="specific-bot"))
        assert resp.status_code == 201
        assert resp.json()["agent_id"] == "specific-bot"

    def test_create_rule_returns_404_when_no_evaluator(self) -> None:
        client = _make_client(evaluator=None)
        resp = client.post("/alerts/rules", json=_rule_payload())
        assert resp.status_code in (404, 503)


# ---------------------------------------------------------------------------
# GET /alerts/rules
# ---------------------------------------------------------------------------


class TestListAlertRules:
    def test_list_alert_rules_empty(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        resp = client.get("/alerts/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rules"] == []
        assert data["count"] == 0

    def test_list_alert_rules_returns_created_rules(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        client.post("/alerts/rules", json=_rule_payload(name="rule-1"))
        client.post("/alerts/rules", json=_rule_payload(name="rule-2"))
        resp = client.get("/alerts/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        names = {r["name"] for r in data["rules"]}
        assert names == {"rule-1", "rule-2"}


# ---------------------------------------------------------------------------
# GET /alerts/rules/{rule_id}
# ---------------------------------------------------------------------------


class TestGetAlertRule:
    def test_get_alert_rule(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        create_resp = client.post("/alerts/rules", json=_rule_payload())
        rule_id = create_resp.json()["rule_id"]

        resp = client.get(f"/alerts/rules/{rule_id}")
        assert resp.status_code == 200
        assert resp.json()["rule_id"] == rule_id

    def test_get_nonexistent_rule_returns_404(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        resp = client.get("/alerts/rules/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /alerts/rules/{rule_id}
# ---------------------------------------------------------------------------


class TestDeleteAlertRule:
    def test_delete_alert_rule(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        create_resp = client.post("/alerts/rules", json=_rule_payload())
        rule_id = create_resp.json()["rule_id"]

        del_resp = client.delete(f"/alerts/rules/{rule_id}")
        assert del_resp.status_code == 204

        list_resp = client.get("/alerts/rules")
        assert list_resp.json()["count"] == 0

    def test_delete_nonexistent_returns_404(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        resp = client.delete("/alerts/rules/no-such-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /alerts/rules/{rule_id}
# ---------------------------------------------------------------------------


class TestToggleAlertRule:
    def test_toggle_alert_rule_disabled(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        create_resp = client.post("/alerts/rules", json=_rule_payload())
        rule_id = create_resp.json()["rule_id"]

        patch_resp = client.patch(f"/alerts/rules/{rule_id}", json={"enabled": False})
        assert patch_resp.status_code == 200
        assert patch_resp.json()["enabled"] is False
        assert ev.list_rules()[0].enabled is False

    def test_toggle_alert_rule_re_enabled(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        create_resp = client.post("/alerts/rules", json=_rule_payload())
        rule_id = create_resp.json()["rule_id"]

        client.patch(f"/alerts/rules/{rule_id}", json={"enabled": False})
        patch_resp = client.patch(f"/alerts/rules/{rule_id}", json={"enabled": True})
        assert patch_resp.status_code == 200
        assert patch_resp.json()["enabled"] is True


# ---------------------------------------------------------------------------
# GET /alerts/history
# ---------------------------------------------------------------------------


class TestAlertHistory:
    def test_alert_history_empty(self) -> None:
        ev = _make_evaluator()
        client = _make_client(ev)
        resp = client.get("/alerts/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert data["count"] == 0

    def test_alert_history_populated(self) -> None:
        from grampus.server.app import create_app

        runner = _make_runner()
        agent_def = _make_agent_def()
        ev = _make_evaluator()
        app = create_app(runner, agent_def, alert_evaluator=ev)

        # Manually inject a fired event into alert_history
        event = AlertEvent(
            rule_id="r1",
            rule_name="test-rule",
            agent_id="bot-1",
            session_id="sess-1",
            severity=AlertSeverity.WARNING,
            threshold_type=ThresholdType.PER_SESSION_USD,
            threshold_usd=0.30,
            actual_usd=0.42,
            message="Agent bot-1 spent $0.42",
        )
        app.state.alert_history.append(event)

        client = TestClient(app)
        resp = client.get("/alerts/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["events"][0]["rule_id"] == "r1"

    def test_alert_history_limit_param(self) -> None:
        from grampus.server.app import create_app

        runner = _make_runner()
        agent_def = _make_agent_def()
        ev = _make_evaluator()
        app = create_app(runner, agent_def, alert_evaluator=ev)

        for i in range(10):
            event = AlertEvent(
                rule_id=f"rule-{i}",
                rule_name=f"rule-{i}",
                agent_id="bot-1",
                session_id="sess-1",
                severity=AlertSeverity.WARNING,
                threshold_type=ThresholdType.PER_SESSION_USD,
                threshold_usd=0.30,
                actual_usd=0.42,
                message=f"event {i}",
            )
            app.state.alert_history.append(event)

        client = TestClient(app)
        resp = client.get("/alerts/history?limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) == 3

    def test_alert_history_agent_id_filter(self) -> None:
        from grampus.server.app import create_app

        runner = _make_runner()
        agent_def = _make_agent_def()
        ev = _make_evaluator()
        app = create_app(runner, agent_def, alert_evaluator=ev)

        for agent in ("bot-1", "bot-2", "bot-1"):
            event = AlertEvent(
                rule_id="r1",
                rule_name="rule",
                agent_id=agent,
                session_id="sess-1",
                severity=AlertSeverity.WARNING,
                threshold_type=ThresholdType.PER_SESSION_USD,
                threshold_usd=0.30,
                actual_usd=0.42,
                message=f"event for {agent}",
            )
            app.state.alert_history.append(event)

        client = TestClient(app)
        resp = client.get("/alerts/history?agent_id=bot-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        for e in data["events"]:
            assert e["agent_id"] == "bot-1"
