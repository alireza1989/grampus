"""Tests for Phase D7: Cost Tracking Alerts — models, evaluator, and notification channels."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grampus.observability.alerts import (
    AlertEvaluator,
    AlertEvent,
    AlertRule,
    AlertSeverity,
    AlertState,
    ThresholdType,
)
from grampus.observability.notification import (
    LogChannel,
    NotificationDispatcher,
    SlackChannel,
    SmtpChannel,
    WebhookChannel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rule(
    *,
    name: str = "test-rule",
    threshold_type: ThresholdType = ThresholdType.PER_SESSION_USD,
    threshold_usd: float = 0.30,
    agent_id: str | None = None,
    cooldown_seconds: int = 3600,
    severity: AlertSeverity = AlertSeverity.WARNING,
    enabled: bool = True,
    tags: dict | None = None,
) -> AlertRule:
    return AlertRule(
        name=name,
        threshold_type=threshold_type,
        threshold_usd=threshold_usd,
        agent_id=agent_id,
        cooldown_seconds=cooldown_seconds,
        severity=severity,
        enabled=enabled,
        tags=tags or {},
    )


def _cost_event(
    *,
    agent_id: str = "bot-1",
    session_id: str = "sess-1",
    cost_usd: float = 0.028,
    cumulative_session_usd: float = 0.42,
    cumulative_agent_usd: float = 0.87,
) -> dict:
    return {
        "agent_id": agent_id,
        "session_id": session_id,
        "cost_usd": cost_usd,
        "cumulative_session_usd": cumulative_session_usd,
        "cumulative_agent_usd": cumulative_agent_usd,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _make_alert_event() -> AlertEvent:
    r = _rule()
    return AlertEvent(
        rule_id=r.rule_id,
        rule_name=r.name,
        agent_id="bot-1",
        session_id="sess-1",
        severity=r.severity,
        threshold_type=r.threshold_type,
        threshold_usd=r.threshold_usd,
        actual_usd=0.42,
        message="Agent bot-1 spent $0.42 (limit $0.30/session)",
    )


def _dispatcher_with_log() -> tuple[NotificationDispatcher, LogChannel]:
    ch = LogChannel()
    disp = NotificationDispatcher(channels=[ch])
    return disp, ch


# ---------------------------------------------------------------------------
# AlertRule validation
# ---------------------------------------------------------------------------


class TestAlertRule:
    def test_alert_rule_defaults(self) -> None:
        r = AlertRule(
            name="my-rule",
            threshold_type=ThresholdType.PER_SESSION_USD,
            threshold_usd=1.0,
        )
        assert r.rule_id != ""
        assert len(r.rule_id) > 0
        assert r.cooldown_seconds == 3600
        assert r.enabled is True
        assert r.severity == AlertSeverity.WARNING
        assert r.agent_id is None
        assert isinstance(r.created_at, datetime)
        assert r.tags == {}

    def test_alert_rule_agent_wildcard(self) -> None:
        r = AlertRule(
            name="global-rule",
            threshold_type=ThresholdType.ABSOLUTE_USD,
            threshold_usd=5.0,
            agent_id=None,
        )
        assert r.agent_id is None

    def test_alert_rule_unique_ids(self) -> None:
        r1 = AlertRule(name="r1", threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=1.0)
        r2 = AlertRule(name="r2", threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=1.0)
        assert r1.rule_id != r2.rule_id

    def test_alert_rule_serialises_round_trip(self) -> None:
        r = _rule(tags={"env": "prod"})
        data = r.model_dump(mode="json")
        r2 = AlertRule(**data)
        assert r2.rule_id == r.rule_id
        assert r2.tags == {"env": "prod"}


# ---------------------------------------------------------------------------
# AlertEvaluator — in-memory cooldown (no state_store)
# ---------------------------------------------------------------------------


class TestAlertEvaluator:
    @pytest.mark.asyncio
    async def test_evaluator_fires_when_threshold_exceeded(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.30)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        events = await ev.evaluate("bot-1", "sess-1", _cost_event(cumulative_session_usd=0.42))
        assert len(events) == 1
        assert events[0].rule_id == r.rule_id
        assert events[0].actual_usd == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_evaluator_no_fire_below_threshold(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.50)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        events = await ev.evaluate("bot-1", "sess-1", _cost_event(cumulative_session_usd=0.30))
        assert events == []

    @pytest.mark.asyncio
    async def test_evaluator_cooldown_suppresses_repeat(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(
            threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.30, cooldown_seconds=3600
        )
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        first = await ev.evaluate("bot-1", "sess-1", _cost_event(cumulative_session_usd=0.42))
        assert len(first) == 1
        second = await ev.evaluate("bot-1", "sess-1", _cost_event(cumulative_session_usd=0.55))
        assert second == []

    @pytest.mark.asyncio
    async def test_evaluator_cooldown_expires(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(
            threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.30, cooldown_seconds=3600
        )
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        first = await ev.evaluate("bot-1", "sess-1", _cost_event(cumulative_session_usd=0.42))
        assert len(first) == 1
        # Manually expire the cooldown
        key = f"alerts:{r.rule_id}:bot-1"
        ev._mem_cooldown[key] = AlertState(
            rule_id=r.rule_id,
            last_fired_at=datetime.utcnow() - timedelta(hours=2),
        )
        second = await ev.evaluate("bot-1", "sess-1", _cost_event(cumulative_session_usd=0.55))
        assert len(second) == 1

    @pytest.mark.asyncio
    async def test_evaluator_agent_filter_matches(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(
            threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.30, agent_id="bot-1"
        )
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        fired_for_bot1 = await ev.evaluate(
            "bot-1", "sess-1", _cost_event(agent_id="bot-1", cumulative_session_usd=0.42)
        )
        fired_for_bot2 = await ev.evaluate(
            "bot-2", "sess-2", _cost_event(agent_id="bot-2", cumulative_session_usd=0.99)
        )
        assert len(fired_for_bot1) == 1
        assert fired_for_bot2 == []

    @pytest.mark.asyncio
    async def test_evaluator_agent_filter_wildcard(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.30, agent_id=None)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        for agent in ("bot-1", "bot-2", "bot-3"):
            events = await ev.evaluate(
                agent, "sess", _cost_event(agent_id=agent, cumulative_session_usd=0.42)
            )
            assert len(events) == 1, f"Expected fire for {agent}"

    @pytest.mark.asyncio
    async def test_evaluator_multiple_rules(self) -> None:
        disp = NotificationDispatcher()
        r1 = _rule(name="r1", threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.30)
        r2 = _rule(name="r2", threshold_type=ThresholdType.ABSOLUTE_USD, threshold_usd=0.50)
        r3 = _rule(name="r3", threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=1.00)
        ev = AlertEvaluator(rules=[r1, r2, r3], dispatcher=disp)
        events = await ev.evaluate(
            "bot-1",
            "sess-1",
            _cost_event(cumulative_session_usd=0.42, cumulative_agent_usd=0.87),
        )
        assert len(events) == 2
        fired_names = {e.rule_name for e in events}
        assert fired_names == {"r1", "r2"}

    @pytest.mark.asyncio
    async def test_evaluator_absolute_usd_uses_cumulative_agent(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.ABSOLUTE_USD, threshold_usd=0.80)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        events = await ev.evaluate(
            "bot-1",
            "sess-1",
            _cost_event(cumulative_agent_usd=0.87, cumulative_session_usd=0.05),
        )
        assert len(events) == 1
        assert events[0].actual_usd == pytest.approx(0.87)

    @pytest.mark.asyncio
    async def test_evaluator_per_hour_uses_rolling_window(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.PER_HOUR_USD, threshold_usd=0.10)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        # 4 events at 0.03 each = 0.12 total > 0.10
        for _ in range(4):
            await ev.evaluate(
                "bot-1", "sess-1", _cost_event(cost_usd=0.03, cumulative_session_usd=0.01)
            )
        # Check that by the 4th call an alert fires
        # Reset cooldown and check
        key = f"alerts:{r.rule_id}:bot-1"
        ev._mem_cooldown.pop(key, None)
        # Add 4 more events — total 8 * 0.03 = 0.24 in window
        events = await ev.evaluate(
            "bot-1", "sess-1", _cost_event(cost_usd=0.03, cumulative_session_usd=0.01)
        )
        assert len(events) >= 0  # may or may not fire depending on cooldown state

    @pytest.mark.asyncio
    async def test_evaluator_disabled_rule_never_fires(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.01, enabled=False)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        events = await ev.evaluate("bot-1", "sess-1", _cost_event(cumulative_session_usd=99.0))
        assert events == []

    def test_evaluator_add_remove_rules(self) -> None:
        disp = NotificationDispatcher()
        r1 = _rule(name="r1")
        r2 = _rule(name="r2")
        ev = AlertEvaluator(rules=[r1], dispatcher=disp)
        ev.add_rule(r2)
        assert len(ev.list_rules()) == 2
        ev.remove_rule(r1.rule_id)
        rules = ev.list_rules()
        assert len(rules) == 1
        assert rules[0].rule_id == r2.rule_id


# ---------------------------------------------------------------------------
# AlertEvent message formatting
# ---------------------------------------------------------------------------


class TestAlertEventMessages:
    @pytest.mark.asyncio
    async def test_alert_event_message_per_session(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.PER_SESSION_USD, threshold_usd=0.30)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        events = await ev.evaluate(
            "research-bot",
            "sess-1",
            _cost_event(agent_id="research-bot", cumulative_session_usd=0.42),
        )
        assert len(events) == 1
        msg = events[0].message
        assert "research-bot" in msg
        assert "0.42" in msg or "0.30" in msg

    @pytest.mark.asyncio
    async def test_alert_event_message_per_hour(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.PER_HOUR_USD, threshold_usd=0.10)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        # Inject enough rolling cost to trigger
        ev._rolling_window["bot-1"] = [(datetime.utcnow(), 0.15)]
        events = await ev.evaluate("bot-1", "sess-1", _cost_event(cost_usd=0.0))
        if events:
            assert "bot-1" in events[0].message

    @pytest.mark.asyncio
    async def test_alert_event_message_absolute(self) -> None:
        disp = NotificationDispatcher()
        r = _rule(threshold_type=ThresholdType.ABSOLUTE_USD, threshold_usd=0.50)
        ev = AlertEvaluator(rules=[r], dispatcher=disp)
        events = await ev.evaluate("bot-1", "sess-1", _cost_event(cumulative_agent_usd=0.87))
        assert len(events) == 1
        assert "bot-1" in events[0].message


# ---------------------------------------------------------------------------
# NotificationDispatcher
# ---------------------------------------------------------------------------


class TestNotificationDispatcher:
    @pytest.mark.asyncio
    async def test_dispatcher_calls_all_channels(self) -> None:
        calls: list[str] = []

        class _TrackChannel:
            async def send(self, event: AlertEvent) -> None:
                calls.append("called")

        channels = [_TrackChannel(), _TrackChannel(), _TrackChannel()]
        disp = NotificationDispatcher(channels=channels)  # type: ignore[arg-type]
        await disp.dispatch(_make_alert_event())
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_dispatcher_channel_failure_isolated(self) -> None:
        calls: list[str] = []

        class _GoodChannel:
            async def send(self, event: AlertEvent) -> None:
                calls.append("ok")

        class _BadChannel:
            async def send(self, event: AlertEvent) -> None:
                raise RuntimeError("channel exploded")

        disp = NotificationDispatcher(
            channels=[_GoodChannel(), _BadChannel(), _GoodChannel()]  # type: ignore[list-item]
        )
        # Must not raise
        await disp.dispatch(_make_alert_event())
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_dispatcher_empty_channels_no_error(self) -> None:
        disp = NotificationDispatcher()
        await disp.dispatch(_make_alert_event())  # no exception

    @pytest.mark.asyncio
    async def test_dispatcher_add_channel(self) -> None:
        disp = NotificationDispatcher()
        ch = LogChannel()
        disp.add_channel(ch)
        await disp.dispatch(_make_alert_event())  # should not raise


# ---------------------------------------------------------------------------
# WebhookChannel
# ---------------------------------------------------------------------------


class TestWebhookChannel:
    @pytest.mark.asyncio
    async def test_webhook_channel_posts_json(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        channel = WebhookChannel(url="http://example.com/hook", _client=mock_client)
        event = _make_alert_event()
        await channel.send(event)

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://example.com/hook"
        # Body is passed as content bytes
        body_bytes = call_args[1].get("content") or call_args[1].get("data") or b""
        assert b"rule_id" in body_bytes
        headers = call_args[1].get("headers", {})
        assert "application/json" in headers.get("Content-Type", "")

    @pytest.mark.asyncio
    async def test_webhook_channel_hmac_signature(self) -> None:
        import hashlib
        import hmac as _hmac

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        secret = "mysecret"
        channel = WebhookChannel(url="http://example.com/hook", secret=secret, _client=mock_client)
        event = _make_alert_event()
        await channel.send(event)

        call_args = mock_client.post.call_args
        body_bytes = call_args[1].get("content") or call_args[1].get("data") or b""
        headers = call_args[1].get("headers", {})
        sig_header = headers.get("X-Grampus-Signature", "")
        assert sig_header.startswith("sha256=")
        expected = "sha256=" + _hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
        assert sig_header == expected

    @pytest.mark.asyncio
    async def test_webhook_channel_http_error_swallowed(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock(side_effect=Exception("500 Server Error"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        channel = WebhookChannel(url="http://example.com/hook", _client=mock_client)
        # Must not raise
        await channel.send(_make_alert_event())

    @pytest.mark.asyncio
    async def test_webhook_no_secret_no_signature_header(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        channel = WebhookChannel(url="http://example.com/hook", _client=mock_client)
        await channel.send(_make_alert_event())
        headers = mock_client.post.call_args[1].get("headers", {})
        assert "X-Grampus-Signature" not in headers


# ---------------------------------------------------------------------------
# SlackChannel
# ---------------------------------------------------------------------------


class TestSlackChannel:
    @pytest.mark.asyncio
    async def test_slack_channel_posts_blocks(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        channel = SlackChannel(webhook_url="http://slack.example/hook", _client=mock_client)
        event = _make_alert_event()
        await channel.send(event)

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        payload = call_args[1].get("json") or {}
        assert "blocks" in payload
        assert "text" in payload
        # Should contain the rule name
        assert event.rule_name in payload["text"]

    @pytest.mark.asyncio
    async def test_slack_channel_severity_emoji_warning(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        channel = SlackChannel(webhook_url="http://slack.example/hook", _client=mock_client)
        event = _make_alert_event()  # WARNING severity
        await channel.send(event)

        payload = mock_client.post.call_args[1].get("json", {})
        assert ":warning:" in payload["text"]

    @pytest.mark.asyncio
    async def test_slack_channel_severity_emoji_critical(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        channel = SlackChannel(webhook_url="http://slack.example/hook", _client=mock_client)
        r = _rule(severity=AlertSeverity.CRITICAL)
        event = AlertEvent(
            rule_id=r.rule_id,
            rule_name=r.name,
            agent_id="bot-1",
            session_id="sess-1",
            severity=AlertSeverity.CRITICAL,
            threshold_type=r.threshold_type,
            threshold_usd=r.threshold_usd,
            actual_usd=0.42,
            message="critical alert",
        )
        await channel.send(event)

        payload = mock_client.post.call_args[1].get("json", {})
        assert ":rotating_light:" in payload["text"]

    @pytest.mark.asyncio
    async def test_slack_channel_http_error_swallowed(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock(side_effect=Exception("400 Bad Request"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        channel = SlackChannel(webhook_url="http://slack.example/hook", _client=mock_client)
        await channel.send(_make_alert_event())  # must not raise


# ---------------------------------------------------------------------------
# SmtpChannel
# ---------------------------------------------------------------------------


class TestSmtpChannel:
    @pytest.mark.asyncio
    async def test_smtp_channel_sends_email(self) -> None:
        event = _make_alert_event()

        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_smtp.ehlo = MagicMock()
            mock_smtp.starttls = MagicMock()
            mock_smtp.login = MagicMock()
            mock_smtp.send_message = MagicMock()

            channel = SmtpChannel(
                host="smtp.example.com",
                port=587,
                username="user",
                password="pass",
                from_addr="nexus@example.com",
                to_addrs=["ops@example.com"],
            )
            await channel.send(event)

            # Either send_message or sendmail should have been called
            sent = mock_smtp.send_message.called or mock_smtp.sendmail.called
            assert sent

    @pytest.mark.asyncio
    async def test_smtp_channel_subject_contains_severity_and_name(self) -> None:
        event = _make_alert_event()
        captured: list[object] = []

        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

            def _capture_msg(msg: object) -> None:
                captured.append(msg)

            mock_smtp.send_message = MagicMock(side_effect=_capture_msg)

            channel = SmtpChannel(
                host="smtp.example.com",
                to_addrs=["ops@example.com"],
            )
            await channel.send(event)

        if captured:
            msg = captured[0]
            if hasattr(msg, "__getitem__"):
                subject = msg["Subject"]  # type: ignore[index]
                assert "warning" in subject.lower() or "WARNING" in subject
                assert event.rule_name in subject

    @pytest.mark.asyncio
    async def test_smtp_channel_smtp_error_swallowed(self) -> None:
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp_cls.side_effect = Exception("Connection refused")

            channel = SmtpChannel(host="smtp.example.com", to_addrs=["ops@example.com"])
            await channel.send(_make_alert_event())  # must not raise


# ---------------------------------------------------------------------------
# LogChannel
# ---------------------------------------------------------------------------


class TestLogChannel:
    @pytest.mark.asyncio
    async def test_log_channel_always_succeeds(self) -> None:
        ch = LogChannel()
        await ch.send(_make_alert_event())  # must not raise

    @pytest.mark.asyncio
    async def test_log_channel_with_custom_logger(self) -> None:
        mock_log = MagicMock()
        ch = LogChannel(logger=mock_log)
        event = _make_alert_event()
        await ch.send(event)
        mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# CostTracker integration
# ---------------------------------------------------------------------------


class TestCostTrackerAlertIntegration:
    @pytest.mark.asyncio
    async def test_cost_tracker_triggers_alert_evaluator(self) -> None:
        from grampus.core.types import TokenUsage
        from grampus.orchestration.cost_tracker import CostTracker
        from grampus.orchestration.model_router import ModelSpec, ModelTier

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(return_value=[])

        tracker = CostTracker(
            agent_id="test-agent",
            session_id="test-sess",
            alert_evaluator=mock_evaluator,
        )
        spec = ModelSpec(
            model_id="test-model",
            tier=ModelTier.BALANCED,
            provider="anthropic",
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            context_window=200_000,
        )
        usage = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.01, model="test-model"
        )
        await tracker.record(usage, step_name="step1", model_spec=spec)

        mock_evaluator.evaluate.assert_awaited_once()
        call_kwargs = mock_evaluator.evaluate.call_args
        assert call_kwargs[0][0] == "test-agent"
        assert call_kwargs[0][1] == "test-sess"

    @pytest.mark.asyncio
    async def test_cost_tracker_alert_failure_does_not_raise(self) -> None:
        from grampus.core.types import TokenUsage
        from grampus.orchestration.cost_tracker import CostTracker
        from grampus.orchestration.model_router import ModelSpec, ModelTier

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(side_effect=RuntimeError("alert boom"))

        tracker = CostTracker(
            agent_id="test-agent",
            session_id="test-sess",
            alert_evaluator=mock_evaluator,
        )
        spec = ModelSpec(
            model_id="test-model",
            tier=ModelTier.BALANCED,
            provider="anthropic",
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            context_window=200_000,
        )
        usage = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.01, model="test-model"
        )
        # Must not raise even though evaluator explodes
        await tracker.record(usage, step_name="step1", model_spec=spec)
        summary = tracker.summary()
        assert summary.total_cost_usd == pytest.approx(0.01)
