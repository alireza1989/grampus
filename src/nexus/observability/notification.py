"""Notification channel implementations and dispatcher for cost tracking alerts."""

from __future__ import annotations

import asyncio
import email.mime.text
import hashlib
import hmac as _hmac
import json
import smtplib
from typing import Any, Protocol

import structlog

from nexus.observability.alerts import AlertEvent, AlertSeverity

_log = structlog.get_logger("nexus.alerts")

_SEVERITY_EMOJI = {
    AlertSeverity.INFO: ":information_source:",
    AlertSeverity.WARNING: ":warning:",
    AlertSeverity.CRITICAL: ":rotating_light:",
}


class NotificationChannel(Protocol):
    """Protocol that all notification channel implementations must satisfy."""

    async def send(self, event: AlertEvent) -> None: ...


class WebhookChannel:
    """Sends alert events as JSON to an HTTP endpoint with optional HMAC signing.

    Args:
        url: Target webhook URL.
        secret: Optional shared secret. When set, adds X-Nexus-Signature header.
        _client: Optional pre-built httpx.AsyncClient for testing.
    """

    def __init__(
        self,
        url: str,
        secret: str | None = None,
        _client: Any | None = None,
    ) -> None:
        self._url = url
        self._secret = secret
        self._client = _client

    async def send(self, event: AlertEvent) -> None:
        body_bytes = json.dumps(event.model_dump(mode="json")).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._secret:
            sig = (
                "sha256=" + _hmac.new(self._secret.encode(), body_bytes, hashlib.sha256).hexdigest()
            )
            headers["X-Nexus-Signature"] = sig
        try:
            if self._client is not None:
                resp = await self._client.post(self._url, content=body_bytes, headers=headers)
                resp.raise_for_status()
            else:
                import httpx

                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(self._url, content=body_bytes, headers=headers)
                    resp.raise_for_status()
        except Exception as exc:
            _log.warning("webhook_channel_send_failed", url=self._url, error=str(exc))


class SlackChannel:
    """Sends alert events to a Slack incoming webhook with block kit formatting.

    Args:
        webhook_url: Slack incoming webhook URL.
        _client: Optional pre-built httpx.AsyncClient for testing.
    """

    def __init__(self, webhook_url: str, _client: Any | None = None) -> None:
        self._url = webhook_url
        self._client = _client

    async def send(self, event: AlertEvent) -> None:
        emoji = _SEVERITY_EMOJI.get(event.severity, ":bell:")
        payload = {
            "text": f"{emoji} *{event.rule_name}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": event.message},
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"Severity: *{event.severity}* | "
                                f"Agent: `{event.agent_id}` | "
                                f"Actual: ${event.actual_usd:.4f} | "
                                f"Limit: ${event.threshold_usd:.2f}"
                            ),
                        }
                    ],
                },
            ],
        }
        try:
            if self._client is not None:
                resp = await self._client.post(self._url, json=payload)
                resp.raise_for_status()
            else:
                import httpx

                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(self._url, json=payload)
                    resp.raise_for_status()
        except Exception as exc:
            _log.warning("slack_channel_send_failed", url=self._url, error=str(exc))


class SmtpChannel:
    """Sends alert events as MIME email via smtplib in a thread executor.

    Args:
        host: SMTP server hostname.
        port: SMTP port (587 triggers STARTTLS).
        username: Optional SMTP auth username.
        password: Optional SMTP auth password.
        from_addr: Sender address.
        to_addrs: List of recipient addresses.
    """

    def __init__(
        self,
        host: str,
        port: int = 587,
        username: str | None = None,
        password: str | None = None,
        from_addr: str = "nexus@localhost",
        to_addrs: list[str] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._from_addr = from_addr
        self._to_addrs: list[str] = to_addrs or []

    async def send(self, event: AlertEvent) -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._do_send, event)
        except Exception as exc:
            _log.warning("smtp_channel_send_failed", host=self._host, error=str(exc))

    def _do_send(self, event: AlertEvent) -> None:
        subject = f"[Nexus Alert][{event.severity.upper()}] {event.rule_name}"
        details = json.dumps(event.model_dump(mode="json"), indent=2)
        body = f"{event.message}\n\n{details}"
        msg = email.mime.text.MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = self._from_addr
        msg["To"] = ", ".join(self._to_addrs)

        with smtplib.SMTP(self._host, self._port) as smtp:
            if self._port != 25:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
            if self._username and self._password:
                smtp.login(self._username, self._password)
            smtp.send_message(msg)


class LogChannel:
    """Writes alert events to the structlog logger — always succeeds.

    Args:
        logger: Optional pre-built logger (defaults to nexus.alerts).
    """

    def __init__(self, logger: Any | None = None) -> None:
        self._log = logger if logger is not None else _log

    async def send(self, event: AlertEvent) -> None:
        self._log.warning(event.message, **event.model_dump(mode="json"))


class NotificationDispatcher:
    """Dispatches an AlertEvent to all registered channels concurrently.

    Individual channel failures are caught and logged — they never propagate.
    """

    def __init__(self, channels: list[NotificationChannel] | None = None) -> None:
        self._channels: list[NotificationChannel] = list(channels or [])

    def add_channel(self, channel: NotificationChannel) -> None:
        self._channels.append(channel)

    async def dispatch(self, event: AlertEvent) -> None:
        if not self._channels:
            return
        results = await asyncio.gather(
            *[ch.send(event) for ch in self._channels],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                _log.warning(
                    "notification_channel_failed",
                    channel=type(self._channels[i]).__name__,
                    error=str(result),
                )
