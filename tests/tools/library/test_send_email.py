"""Tests for nexus.tools.library.send_email."""

from __future__ import annotations

import smtplib
from unittest.mock import MagicMock, patch

from nexus.tools.library.send_email import send_email


class TestSendEmail:
    async def test_send_email_returns_ok(self) -> None:
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp):
            result = await send_email(
                to="recipient@example.com",
                subject="Test Subject",
                body="Hello body",
                smtp_host="localhost",
                smtp_port=587,
                use_tls=False,
            )
        assert result["ok"] is True
        assert result["to"] == "recipient@example.com"
        assert result["subject"] == "Test Subject"

    async def test_send_email_returns_to_and_subject(self) -> None:
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp):
            result = await send_email(
                to="user@test.com",
                subject="Hello",
                body="World",
                use_tls=False,
            )
        assert result["to"] == "user@test.com"
        assert result["subject"] == "Hello"

    async def test_smtp_auth_failure_returns_err(self) -> None:
        with patch(
            "smtplib.SMTP",
            side_effect=smtplib.SMTPAuthenticationError(535, b"Auth failed"),
        ):
            result = await send_email(
                to="user@test.com",
                subject="Hello",
                body="World",
                use_tls=False,
            )
        assert result["ok"] is False
        assert "error" in result

    async def test_smtp_connection_refused_returns_err(self) -> None:
        with patch(
            "smtplib.SMTP",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            result = await send_email(
                to="user@test.com",
                subject="Hello",
                body="World",
                use_tls=False,
            )
        assert result["ok"] is False
        assert "error" in result

    async def test_does_not_raise(self) -> None:
        with patch(
            "smtplib.SMTP",
            side_effect=OSError("network error"),
        ):
            result = await send_email(
                to="user@test.com",
                subject="Hello",
                body="World",
                use_tls=False,
            )
        assert result["ok"] is False
