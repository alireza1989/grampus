"""Email sender tool via stdlib smtplib."""

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage
from typing import Any

from grampus.tools.library._base import err


def _send_sync(
    to: str,
    subject: str,
    body: str,
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    from_address: str,
    use_tls: bool,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as smtp:
        if use_tls:
            smtp.starttls()
        if username:
            smtp.login(username, password)
        smtp.send_message(msg)


async def send_email(
    to: str,
    subject: str,
    body: str,
    smtp_host: str = "localhost",
    smtp_port: int = 587,
    username: str = "",
    password: str = "",
    from_address: str = "grampus@localhost",
    use_tls: bool = True,
) -> dict[str, Any]:
    """Send an email via SMTP.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain-text email body.
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        username: SMTP auth username (empty = no auth).
        password: SMTP auth password.
        from_address: Sender address.
        use_tls: Whether to use STARTTLS.

    Returns:
        ``{"ok": True, "to": str, "subject": str}`` or error dict.
    """
    try:
        await asyncio.to_thread(
            _send_sync,
            to,
            subject,
            body,
            smtp_host,
            smtp_port,
            username,
            password,
            from_address,
            use_tls,
        )
    except smtplib.SMTPAuthenticationError as exc:
        return err(f"SMTP authentication failed: {exc}", code="SMTP_AUTH_ERROR")
    except smtplib.SMTPException as exc:
        return err(f"SMTP error: {exc}", code="SMTP_ERROR")
    except (ConnectionRefusedError, OSError) as exc:
        return err(f"Connection failed: {exc}", code="CONNECTION_ERROR")

    return {"ok": True, "to": to, "subject": subject}
