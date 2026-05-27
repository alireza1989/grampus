"""Webhook domain logic: config models, registry, HMAC verification, input extraction."""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import secrets
from typing import Any

from pydantic import BaseModel, Field


class WebhookConfig(BaseModel):
    """Configuration for a registered webhook trigger."""

    id: str = Field(default_factory=lambda: secrets.token_hex(8))
    name: str = ""
    secret: str = Field(default_factory=lambda: secrets.token_hex(32))
    input_template: str = ""
    input_field: str = ""
    async_mode: bool = False
    callback_url: str = ""
    session_prefix: str = "webhook"


class WebhookRegistry:
    """In-memory store of registered webhook configurations."""

    def __init__(self) -> None:
        self._configs: dict[str, WebhookConfig] = {}

    def register(self, config: WebhookConfig) -> WebhookConfig:
        """Register a webhook config and return it."""
        self._configs[config.id] = config
        return config

    def get(self, webhook_id: str) -> WebhookConfig | None:
        """Return the config for webhook_id, or None if not found."""
        return self._configs.get(webhook_id)

    def delete(self, webhook_id: str) -> bool:
        """Remove a webhook. Returns True if found and deleted, False if not found."""
        return self._configs.pop(webhook_id, None) is not None

    def list_all(self) -> list[WebhookConfig]:
        """Return all registered webhook configs."""
        return list(self._configs.values())


def verify_signature(raw_body: bytes, secret: str, signature_header: str | None) -> bool:
    """Verify X-Nexus-Signature header matches HMAC-SHA256(secret, body).

    Returns True when secret is empty (no verification configured).
    Returns False when secret is set but header is missing or invalid.
    """
    if not secret:
        return True
    if not signature_header:
        return False
    expected = "sha256=" + _hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return _hmac.compare_digest(expected, signature_header)


def extract_input(payload: dict[str, Any], config: WebhookConfig) -> str:
    """Extract agent input string from webhook payload.

    Priority: input_field > input_template > full JSON serialization.
    """
    if config.input_field:
        return _dot_get(payload, config.input_field)
    if config.input_template:
        return _render_template(config.input_template, payload)
    return json.dumps(payload, ensure_ascii=False)


def _dot_get(obj: Any, path: str) -> str:
    """Navigate dot-notation path. Returns empty string if path not found."""
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key, "")
        else:
            return ""
    return str(obj)


def _render_template(template: str, payload: dict[str, Any]) -> str:
    """Replace {{key}} and {{nested.key}} placeholders from payload."""
    result = template
    for key, value in _flatten_dict(payload).items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested dict to dot-notation keys: {"a": {"b": 1}} → {"a.b": "1"}."""
    items: dict[str, str] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, full_key))
        else:
            items[full_key] = str(v)
    return items
