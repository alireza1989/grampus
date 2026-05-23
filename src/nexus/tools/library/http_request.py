"""HTTP request tool using httpx."""

from __future__ import annotations

from typing import Any

import httpx

from nexus.tools.library._base import err

_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_BODY_TRUNCATE_CHARS = 10_000


def _make_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


async def http_request(
    url: str,
    method: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Make an HTTP request and return the response.

    Args:
        url: Target URL.
        method: HTTP method — one of GET, POST, PUT, PATCH, DELETE.
        headers: Optional request headers.
        body: Optional JSON body for POST/PUT/PATCH.
        timeout_seconds: Request timeout.

    Returns:
        ``{"ok": True, "status_code": int, "body": ..., "headers": {...}}`` or error dict.
    """
    method_upper = method.upper()
    if method_upper not in _ALLOWED_METHODS:
        return err(
            f"Invalid method {method!r}. Allowed: {sorted(_ALLOWED_METHODS)}",
            code="INVALID_METHOD",
        )

    try:
        async with _make_client(timeout_seconds) as client:
            response = await client.request(
                method=method_upper,
                url=url,
                headers=headers or {},
                json=body if body and method_upper in {"POST", "PUT", "PATCH"} else None,
            )
    except httpx.InvalidURL as exc:
        return err(f"Invalid URL: {exc}", code="INVALID_URL")
    except httpx.TimeoutException as exc:
        return err(f"Request timed out: {exc}", code="TIMEOUT")
    except httpx.HTTPError as exc:
        return err(f"HTTP error: {exc}", code="HTTP_ERROR")

    content_type = response.headers.get("content-type", "")
    raw_text = response.text

    if "application/json" in content_type:
        try:
            parsed_body: Any = response.json()
        except Exception:
            parsed_body = raw_text[:_BODY_TRUNCATE_CHARS]
    else:
        parsed_body = raw_text[:_BODY_TRUNCATE_CHARS]

    return {
        "ok": True,
        "status_code": response.status_code,
        "body": parsed_body,
        "headers": dict(response.headers),
    }
