"""API-key validation helpers for A2A endpoints."""

from __future__ import annotations

from collections.abc import Callable

try:
    from fastapi import Request

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False


def make_api_key_verifier(api_key: str | None) -> Callable[..., None]:
    """Return a FastAPI dependency that validates a Bearer token.

    When ``api_key`` is None every request passes through (development mode).
    When set, requests missing or presenting a wrong ``Authorization: Bearer``
    header receive an HTTP 401 response.

    Args:
        api_key: Expected API key, or None to disable auth.

    Returns:
        A FastAPI dependency callable.
    """
    if not api_key:

        def _allow_all(request: Request) -> None:
            return None

        return _allow_all

    expected = f"Bearer {api_key}"

    def _check(request: Request) -> None:
        from fastapi import HTTPException

        auth = request.headers.get("Authorization", "")
        if auth != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return _check
