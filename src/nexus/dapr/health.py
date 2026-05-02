"""Dapr sidecar health checking with retry and backoff."""

from __future__ import annotations

import asyncio
import time

import httpx

from nexus.core.errors import DaprConnectionError
from nexus.core.logging import get_logger

_log = get_logger(__name__)

_HEALTHZ_PATH = "/v1.0/healthz"
_INITIAL_BACKOFF = 0.5
_MAX_BACKOFF = 8.0


async def is_sidecar_healthy(host: str, port: int) -> bool:
    """Return True if the Dapr sidecar responds with HTTP 200 on /v1.0/healthz."""
    url = f"http://{host}:{port}{_HEALTHZ_PATH}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=2.0)
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False


async def wait_for_sidecar(
    host: str,
    port: int,
    timeout_seconds: float = 30.0,
) -> None:
    """Poll the Dapr sidecar until healthy or timeout expires.

    Uses exponential backoff between retries (0.5s → 1s → 2s … capped at 8s).

    Raises:
        DaprConnectionError: If the sidecar is not healthy within timeout_seconds.
    """
    deadline = time.monotonic() + timeout_seconds
    backoff = _INITIAL_BACKOFF

    while True:
        if await is_sidecar_healthy(host, port):
            _log.info("dapr_sidecar_healthy", host=host, port=port)
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DaprConnectionError(
                f"Dapr sidecar at {host}:{port} did not become healthy within {timeout_seconds}s",
                code="DAPR_CONNECTION_ERROR",
                details={"host": host, "port": port, "timeout_seconds": timeout_seconds},
            )

        sleep_time = min(backoff, remaining, _MAX_BACKOFF)
        _log.debug("dapr_sidecar_not_ready", host=host, port=port, retry_in=sleep_time)
        await asyncio.sleep(sleep_time)
        backoff = min(backoff * 2, _MAX_BACKOFF)
