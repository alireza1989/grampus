"""Dapr Jobs API client — schedule recurring and one-shot agent runs."""

from __future__ import annotations

import json
from typing import Any

import httpx

from nexus.core.errors import DaprJobsError
from nexus.core.logging import get_logger

_log = get_logger(__name__)
_JOBS_PATH = "/v1.0-alpha1/jobs"


class DaprJobsClient:
    """HTTP client for the Dapr Jobs API (v1.0-alpha1).

    Args:
        host: Dapr sidecar host.
        port: Dapr sidecar HTTP port.
    """

    def __init__(self, host: str = "localhost", port: int = 3500) -> None:
        self._base = f"http://{host}:{port}{_JOBS_PATH}"

    async def schedule(
        self,
        name: str,
        *,
        cron: str,
        data: dict[str, Any],
        repeats: int = 0,
        ttl: str = "",
    ) -> None:
        """Create or update a scheduled job.

        Args:
            name: Unique job name (used as the callback path segment).
            cron: Schedule string — cron ("0 9 * * *") or interval ("@every 1h").
            data: Arbitrary payload delivered back to the app when the job fires.
            repeats: Max number of executions (0 = unlimited).
            ttl: Job time-to-live (e.g. "24h"). Empty = no expiry.

        Raises:
            DaprJobsError: On HTTP error from the sidecar.
        """
        body: dict[str, Any] = {
            "schedule": cron,
            "data": {
                "@type": "type.googleapis.com/google.protobuf.StringValue",
                "value": json.dumps(data),
            },
        }
        if repeats:
            body["repeats"] = repeats
        if ttl:
            body["ttl"] = ttl

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{self._base}/{name}", json=body)
            if resp.status_code not in (200, 204):
                raise DaprJobsError(
                    f"Failed to schedule job '{name}': HTTP {resp.status_code} — {resp.text}",
                    code="JOBS_SCHEDULE_FAILED",
                )
        _log.info("dapr_job_scheduled", name=name, cron=cron)

    async def get(self, name: str) -> dict[str, Any] | None:
        """Return job info dict or None if not found.

        Args:
            name: Job name to look up.

        Returns:
            Job info dict, or None if not found.

        Raises:
            DaprJobsError: On unexpected HTTP error from the sidecar.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self._base}/{name}")
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                raise DaprJobsError(
                    f"Failed to get job '{name}': HTTP {resp.status_code}",
                    code="JOBS_GET_FAILED",
                )
            result: dict[str, Any] = resp.json()
            return result

    async def delete(self, name: str) -> bool:
        """Delete a job.

        Args:
            name: Job name to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            DaprJobsError: On unexpected HTTP error from the sidecar.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{self._base}/{name}")
            if resp.status_code == 404:
                return False
            if resp.status_code not in (200, 204):
                raise DaprJobsError(
                    f"Failed to delete job '{name}': HTTP {resp.status_code}",
                    code="JOBS_DELETE_FAILED",
                )
        _log.info("dapr_job_deleted", name=name)
        return True
