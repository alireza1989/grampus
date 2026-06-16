"""A2AAgentClient — outbound calls to external A2A-compliant agents."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

try:
    import httpx
    from a2a.client.card_resolver import parse_agent_card
    from a2a.types.a2a_pb2 import (
        AgentCard,
        Task,
        TaskState,
    )
    from google.protobuf.json_format import ParseDict

    _HAS_A2A = True
except ImportError:  # pragma: no cover
    _HAS_A2A = False

from grampus.core.errors import OrchestrationError, ToolError
from grampus.core.logging import get_logger

_log = get_logger(__name__)

_TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}


def _require_sdk() -> None:
    if not _HAS_A2A:
        raise ToolError(
            "a2a-sdk is not installed. Install with: pip install 'grampus-ai[a2a]'",
            code="A2A_SDK_MISSING",
            hint="pip install 'grampus-ai[a2a]'",
        )


class _JsonRpcCall:
    """Minimal JSON-RPC 2.0 call helper."""

    @staticmethod
    def build(method: str, params: dict[str, Any], *, req_id: Any = None) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id if req_id is not None else str(uuid4()),
            "method": method,
            "params": params,
        }

    @staticmethod
    def result(response: dict[str, Any]) -> Any:
        if "error" in response:
            err = response["error"]
            raise OrchestrationError(
                f"A2A JSON-RPC error: {err}",
                code="A2A_CLIENT_ERROR",
            )
        return response.get("result")


def _parse_task(result_dict: dict[str, Any]) -> Task:
    """Parse a JSON-RPC task result dict into a Task proto."""
    task: Task = Task()
    ParseDict(result_dict, task, ignore_unknown_fields=True)
    return task


class A2AAgentClient:
    """Client for calling external A2A-compliant agents.

    Handles AgentCard discovery, message/send, message/stream, tasks/get.

    Args:
        base_url: Root URL of the remote agent (e.g. ``http://agent.example.com``).
        api_key: Bearer token to include in ``Authorization`` header, or None.
        timeout_seconds: Default HTTP timeout for non-streaming requests.
        poll_interval: Seconds between task status polls in wait_for_completion.
        _http_client: Injected httpx.AsyncClient for testing; a fresh client is
            created per request when None.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        poll_interval: float = 1.0,
        _http_client: Any | None = None,
    ) -> None:
        _require_sdk()
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._poll_interval = poll_interval
        self._injected_client: Any = _http_client
        self._cached_card: AgentCard | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if extra:
            headers.update(extra)
        return headers

    def _rpc_url(self) -> str:
        return f"{self._base_url}/a2a"

    async def _post_rpc(
        self,
        payload: dict[str, Any],
        http: Any,
    ) -> dict[str, Any]:
        resp = await http.post(
            self._rpc_url(),
            json=payload,
            headers=self._headers({"x-a2a-version": "1.0"}),
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise OrchestrationError(
                f"A2A remote error {resp.status_code} from {self._base_url}",
                code="A2A_CLIENT_ERROR",
                hint=f"Check that {self._base_url} is reachable and running.",
            )
        return resp.json()  # type: ignore[no-any-return]

    async def _get_http_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        return httpx.AsyncClient(timeout=self._timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_agent_card(self) -> AgentCard:
        """GET /.well-known/agent-card.json and parse into an AgentCard proto."""
        http = await self._get_http_client()
        try:
            resp = await http.get(
                f"{self._base_url}/.well-known/agent-card.json",
                headers=self._headers(),
                timeout=self._timeout,
            )
            if resp.status_code >= 400:
                raise OrchestrationError(
                    f"Failed to fetch AgentCard from {self._base_url}: HTTP {resp.status_code}",
                    code="A2A_CLIENT_ERROR",
                )
            card = parse_agent_card(resp.json())
            self._cached_card = card
            return card
        finally:
            if self._injected_client is None:
                await http.aclose()

    async def send_message(
        self,
        text: str,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> Any:
        """Send a message via message/send JSON-RPC and return the result."""
        message: dict[str, Any] = {
            "role": "user",
            "parts": [{"text": text}],
            "messageId": str(uuid4()),
        }
        if task_id:
            message["taskId"] = task_id
        if context_id:
            message["contextId"] = context_id

        payload = _JsonRpcCall.build("message/send", {"message": message})
        http = await self._get_http_client()
        try:
            response = await self._post_rpc(payload, http)
        finally:
            if self._injected_client is None:
                await http.aclose()

        return _JsonRpcCall.result(response)

    async def stream_message(
        self,
        text: str,
        task_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Send via message/stream and yield text delta strings as SSE arrives."""
        import json as _json

        message: dict[str, Any] = {
            "role": "user",
            "parts": [{"text": text}],
            "messageId": str(uuid4()),
        }
        if task_id:
            message["taskId"] = task_id

        payload = _JsonRpcCall.build("message/stream", {"message": message})

        http = await self._get_http_client()
        try:
            async with http.stream(
                "POST",
                self._rpc_url(),
                json=payload,
                headers=self._headers({"Accept": "text/event-stream", "x-a2a-version": "1.0"}),
                timeout=self._timeout,
            ) as resp:
                if resp.status_code >= 400:
                    raise OrchestrationError(
                        f"A2A stream error {resp.status_code}",
                        code="A2A_CLIENT_ERROR",
                    )
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        try:
                            event = _json.loads(raw)
                        except _json.JSONDecodeError:
                            continue
                        result = event.get("result", {})
                        status = result.get("status", {})
                        msg = status.get("message", {})
                        for part in msg.get("parts", []):
                            if "text" in part:
                                yield part["text"]
        finally:
            if self._injected_client is None:
                await http.aclose()

    async def get_task(self, task_id: str) -> Task:
        """Retrieve current task state via tasks/get JSON-RPC."""
        payload = _JsonRpcCall.build("tasks/get", {"id": task_id})
        http = await self._get_http_client()
        try:
            response = await self._post_rpc(payload, http)
        finally:
            if self._injected_client is None:
                await http.aclose()

        result = _JsonRpcCall.result(response)
        if not isinstance(result, dict):
            raise OrchestrationError(
                f"Unexpected tasks/get response: {result}",
                code="A2A_CLIENT_ERROR",
            )
        return _parse_task(result)

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a task via tasks/cancel JSON-RPC."""
        payload = _JsonRpcCall.build("tasks/cancel", {"id": task_id})
        http = await self._get_http_client()
        try:
            response = await self._post_rpc(payload, http)
        finally:
            if self._injected_client is None:
                await http.aclose()

        _JsonRpcCall.result(response)

    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float | None = None,
        timeout: float = 300.0,
    ) -> Task:
        """Poll tasks/get until the task reaches a terminal state or timeout.

        Args:
            task_id: ID of the task to wait for.
            poll_interval: Seconds between polls; defaults to constructor value.
            timeout: Maximum seconds to wait before raising.

        Raises:
            OrchestrationError: code="A2A_TIMEOUT" if deadline exceeded.
        """
        interval = poll_interval if poll_interval is not None else self._poll_interval
        deadline = time.monotonic() + timeout

        while True:
            if time.monotonic() >= deadline:
                raise OrchestrationError(
                    f"Timeout waiting for task '{task_id}' to complete after {timeout}s",
                    code="A2A_TIMEOUT",
                )
            task = await self.get_task(task_id)
            if task.status.state in _TERMINAL_STATES:
                return task
            await asyncio.sleep(interval)
