"""MCP (Model Context Protocol) client — JSON-RPC 2.0 over HTTP."""

from __future__ import annotations

import time
from typing import Any

import httpx

from nexus.core.logging import get_logger
from nexus.core.types import ToolDefinition, ToolParameter, ToolResult

logger = get_logger(__name__)


class MCPClient:
    """Client for an MCP-compatible tool server.

    Communicates via JSON-RPC 2.0 over HTTP.  The server must expose:

    * ``POST /tools/list`` — list available tools
    * ``POST /tools/call``  — invoke a tool

    Args:
        server_url: Base URL of the MCP server (no trailing slash).
        timeout_seconds: HTTP request timeout in seconds.
        _http_client: Optional pre-built AsyncClient (for testing).
    """

    def __init__(
        self,
        server_url: str,
        *,
        timeout_seconds: float = 30.0,
        _http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout_seconds
        self._http = _http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def list_tools(self) -> list[ToolDefinition]:
        """Discover tools available on the MCP server.

        Returns:
            List of ToolDefinition objects parsed from the server response.
        """
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        try:
            response = await self._http.post(
                f"{self._server_url}/tools/list",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.warning("mcp.list_tools.error", error=str(exc))
            return []

        result = data.get("result", {})
        raw_tools: list[dict[str, Any]] = result.get("tools", [])
        return [self._parse_tool_definition(t) for t in raw_tools]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a tool on the MCP server.

        Args:
            name: Tool name as registered on the server.
            arguments: Key/value arguments to pass to the tool.

        Returns:
            ToolResult with output on success or error message on failure.
        """
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        started = time.monotonic()
        try:
            response = await self._http.post(
                f"{self._server_url}/tools/call",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning("mcp.invoke.http_error", tool=name, error=str(exc))
            return ToolResult(tool_call_id="", output=None, error=str(exc), duration_ms=duration_ms)

        duration_ms = int((time.monotonic() - started) * 1000)

        if "error" in data:
            error_msg: str = data["error"].get("message", "MCP error")
            logger.warning("mcp.invoke.rpc_error", tool=name, error=error_msg)
            return ToolResult(
                tool_call_id="", output=None, error=error_msg, duration_ms=duration_ms
            )

        content: list[dict[str, Any]] = data.get("result", {}).get("content", [])
        output = "".join(item.get("text", "") for item in content if item.get("type") == "text")
        logger.debug("mcp.invoke.success", tool=name, duration_ms=duration_ms)
        return ToolResult(tool_call_id="", output=output, error=None, duration_ms=duration_ms)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> MCPClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @staticmethod
    def _parse_tool_definition(raw: dict[str, Any]) -> ToolDefinition:
        """Convert a raw MCP tool descriptor to a ToolDefinition."""
        schema: dict[str, Any] = raw.get("inputSchema", {})
        properties: dict[str, Any] = schema.get("properties", {})
        required_names: list[str] = schema.get("required", [])

        parameters: list[ToolParameter] = []
        for param_name, prop in properties.items():
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=prop.get("type", "string"),
                    description=prop.get("description", ""),
                    required=param_name in required_names,
                )
            )

        return ToolDefinition(
            name=raw["name"],
            description=raw.get("description", ""),
            parameters=parameters,
        )
