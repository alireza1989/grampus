"""Tests for nexus.tools.mcp_client — MCPClient."""

from __future__ import annotations

import httpx

from nexus.core.types import ToolDefinition, ToolResult
from nexus.tools.mcp_client import MCPClient

SERVER_URL = "http://mcp-server"


def _make_transport(
    responses: list[dict],
) -> httpx.MockTransport:
    """Build an httpx MockTransport that returns JSON responses in order."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        payload = responses[call_count % len(responses)]
        call_count += 1
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _list_tools_response(tools: list[dict]) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}}


def _call_response(content: list[dict]) -> dict:
    return {"jsonrpc": "2.0", "id": 2, "result": {"content": content}}


def _error_response(message: str) -> dict:
    return {"jsonrpc": "2.0", "id": 2, "error": {"code": -32000, "message": message}}


def _make_client(*responses: dict) -> MCPClient:
    transport = _make_transport(list(responses))
    http = httpx.AsyncClient(transport=transport)
    return MCPClient(SERVER_URL, _http_client=http)


class TestListTools:
    async def test_parses_tool_definitions(self) -> None:
        client = _make_client(
            _list_tools_response(
                [
                    {
                        "name": "search",
                        "description": "Web search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            },
                            "required": ["query"],
                        },
                    }
                ]
            )
        )
        defs = await client.list_tools()
        assert len(defs) == 1
        assert isinstance(defs[0], ToolDefinition)
        assert defs[0].name == "search"

    async def test_parses_multiple_tools(self) -> None:
        tools = [
            {
                "name": f"tool_{i}",
                "description": f"Tool {i}",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            }
            for i in range(3)
        ]
        client = _make_client(_list_tools_response(tools))
        defs = await client.list_tools()
        assert len(defs) == 3

    async def test_returns_empty_list_when_no_tools(self) -> None:
        client = _make_client(_list_tools_response([]))
        defs = await client.list_tools()
        assert defs == []

    async def test_maps_input_schema_properties_to_parameters(self) -> None:
        client = _make_client(
            _list_tools_response(
                [
                    {
                        "name": "calc",
                        "description": "Calculator",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer", "description": "First operand"},
                                "y": {"type": "integer", "description": "Second operand"},
                            },
                            "required": ["x"],
                        },
                    }
                ]
            )
        )
        defs = await client.list_tools()
        params = {p.name: p for p in defs[0].parameters}
        assert "x" in params
        assert params["x"].type == "integer"
        assert params["x"].required is True
        assert "y" in params
        assert params["y"].required is False

    async def test_posts_to_tools_list_path(self) -> None:
        captured_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(200, json=_list_tools_response([]))

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = MCPClient(SERVER_URL, _http_client=http)
        await client.list_tools()
        assert any("/tools/list" in url for url in captured_urls)


class TestInvoke:
    async def test_returns_tool_result_with_output(self) -> None:
        client = _make_client(_call_response([{"type": "text", "text": "42"}]))
        result = await client.invoke("add", {"a": 1, "b": 2})
        assert isinstance(result, ToolResult)
        assert result.output == "42"
        assert result.error is None

    async def test_joins_multiple_content_items(self) -> None:
        client = _make_client(
            _call_response([{"type": "text", "text": "Hello"}, {"type": "text", "text": " World"}])
        )
        result = await client.invoke("greet", {})
        assert result.output == "Hello World"

    async def test_sets_error_on_jsonrpc_error_response(self) -> None:
        client = _make_client(_error_response("Tool not found"))
        result = await client.invoke("missing", {})
        assert result.error is not None
        assert "Tool not found" in result.error
        assert result.output is None

    async def test_sets_error_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = MCPClient(SERVER_URL, _http_client=http)
        result = await client.invoke("tool", {})
        assert result.error is not None
        assert result.output is None

    async def test_sets_error_on_connect_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = MCPClient(SERVER_URL, _http_client=http)
        result = await client.invoke("tool", {})
        assert result.error is not None
        assert result.output is None

    async def test_duration_ms_is_non_negative(self) -> None:
        client = _make_client(_call_response([{"type": "text", "text": "ok"}]))
        result = await client.invoke("tool", {})
        assert result.duration_ms >= 0

    async def test_posts_to_tools_call_path(self) -> None:
        captured_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(200, json=_call_response([{"type": "text", "text": "ok"}]))

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = MCPClient(SERVER_URL, _http_client=http)
        await client.invoke("tool", {})
        assert any("/tools/call" in url for url in captured_urls)

    async def test_tool_call_id_empty_string(self) -> None:
        client = _make_client(_call_response([{"type": "text", "text": "result"}]))
        result = await client.invoke("tool", {})
        assert result.tool_call_id == ""


class TestContextManager:
    async def test_aenter_returns_client(self) -> None:
        client = _make_client()
        async with client as c:
            assert c is client

    async def test_aexit_calls_close(self) -> None:
        closed = False

        async def _close() -> None:
            nonlocal closed
            closed = True

        client = _make_client()
        client.close = _close  # type: ignore[method-assign]
        async with client:
            pass
        assert closed
