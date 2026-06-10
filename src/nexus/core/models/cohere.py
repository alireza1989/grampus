"""Cohere model client implementation (SDK v2, AsyncClientV2)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from nexus.core.errors import ModelError
from nexus.core.models.base import ModelClient, ModelResponse
from nexus.core.types import Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition

# Per-model pricing in USD per 1M tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    "command-a-03-2025": (2.50, 10.00),
    "command-r-plus-08-2024": (2.50, 10.00),
    "command-r-08-2024": (0.15, 0.60),
    "command-r7b-12-2024": (0.0375, 0.15),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _PRICING.get(model, (2.50, 10.00))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def _normalise_stop_reason(reason: str | None) -> str:
    mapping = {
        "COMPLETE": "end_turn",
        "TOOL_CALL": "tool_use",
        "MAX_TOKENS": "max_tokens",
        "ERROR": "error",
    }
    return mapping.get(reason or "", "end_turn")


def _to_cohere_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == Role.TOOL:
            for tr in msg.tool_results:
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": str(tr.output) if tr.output is not None else (tr.error or ""),
                    }
                )
        elif msg.tool_calls:
            result.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
        else:
            result.append({"role": msg.role.value, "content": msg.content or ""})
    return result


def _to_cohere_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": tool.to_function_schema()} for tool in tools]


def _parse_response(response: Any, model: str) -> ModelResponse:
    in_t = response.usage.billed_units.input_tokens
    out_t = response.usage.billed_units.output_tokens
    content_text: str | None = None
    tool_calls: list[ToolCall] = []

    for item in response.message.content:
        if item.type == "text":
            content_text = item.text
            break

    for tc in response.message.tool_calls or []:
        tool_calls.append(
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
        )

    return ModelResponse(
        content=content_text,
        tool_calls=tool_calls,
        token_usage=TokenUsage(
            input_tokens=in_t,
            output_tokens=out_t,
            total_tokens=in_t + out_t,
            cost_usd=_cost(model, in_t, out_t),
            model=model,
        ),
        model=model,
        stop_reason=_normalise_stop_reason(response.finish_reason),
    )


def _cohere_error(exc: Exception, context: str, model: str) -> ModelError:
    msg = str(exc).lower()
    hint = (
        "Set COHERE_API_KEY in your environment or in nexus.yaml under model.api_key."
        if "auth" in msg or "unauthorized" in msg or "api_key" in msg or "401" in msg
        else "Check the Cohere status page or reduce max_tokens / request size."
    )
    return ModelError(
        f"Cohere {context}: {exc}",
        code="MODEL_API_ERROR",
        details={"provider": "cohere", "model": model},
        hint=hint,
    )


class CohereClient(ModelClient):
    """ModelClient implementation backed by the Cohere SDK v2 (AsyncClientV2)."""

    def __init__(self, api_key: str, _client: Any = None) -> None:
        if _client is not None:
            self._sdk = _client
        else:
            import cohere  # lazy import

            self._sdk = cohere.AsyncClientV2(api_key=api_key)

    async def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send a completion request to the Cohere v2 API."""
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": _to_cohere_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            call_kwargs["tools"] = _to_cohere_tools(tools)

        try:
            response = await self._sdk.chat(**call_kwargs)
        except Exception as exc:
            raise _cohere_error(exc, "API error", model) from exc

        return _parse_response(response, model)

    async def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream tokens from the Cohere v2 chat_stream API."""
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": _to_cohere_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            call_kwargs["tools"] = _to_cohere_tools(tools)

        try:
            async for event in self._sdk.chat_stream(**call_kwargs):
                if event.type == "content-delta":
                    try:
                        text = event.delta.message.content[0].text
                        if text:
                            yield StreamChunk(delta=text, model=model)
                    except (AttributeError, IndexError, TypeError):
                        pass

                elif event.type == "message-end":
                    try:
                        bu = event.delta.usage.billed_units
                        in_t = bu.input_tokens or 0
                        out_t = bu.output_tokens or 0
                        finish_reason = _normalise_stop_reason(
                            getattr(event.delta, "finish_reason", None)
                        )
                    except AttributeError:
                        in_t, out_t, finish_reason = 0, 0, "end_turn"

                    yield StreamChunk(
                        delta="",
                        finish_reason=finish_reason,
                        token_usage=TokenUsage(
                            input_tokens=in_t,
                            output_tokens=out_t,
                            total_tokens=in_t + out_t,
                            cost_usd=_cost(model, in_t, out_t),
                            model=model,
                        ),
                        model=model,
                        is_final=True,
                    )
        except Exception as exc:
            raise _cohere_error(exc, "stream error", model) from exc
