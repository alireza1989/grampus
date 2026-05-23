"""Anthropic model client implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from nexus.core.errors import ModelError
from nexus.core.models.base import ModelClient, ModelResponse
from nexus.core.types import Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition

# Per-model pricing in USD per 1M tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _PRICING.get(model, (3.00, 15.00))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result = []
    for msg in messages:
        if msg.role == Role.SYSTEM:
            continue
        if msg.role == Role.TOOL:
            for tr in msg.tool_results:
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.tool_call_id,
                                "content": str(tr.output)
                                if tr.output is not None
                                else (tr.error or ""),
                            }
                        ],
                    }
                )
        elif msg.tool_calls:
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            result.append({"role": "assistant", "content": content})
        else:
            result.append({"role": msg.role.value, "content": msg.content or ""})
    return result


def _to_anthropic_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        schema = tool.to_function_schema()
        result.append(
            {
                "name": schema["name"],
                "description": schema["description"],
                "input_schema": schema["parameters"],
            }
        )
    return result


class AnthropicClient(ModelClient):
    """ModelClient implementation backed by the Anthropic SDK."""

    def __init__(self, api_key: str, _client: Any = None) -> None:
        if _client is not None:
            self._sdk = _client
        else:
            import anthropic  # lazy import

            self._sdk = anthropic.AsyncAnthropic(api_key=api_key)

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
        system = next((m.content for m in messages if m.role == Role.SYSTEM and m.content), None)
        anthropic_messages = _to_anthropic_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = _to_anthropic_tools(tools)

        try:
            response = await self._sdk.messages.create(**call_kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            hint = (
                "Set ANTHROPIC_API_KEY in your environment or in nexus.yaml under model.api_key."
                if "auth" in msg or "unauthorized" in msg or "api_key" in msg or "401" in msg
                else "Check the Anthropic status page or reduce max_tokens / request size."
            )
            raise ModelError(
                f"Anthropic API error: {exc}",
                code="MODEL_API_ERROR",
                details={"provider": "anthropic", "model": model},
                hint=hint,
            ) from exc

        in_t = response.usage.input_tokens
        out_t = response.usage.output_tokens
        tool_calls: list[ToolCall] = []
        content_text: str | None = None

        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
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
            model=response.model,
            stop_reason=response.stop_reason or "end_turn",
        )

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
        """Stream tokens from the Anthropic Messages API.

        Uses messages.stream() context manager. Final chunk has is_final=True
        and token_usage populated. Tool calls appear as finish_reason="tool_use".
        """
        system = next((m.content for m in messages if m.role == Role.SYSTEM and m.content), None)
        anthropic_messages = _to_anthropic_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = _to_anthropic_tools(tools)

        try:
            async with self._sdk.messages.stream(**call_kwargs) as stream_ctx:
                async for text in stream_ctx.text_stream:
                    yield StreamChunk(delta=text, model=model)
                final = await stream_ctx.get_final_message()
                in_t = final.usage.input_tokens
                out_t = final.usage.output_tokens
                yield StreamChunk(
                    delta="",
                    finish_reason=final.stop_reason,
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
            msg = str(exc).lower()
            hint = (
                "Set ANTHROPIC_API_KEY in your environment or in nexus.yaml under model.api_key."
                if "auth" in msg or "unauthorized" in msg or "api_key" in msg or "401" in msg
                else "Check the Anthropic status page or reduce max_tokens / request size."
            )
            raise ModelError(
                f"Anthropic stream error: {exc}",
                code="MODEL_API_ERROR",
                details={"provider": "anthropic", "model": model},
                hint=hint,
            ) from exc
