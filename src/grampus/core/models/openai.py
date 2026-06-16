"""OpenAI model client implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from grampus.core.errors import ModelError
from grampus.core.models.base import ModelClient, ModelResponse
from grampus.core.types import Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition

# Per-model pricing in USD per 1M tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    base = model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model
    in_price, out_price = _PRICING.get(model, _PRICING.get(base, (2.50, 10.00)))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
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
            oai_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
            result.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": oai_tool_calls,
                }
            )
        else:
            result.append({"role": msg.role.value, "content": msg.content or ""})
    return result


def _to_openai_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": tool.to_function_schema()} for tool in tools]


class OpenAIClient(ModelClient):
    """ModelClient implementation backed by the OpenAI SDK."""

    def __init__(self, api_key: str, _client: Any = None) -> None:
        if _client is not None:
            self._sdk = _client
        else:
            import openai  # lazy import

            self._sdk = openai.AsyncOpenAI(api_key=api_key)

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
        oai_messages = _to_openai_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            call_kwargs["tools"] = _to_openai_tools(tools)

        try:
            response = await self._sdk.chat.completions.create(**call_kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            hint = (
                "Set OPENAI_API_KEY in your environment or in grampus.yaml under model.api_key."
                if "auth" in msg or "unauthorized" in msg or "api_key" in msg or "401" in msg
                else "Check the OpenAI status page or reduce max_tokens / request size."
            )
            raise ModelError(
                f"OpenAI API error: {exc}",
                code="MODEL_API_ERROR",
                details={"provider": "openai", "model": model},
                hint=hint,
            ) from exc

        choice = response.choices[0]
        in_t = response.usage.prompt_tokens
        out_t = response.usage.completion_tokens
        tool_calls: list[ToolCall] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return ModelResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            token_usage=TokenUsage(
                input_tokens=in_t,
                output_tokens=out_t,
                total_tokens=response.usage.total_tokens,
                cost_usd=_cost(model, in_t, out_t),
                model=model,
            ),
            model=response.model,
            stop_reason=choice.finish_reason or "stop",
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
        """Stream tokens from the OpenAI Chat Completions API.

        Uses chat.completions.stream() context manager. Final chunk has
        is_final=True and token_usage from get_final_completion().
        """
        oai_messages = _to_openai_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            call_kwargs["tools"] = _to_openai_tools(tools)

        try:
            async with self._sdk.chat.completions.stream(**call_kwargs) as stream_ctx:
                async for chunk in stream_ctx:
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice and choice.delta.content:
                        yield StreamChunk(delta=choice.delta.content, model=model)
                final = await stream_ctx.get_final_completion()
                choice = final.choices[0]
                in_t = final.usage.prompt_tokens
                out_t = final.usage.completion_tokens
                yield StreamChunk(
                    delta="",
                    finish_reason=choice.finish_reason,
                    token_usage=TokenUsage(
                        input_tokens=in_t,
                        output_tokens=out_t,
                        total_tokens=final.usage.total_tokens,
                        cost_usd=_cost(model, in_t, out_t),
                        model=model,
                    ),
                    model=model,
                    is_final=True,
                )
        except Exception as exc:
            msg = str(exc).lower()
            hint = (
                "Set OPENAI_API_KEY in your environment or in grampus.yaml under model.api_key."
                if "auth" in msg or "unauthorized" in msg or "api_key" in msg or "401" in msg
                else "Check the OpenAI status page or reduce max_tokens / request size."
            )
            raise ModelError(
                f"OpenAI stream error: {exc}",
                code="MODEL_API_ERROR",
                details={"provider": "openai", "model": model},
                hint=hint,
            ) from exc
