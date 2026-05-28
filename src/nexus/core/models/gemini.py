"""Google Gemini model client implementation."""

from __future__ import annotations

import uuid as _uuid
from collections.abc import AsyncIterator
from typing import Any

from nexus.core.errors import ModelError
from nexus.core.models.base import ModelClient, ModelResponse
from nexus.core.types import Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition

# Per-model pricing in USD per 1M tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    # Gemini 3.x (released May 2026)
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),  # ≤200k context tier
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    # Gemini 2.5
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    # Gemini 2.0
    "gemini-2.0-flash-001": (0.075, 0.30),
    "gemini-2.0-flash-lite-001": (0.0375, 0.15),
    "gemini-2.0-pro-exp": (0.0, 0.0),
    # Gemini 1.5
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-1.5-pro": (1.25, 5.00),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    key = model
    if key not in _PRICING:
        for k in _PRICING:
            if model.startswith(k):
                key = k
                break
    in_price, out_price = _PRICING.get(key, (0.075, 0.30))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def _to_gemini_contents(messages: list[Message]) -> list[dict[str, Any]]:
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                id_to_name[tc.id] = tc.name

    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == Role.SYSTEM:
            continue

        parts: list[dict[str, Any]] = []

        if msg.role in (Role.USER, Role.ASSISTANT):
            if msg.content:
                parts.append({"text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    # Include id for Gemini 3+ models (ignored by older models)
                    fc_dict: dict[str, Any] = {"name": tc.name, "args": tc.arguments, "id": tc.id}
                    parts.append({"function_call": fc_dict})

        elif msg.role == Role.TOOL:
            for tr in msg.tool_results:
                fn_name = id_to_name.get(tr.tool_call_id, tr.tool_call_id)
                output = str(tr.output) if tr.output is not None else (tr.error or "")
                # id must be echoed back for Gemini 3+ models
                parts.append(
                    {
                        "function_response": {
                            "name": fn_name,
                            "response": {"result": output},
                            "id": tr.tool_call_id,
                        }
                    }
                )

        if parts:
            gemini_role = "model" if msg.role == Role.ASSISTANT else "user"
            result.append({"role": gemini_role, "parts": parts})

    return result


def _to_gemini_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    declarations = []
    for tool in tools:
        schema = tool.to_function_schema()
        declarations.append(
            {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            }
        )
    return [{"function_declarations": declarations}]


def _stop_reason(finish_reason: Any) -> str:
    mapping = {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "safety",
        "RECITATION": "recitation",
        "MALFORMED_FUNCTION_CALL": "tool_use",
        "OTHER": "stop",
    }
    return mapping.get(str(finish_reason), "stop")


class GeminiClient(ModelClient):
    """ModelClient implementation backed by the Google GenAI SDK (google-genai).

    Args:
        api_key: Google AI API key. Set GOOGLE_API_KEY env var or pass explicitly.
        _client: Optional injected SDK client (for testing — pass a MagicMock).
    """

    def __init__(self, api_key: str, _client: Any = None) -> None:
        if _client is not None:
            self._sdk = _client
        else:
            from google import genai  # lazy import

            self._sdk = genai.Client(api_key=api_key)

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
        from google.genai import types as _types

        system = next((m.content for m in messages if m.role == Role.SYSTEM and m.content), None)
        contents = _to_gemini_contents(messages)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = _to_gemini_tools(tools)

        config = _types.GenerateContentConfig(**config_kwargs)

        try:
            response = await self._sdk.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            msg = str(exc).lower()
            hint = (
                "Set GOOGLE_API_KEY in your environment or pass api_key to GeminiClient."
                if any(
                    k in msg
                    for k in ("api_key", "401", "unauthorized", "invalid_argument", "credentials")
                )
                else "Check the Google AI status page or reduce max_tokens / request size."
            )
            raise ModelError(
                f"Gemini API error: {exc}",
                code="MODEL_API_ERROR",
                details={"provider": "gemini", "model": model},
                hint=hint,
            ) from exc

        candidate = response.candidates[0]
        in_t = response.usage_metadata.prompt_token_count
        out_t = response.usage_metadata.candidates_token_count
        tool_calls: list[ToolCall] = []
        content_text: str | None = None

        for part in candidate.content.parts:
            if getattr(part, "text", None):
                content_text = part.text
            fc = getattr(part, "function_call", None)
            if fc is not None:
                # Gemini 3+ returns a real id; older models don't — fall back to generated UUID
                sdk_id = getattr(fc, "id", None)
                call_id = (
                    sdk_id
                    if isinstance(sdk_id, str)
                    else f"gemini-{fc.name}-{_uuid.uuid4().hex[:8]}"
                )
                tool_calls.append(ToolCall(id=call_id, name=fc.name, arguments=dict(fc.args)))

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
            stop_reason=_stop_reason(candidate.finish_reason),
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
        from google.genai import types as _types

        system = next((m.content for m in messages if m.role == Role.SYSTEM and m.content), None)
        contents = _to_gemini_contents(messages)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = _to_gemini_tools(tools)

        config = _types.GenerateContentConfig(**config_kwargs)

        try:
            in_t = 0
            out_t = 0
            finish = "stop"

            async for chunk in await self._sdk.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                if chunk.usage_metadata:
                    in_t = chunk.usage_metadata.prompt_token_count or in_t
                    out_t = chunk.usage_metadata.candidates_token_count or out_t

                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if candidate.finish_reason:
                    finish = _stop_reason(candidate.finish_reason)

                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        yield StreamChunk(delta=part.text, model=model)

            yield StreamChunk(
                delta="",
                finish_reason=finish,
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
                "Set GOOGLE_API_KEY in your environment or pass api_key to GeminiClient."
                if any(
                    k in msg
                    for k in ("api_key", "401", "unauthorized", "invalid_argument", "credentials")
                )
                else "Check the Google AI status page or reduce max_tokens / request size."
            )
            raise ModelError(
                f"Gemini stream error: {exc}",
                code="MODEL_API_ERROR",
                details={"provider": "gemini", "model": model},
                hint=hint,
            ) from exc
