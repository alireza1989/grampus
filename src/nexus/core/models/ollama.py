"""Ollama local model client implementation."""

from __future__ import annotations

import json as _json
from collections.abc import AsyncIterator
from typing import Any

from nexus.core.errors import ModelError
from nexus.core.models.base import ModelClient, ModelResponse
from nexus.core.types import Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition

# Ollama is local — no API cost. Prices are zero for all models.
_PRICING: dict[str, tuple[float, float]] = {
    "llama3.2": (0.0, 0.0),
    "llama3.1": (0.0, 0.0),
    "llama3.1:8b": (0.0, 0.0),
    "llama3.1:70b": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "mistral-nemo": (0.0, 0.0),
    "codellama": (0.0, 0.0),
    "qwen2.5": (0.0, 0.0),
    "qwen2.5-coder": (0.0, 0.0),
    "phi4": (0.0, 0.0),
    "deepseek-r1": (0.0, 0.0),
    "gemma3": (0.0, 0.0),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return 0.0


def _to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                id_to_name[tc.id] = tc.name

    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == Role.SYSTEM:
            result.append({"role": "system", "content": msg.content or ""})

        elif msg.role == Role.USER:
            result.append({"role": "user", "content": msg.content or ""})

        elif msg.role == Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)

        elif msg.role == Role.TOOL:
            for tr in msg.tool_results:
                tool_name = id_to_name.get(tr.tool_call_id, tr.tool_call_id)
                output = str(tr.output) if tr.output is not None else (tr.error or "")
                result.append({"role": "tool", "content": output, "tool_name": tool_name})

    return result


def _to_ollama_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": tool.to_function_schema()} for tool in tools]


def _stop_reason(done_reason: str | None) -> str:
    if done_reason is None:
        return "stop"
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "load": "stop",
        "unload": "stop",
    }
    return mapping.get(done_reason, "stop")


def _connection_hint(msg: str, model: str, host: str) -> str:
    if any(k in msg for k in ("connection", "refused", "not found", "404", "model")):
        return (
            f"Ensure Ollama is running: `ollama serve`. Pull the model with: `ollama pull {model}`."
        )
    return f"Check that Ollama is running at {host}"


class OllamaClient(ModelClient):
    """ModelClient implementation backed by the Ollama SDK for local LLMs.

    Connects to a locally running Ollama server. No API key required.
    Install Ollama from https://ollama.com and pull models with: ollama pull llama3.2

    Args:
        host: Ollama server URL. Defaults to http://localhost:11434.
        _client: Optional injected SDK client (for testing).
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        _client: Any = None,
    ) -> None:
        if _client is not None:
            self._sdk = _client
        else:
            import ollama  # lazy import

            self._sdk = ollama.AsyncClient(host=host)
        self._host = host

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
        ollama_messages = _to_ollama_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        if tools:
            call_kwargs["tools"] = _to_ollama_tools(tools)

        try:
            response = await self._sdk.chat(**call_kwargs)
        except Exception as exc:
            raise ModelError(
                f"Ollama API error: {exc}",
                code="MODEL_API_ERROR",
                details={"provider": "ollama", "model": model, "host": self._host},
                hint=_connection_hint(str(exc).lower(), model, self._host),
            ) from exc

        rmsg = response.message
        in_t: int = getattr(response, "prompt_eval_count", 0) or 0
        out_t: int = getattr(response, "eval_count", 0) or 0
        tool_calls: list[ToolCall] = []
        content_text: str | None = getattr(rmsg, "content", None) or None

        if getattr(rmsg, "tool_calls", None):
            for tc in rmsg.tool_calls:
                fn = tc.function
                raw_args = getattr(fn, "arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args: dict[str, Any] = _json.loads(raw_args)
                    except ValueError:
                        args = {}
                else:
                    args = dict(raw_args) if raw_args else {}
                tool_calls.append(
                    ToolCall(
                        id=f"ollama-{fn.name}-{id(tc):x}",
                        name=fn.name,
                        arguments=args,
                    )
                )

        done_reason: str = getattr(response, "done_reason", "stop") or "stop"

        return ModelResponse(
            content=content_text,
            tool_calls=tool_calls,
            token_usage=TokenUsage(
                input_tokens=in_t,
                output_tokens=out_t,
                total_tokens=in_t + out_t,
                cost_usd=0.0,
                model=model,
            ),
            model=model,
            stop_reason=_stop_reason(done_reason),
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
        ollama_messages = _to_ollama_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": True,
        }
        if tools:
            call_kwargs["tools"] = _to_ollama_tools(tools)

        try:
            in_t = 0
            out_t = 0
            finish = "end_turn"

            async for chunk in await self._sdk.chat(**call_kwargs):
                if getattr(chunk, "done", False):
                    in_t = getattr(chunk, "prompt_eval_count", 0) or 0
                    out_t = getattr(chunk, "eval_count", 0) or 0
                    finish = _stop_reason(getattr(chunk, "done_reason", "stop"))
                    continue

                delta = getattr(getattr(chunk, "message", None), "content", "") or ""
                if delta:
                    yield StreamChunk(delta=delta, model=model)

            yield StreamChunk(
                delta="",
                finish_reason=finish,
                token_usage=TokenUsage(
                    input_tokens=in_t,
                    output_tokens=out_t,
                    total_tokens=in_t + out_t,
                    cost_usd=0.0,
                    model=model,
                ),
                model=model,
                is_final=True,
            )
        except Exception as exc:
            raise ModelError(
                f"Ollama stream error: {exc}",
                code="MODEL_API_ERROR",
                details={"provider": "ollama", "model": model, "host": self._host},
                hint=_connection_hint(str(exc).lower(), model, self._host),
            ) from exc
