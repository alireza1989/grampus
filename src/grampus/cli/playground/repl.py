"""Interactive REPL and compare runner for the playground."""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from grampus.cli.playground.model_factory import make_client
from grampus.cli.playground.renderer import Renderer
from grampus.cli.playground.session import _SESSIONS_DIR, PlaygroundSession, PlaygroundTurn
from grampus.core.config import GrampusConfig
from grampus.core.models.base import ModelClient
from grampus.core.types import Message, Role, TokenUsage
from grampus.evaluation.prompt_versions import PromptVersionManager

_MODEL_NAMES = [
    # Anthropic
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    # OpenAI
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1",
    "o3-mini",
    # Gemini
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-001",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    # Ollama (local)
    "llama3.2",
    "llama3.1",
    "llama3.1:8b",
    "llama3.1:70b",
    "mistral",
    "mistral-nemo",
    "codellama",
    "qwen2.5",
    "qwen2.5-coder",
    "phi4",
    "deepseek-r1",
    "gemma3",
]


class CompareResult(BaseModel):
    """Result from one model in a multi-model compare run."""

    model: str
    output: str
    token_usage: TokenUsage | None
    duration_seconds: float
    error: str | None = None


@dataclass
class _ReplState:
    """All mutable state for one REPL session."""

    session: PlaygroundSession
    active_model: str
    client: ModelClient
    renderer: Renderer
    history: list[Message]
    config: GrampusConfig
    version_manager: PromptVersionManager
    sessions_dir: Path = field(default_factory=lambda: _SESSIONS_DIR)


def _prompt_str(model: str) -> str:
    return f"[{model}] > "


async def _send_message(text: str, state: _ReplState) -> None:
    """Append user message, stream LLM response, and record the turn."""
    state.history.append(Message(role=Role.USER, content=text))

    start = time.monotonic()
    print(state.renderer.model_header(state.active_model))

    output_parts: list[str] = []
    usage: TokenUsage | None = None

    try:
        async for chunk in state.client.stream(
            messages=state.history,
            model=state.active_model,
        ):
            if chunk.is_final:
                usage = chunk.token_usage
            else:
                output_parts.append(chunk.delta)
                sys.stdout.write(chunk.delta)
                sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write("\n")
        print(state.renderer.error(f"Error: {exc}"))
        # Remove the optimistically appended user message on failure
        if state.history and state.history[-1].role == Role.USER:
            state.history.pop()
        return

    elapsed = time.monotonic() - start
    sys.stdout.write("\n")
    print(state.renderer.model_footer(usage, elapsed))

    assistant_output = "".join(output_parts)
    state.history.append(Message(role=Role.ASSISTANT, content=assistant_output))

    turn = PlaygroundTurn(
        user_input=text,
        assistant_output=assistant_output,
        model=state.active_model,
        token_usage=usage,
        duration_seconds=elapsed,
    )
    state.session.turns.append(turn)


async def _handle_command(line: str, state: _ReplState) -> bool:
    """Dispatch a /command. Returns False to signal the REPL should exit."""
    parts = line.split(None, 1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/exit", "/quit"):
        print(state.renderer.info("Goodbye!"))
        return False

    if command == "/help":
        print(state.renderer.help_text())

    elif command == "/model":
        if not arg:
            print(state.renderer.error("Usage: /model <name>"))
            return True
        state.active_model = arg
        state.client = make_client(state.active_model, state.config)
        state.session.model = state.active_model
        print(state.renderer.success(f"Switched to {state.active_model}"))

    elif command == "/models":
        for name in _MODEL_NAMES:
            print(f"  {name}")

    elif command == "/system":
        if not arg:
            print(state.renderer.error("Usage: /system <text> or /system file:<path>"))
            return True
        if arg.startswith("file:"):
            path = Path(arg[5:])
            if not path.exists():
                print(state.renderer.error(f"File not found: {path}"))
                return True
            text = path.read_text()
        else:
            text = arg
        state.session.system_prompt = text
        state.history = [m for m in state.history if m.role != Role.SYSTEM]
        state.history.insert(0, Message(role=Role.SYSTEM, content=text))
        print(state.renderer.success("System prompt updated"))

    elif command == "/compare":
        if not arg:
            print(state.renderer.error("Usage: /compare <model2> [model3...]"))
            return True
        last_user = next((m.content for m in reversed(state.history) if m.role == Role.USER), None)
        if not last_user:
            print(state.renderer.error("No user message found — send a message first."))
            return True
        other_models = arg.split()
        all_models = [state.active_model] + other_models
        compare_msgs = [m for m in state.history if m.role in (Role.SYSTEM, Role.USER)]
        results = await run_compare(
            user_message=last_user,
            models=all_models,
            system_prompt=state.session.system_prompt,
            config=state.config,
            _messages=compare_msgs,
        )
        print(state.renderer.comparison_header(all_models))
        for r in results:
            print(state.renderer.model_header(r.model))
            if r.error:
                print(state.renderer.error(r.error))
            else:
                print(r.output)
            print(state.renderer.model_footer(r.token_usage, r.duration_seconds))

    elif command == "/cost":
        print(state.renderer.cost_summary(state.session))

    elif command == "/reset":
        state.history = []
        if state.session.system_prompt:
            state.history.append(Message(role=Role.SYSTEM, content=state.session.system_prompt))
        state.session.turns.clear()
        print(state.renderer.success("Session reset"))

    elif command == "/save":
        if arg:
            state.session.name = arg
        path = state.session.save(state.sessions_dir)
        print(state.renderer.success(f"Session saved to {path}"))

    elif command == "/load":
        if not arg:
            print(state.renderer.error("Usage: /load <name>"))
            return True
        try:
            loaded = PlaygroundSession.load(arg, state.sessions_dir)
        except FileNotFoundError as exc:
            print(state.renderer.error(str(exc)))
            return True
        state.session = loaded
        state.history = []
        if loaded.system_prompt:
            state.history.append(Message(role=Role.SYSTEM, content=loaded.system_prompt))
        for turn in loaded.turns:
            state.history.append(Message(role=Role.USER, content=turn.user_input))
            state.history.append(Message(role=Role.ASSISTANT, content=turn.assistant_output))
        print(state.renderer.success(f"Loaded '{arg}' with {len(loaded.turns)} turns"))

    elif command == "/sessions":
        dir_ = state.sessions_dir
        if not dir_.exists() or not list(dir_.glob("*.json")):
            print(state.renderer.info("No saved sessions found"))
            return True
        for p in sorted(dir_.glob("*.json")):
            try:
                data = PlaygroundSession.model_validate_json(p.read_text())
                n = len(data.turns)
                cost = state.renderer.format_usd(data.total_cost_usd())
                label = data.name or data.session_id[:8]
                print(
                    f"  {label:<20}  {n} turn{'s' if n != 1 else ''}  {cost}  "
                    f"{data.created_at.strftime('%Y-%m-%d')}"
                )
            except Exception:
                print(f"  {p.stem} (unreadable)")

    elif command == "/export":
        if not state.session.turns:
            print(state.renderer.error("No turns to export"))
            return True
        try:
            case = state.session.to_eval_case()
        except ValueError as exc:
            print(state.renderer.error(str(exc)))
            return True
        json_str = case.model_dump_json(indent=2)
        if arg:
            out_path = Path(arg)
            out_path.write_text(json_str)
            print(state.renderer.success(f"EvalCase exported to {out_path}"))
        else:
            print(json_str)

    elif command == "/version":
        sub_parts = arg.split(None, 1)
        sub_cmd = sub_parts[0].lower() if sub_parts else ""
        sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

        if sub_cmd == "save":
            if not sub_arg:
                print(state.renderer.error("Usage: /version save <name>"))
                return True
            try:
                state.version_manager.register(sub_arg, state.session.system_prompt)
                print(state.renderer.success(f"System prompt saved as version '{sub_arg}'"))
            except ValueError as exc:
                print(state.renderer.error(str(exc)))

        elif sub_cmd == "diff":
            versions = sub_arg.split(None, 1)
            if len(versions) < 2:
                print(state.renderer.error("Usage: /version diff <v1> <v2>"))
                return True
            v1, v2 = versions[0], versions[1]
            try:
                diff = state.version_manager.diff(v1, v2)
            except ValueError as exc:
                print(state.renderer.error(str(exc)))
                return True
            for removed in diff.removed_lines:
                print(f"- {removed}")
            for added in diff.added_lines:
                print(f"+ {added}")
            print(state.renderer.info(f"Similarity: {diff.similarity_ratio:.1%}"))

        else:
            print(state.renderer.error("Usage: /version save <name> | /version diff <v1> <v2>"))

    else:
        print(state.renderer.error(f"Unknown command: {command}  —  type /help for commands"))

    return True


async def run_compare(
    user_message: str,
    models: list[str],
    system_prompt: str,
    config: GrampusConfig,
    *,
    _messages: list[Message] | None = None,
) -> list[CompareResult]:
    """Run *user_message* against each model concurrently and collect results.

    Args:
        user_message: The prompt text to send.
        models: One or more model names to query.
        system_prompt: System prompt prepended if no system message in _messages.
        config: GrampusConfig used to build each ModelClient.
        _messages: Optional pre-built message list (skips building from scratch).

    Returns:
        One CompareResult per model, preserving order.

    Raises:
        ValueError: If models list is empty.
    """
    if not models:
        raise ValueError("at least one model name is required")

    if _messages is not None:
        messages = list(_messages)
    else:
        messages = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=user_message))

    async def _call_model(model: str) -> CompareResult:
        client = make_client(model, config)
        start = time.monotonic()
        try:
            response = await client.complete(messages=messages, model=model)
            elapsed = time.monotonic() - start
            return CompareResult(
                model=model,
                output=response.content or "",
                token_usage=response.token_usage,
                duration_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return CompareResult(
                model=model,
                output="",
                token_usage=None,
                duration_seconds=elapsed,
                error=str(exc),
            )

    results = await asyncio.gather(*(_call_model(m) for m in models))
    return list(results)


async def run_repl(
    config: GrampusConfig,
    model: str,
    system_prompt: str,
    *,
    sessions_dir: Path | None = None,
    _client: ModelClient | None = None,
) -> None:
    """Start the interactive playground REPL.

    Args:
        config: Application configuration.
        model: Starting model name.
        system_prompt: Optional system prompt text.
        sessions_dir: Override save/load directory (used in tests).
        _client: Inject a pre-built client (used in tests).
    """
    dir_ = sessions_dir or _SESSIONS_DIR
    client = _client if _client is not None else make_client(model, config)
    renderer = Renderer()
    version_manager = PromptVersionManager(agent_id="playground")

    session = PlaygroundSession(model=model, system_prompt=system_prompt)
    history: list[Message] = []
    if system_prompt:
        history.append(Message(role=Role.SYSTEM, content=system_prompt))

    state = _ReplState(
        session=session,
        active_model=model,
        client=client,
        renderer=renderer,
        history=history,
        config=config,
        version_manager=version_manager,
        sessions_dir=dir_,
    )

    print(renderer.separator(f"Nexus Playground — {model}"))
    print(renderer.info("Type /help for commands, /exit to quit"))

    while True:
        try:
            prompt = _prompt_str(state.active_model)
            line = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            should_continue = await _handle_command(line, state)
            if not should_continue:
                break
        else:
            await _send_message(line, state)
