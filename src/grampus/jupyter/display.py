"""Rich HTML rendering helpers for Jupyter notebook display."""

from __future__ import annotations

import html as _html_module
from typing import TYPE_CHECKING, Any

from grampus.core.types import ExecutionResult, Message

if TYPE_CHECKING:
    pass


def _ipython_available() -> bool:
    try:
        import IPython.display  # noqa: F401

        return True
    except ImportError:
        return False


def _html_or_str(html_str: str) -> Any:
    """Return an IPython HTML display object, or the raw string when IPython is absent."""
    if _ipython_available():
        from IPython.display import HTML

        return HTML(html_str)
    return html_str


def render_result(result: ExecutionResult, *, agent_name: str = "Agent") -> Any:
    """Return a styled HTML card summarising an ExecutionResult.

    Returns an ``IPython.display.HTML`` object when IPython is available,
    or a plain HTML string otherwise (so tests can run without IPython).
    """
    output = _html_module.escape(result.output or "")
    total_tokens = result.token_usage.total_tokens
    cost_usd = result.token_usage.cost_usd
    duration = result.duration_seconds
    tool_calls_made = result.tool_calls_made
    steps_taken = result.steps_taken

    html_str = (
        '<div style="font-family: monospace; border: 1px solid #e0e0e0; border-radius: 6px; '
        'padding: 16px; margin: 8px 0; background: #fafafa;">'
        f'<div style="font-weight: bold; color: #1a1a1a; margin-bottom: 8px;">'
        f"🤖 {_html_module.escape(agent_name)}</div>"
        f'<div style="white-space: pre-wrap; color: #333;">{output}</div>'
        '<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 12px 0;">'
        '<div style="font-size: 0.85em; color: #666;">'
        f"⏱ {duration:.1f}s &nbsp;|&nbsp;"
        f"🔢 {total_tokens:,} tokens &nbsp;|&nbsp;"
        f"💰 ${cost_usd:.4f} &nbsp;|&nbsp;"
        f"🔧 {tool_calls_made} tool calls &nbsp;|&nbsp;"
        f"📋 {steps_taken} steps"
        "</div></div>"
    )
    return _html_or_str(html_str)


def render_stream_token(delta: str) -> None:
    """Display a streaming token inline in the notebook cell output."""
    if _ipython_available():
        from IPython.display import HTML, display

        display(HTML(delta.replace("\n", "<br>")), clear=False)
    else:
        print(delta, end="", flush=True)


def render_tool_call(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Return a compact yellow-tinted HTML badge describing a tool call."""
    args_preview = _html_module.escape(str(arguments)[:80])
    name_escaped = _html_module.escape(tool_name)
    html_str = (
        '<div style="font-family: monospace; font-size: 0.85em; background: #fff8e1; '
        'border-left: 3px solid #f9a825; padding: 6px 10px; margin: 4px 0; color: #555;">'
        f"⚡ {name_escaped}({args_preview})"
        "</div>"
    )
    return _html_or_str(html_str)


def render_tool_result(tool_name: str, output: Any) -> Any:
    """Return a green-tinted HTML badge describing a tool result."""
    result_preview = _html_module.escape(str(output)[:80])
    name_escaped = _html_module.escape(tool_name)
    html_str = (
        '<div style="font-family: monospace; font-size: 0.85em; background: #e8f5e9; '
        'border-left: 3px solid #43a047; padding: 6px 10px; margin: 4px 0; color: #555;">'
        f"✓ {name_escaped} → {result_preview}"
        "</div>"
    )
    return _html_or_str(html_str)


def render_messages(messages: list[Message]) -> Any:
    """Render a conversation history as a chat-style HTML block."""
    if not messages:
        return _html_or_str('<div style="color: #999; font-style: italic;">No messages</div>')

    _role_styles: dict[str, str] = {
        "system": "background: #f5f5f5; color: #666; font-style: italic;",
        "user": "background: #e3f2fd; color: #1565c0;",
        "assistant": "background: #e8f5e9; color: #2e7d32;",
        "tool": "background: #fff8e1; color: #555; font-family: monospace;",
    }

    parts: list[str] = []
    for msg in messages:
        role = str(msg.role)
        style = _role_styles.get(role, "background: #fafafa; color: #333;")
        content = _html_module.escape(msg.content or "")
        parts.append(
            f'<div style="{style} padding: 8px 12px; margin: 4px 0; border-radius: 4px;">'
            f"<strong>{_html_module.escape(role)}</strong>: {content}"
            "</div>"
        )

    return _html_or_str("<div>" + "".join(parts) + "</div>")
