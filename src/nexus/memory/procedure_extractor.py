"""Procedure extractor: derive reusable procedures from completed tool-call sequences."""

from __future__ import annotations

import json
import uuid
from typing import Any

from nexus.core.logging import get_logger
from nexus.core.types import Message, Role, ToolCall
from nexus.memory.types import Procedure, ProcedureStep

_log = get_logger(__name__)

_MIN_STEPS = 2

_EXTRACTION_PROMPT = """\
A task was completed using the following tool call sequence:

Task description: {task_description}

Tool calls (in order):
{tool_calls}

Generalize this into a reusable procedure template.
Return ONLY a valid JSON object (no markdown, no extra text) with exactly these fields:
  name        — short snake_case identifier
  description — one sentence describing what this procedure does
  steps       — array of step objects, each with:
                  action (string), tool_name (string or null),
                  parameters_template (object), expected_outcome (string or null)
  trigger_conditions — array of strings describing when to use this procedure

Example response format:
{{
  "name": "example_proc",
  "description": "Does X then Y.",
  "steps": [{{"action": "do X", "tool_name": "tool_x", "parameters_template": {{}}, "expected_outcome": "X result"}}],
  "trigger_conditions": ["when user asks to do X"]
}}"""


def _format_tool_calls(tool_calls: list[ToolCall]) -> str:
    lines: list[str] = []
    for i, tc in enumerate(tool_calls, start=1):
        lines.append(f"{i}. {tc.name}({json.dumps(tc.arguments)})")
    return "\n".join(lines)


class ProcedureExtractor:
    """Extract reusable procedure templates from completed tool-call sequences via LLM.

    Args:
        model_client: LLM client used for extraction.
        procedural_memory: Where extracted procedures are stored.
        agent_id: Scopes stored procedures to this agent.
    """

    def __init__(
        self,
        model_client: Any,
        procedural_memory: Any,
        *,
        agent_id: str,
    ) -> None:
        self._client = model_client
        self._memory = procedural_memory
        self._agent_id = agent_id

    async def extract(
        self,
        tool_calls: list[ToolCall],
        task_description: str,
    ) -> Procedure | None:
        """Attempt to extract a Procedure from *tool_calls*.

        Returns None (without calling the LLM) if fewer than 2 tool calls are
        provided. Returns None if the LLM response is unparseable or yields an
        empty steps list. On success, stores the procedure and returns it.
        """
        if len(tool_calls) < _MIN_STEPS:
            _log.debug("procedure_extraction_skipped_too_few_steps", count=len(tool_calls))
            return None

        raw = await self._call_llm(tool_calls, task_description)
        procedure = _parse_procedure(raw, agent_id=self._agent_id)
        if procedure is None:
            return None

        await self._memory.store(procedure)
        _log.debug(
            "procedure_extracted",
            procedure_id=procedure.id,
            name=procedure.name,
            steps=len(procedure.steps),
            agent=self._agent_id,
        )
        return procedure

    async def _call_llm(self, tool_calls: list[ToolCall], task_description: str) -> str:
        prompt = _EXTRACTION_PROMPT.format(
            task_description=task_description,
            tool_calls=_format_tool_calls(tool_calls),
        )
        messages = [Message(role=Role.USER, content=prompt)]
        try:
            response = await self._client.complete(
                messages=messages,
                model="claude-haiku-4-5-20251001",
                temperature=0.0,
            )
            return (response.content or "").strip()
        except Exception as exc:
            _log.warning("procedure_extraction_llm_failed", error=str(exc))
            return ""


def _parse_procedure(raw: str, *, agent_id: str) -> Procedure | None:
    """Parse LLM output into a Procedure. Returns None on any failure."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        _log.warning("procedure_extraction_invalid_json", raw_len=len(raw))
        return None

    if not isinstance(data, dict):
        _log.warning("procedure_extraction_not_a_dict")
        return None

    raw_steps: list[Any] = data.get("steps") or []
    if not raw_steps:
        _log.debug("procedure_extraction_no_steps")
        return None

    steps = [
        ProcedureStep(
            action=s.get("action", ""),
            tool_name=s.get("tool_name"),
            parameters_template=s.get("parameters_template") or {},
            expected_outcome=s.get("expected_outcome"),
        )
        for s in raw_steps
        if isinstance(s, dict)
    ]

    if not steps:
        return None

    return Procedure(
        id=str(uuid.uuid4()),
        name=data.get("name", "unnamed_procedure"),
        description=data.get("description", ""),
        steps=steps,
        trigger_conditions=list(data.get("trigger_conditions") or []),
        agent_id=agent_id,
    )
