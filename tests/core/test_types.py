"""Tests for nexus.core.types — Pydantic v2 data models."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from nexus.core.types import (
    AgentDefinition,
    AgentState,
    AgentStatus,
    ExecutionResult,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Role and AgentStatus enums
# ---------------------------------------------------------------------------


class TestRoleEnum:
    def test_role_has_expected_values(self) -> None:
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_role_from_string(self) -> None:
        assert Role("user") == Role.USER


class TestAgentStatusEnum:
    def test_status_has_expected_values(self) -> None:
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.WAITING_FOR_HUMAN.value == "waiting_for_human"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_create_basic(self) -> None:
        tc = ToolCall(id="call-1", name="search", arguments={"query": "hello"})
        assert tc.id == "call-1"
        assert tc.name == "search"
        assert tc.arguments == {"query": "hello"}

    def test_json_round_trip(self) -> None:
        tc = ToolCall(id="c1", name="calc", arguments={"x": 1})
        restored = ToolCall.model_validate_json(tc.model_dump_json())
        assert restored == tc

    def test_arguments_default_empty(self) -> None:
        tc = ToolCall(id="c2", name="ping", arguments={})
        assert tc.arguments == {}

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall.model_validate({"id": "c3"})  # missing name


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_create_success(self) -> None:
        tr = ToolResult(tool_call_id="call-1", output="result", duration_ms=50)
        assert tr.tool_call_id == "call-1"
        assert tr.output == "result"
        assert tr.error is None

    def test_create_with_error(self) -> None:
        tr = ToolResult(tool_call_id="c1", output=None, error="timeout", duration_ms=100)
        assert tr.error == "timeout"
        assert tr.output is None

    def test_json_round_trip(self) -> None:
        tr = ToolResult(tool_call_id="x", output="ok", duration_ms=10)
        assert ToolResult.model_validate_json(tr.model_dump_json()) == tr


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_simple_user_message(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_results == []

    def test_timestamp_auto_set(self) -> None:
        msg = Message(role=Role.USER, content="hi")
        assert isinstance(msg.timestamp, datetime)

    def test_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments={"q": "test"})
        msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_json_round_trip(self) -> None:
        msg = Message(role=Role.USER, content="round trip")
        restored = Message.model_validate_json(msg.model_dump_json())
        assert restored.role == msg.role
        assert restored.content == msg.content


# ---------------------------------------------------------------------------
# ToolParameter and ToolDefinition
# ---------------------------------------------------------------------------


class TestToolParameter:
    def test_required_param(self) -> None:
        p = ToolParameter(name="query", type="string", description="search query", required=True)
        assert p.required is True
        assert p.default is None
        assert p.enum is None

    def test_optional_param_with_default(self) -> None:
        p = ToolParameter(
            name="limit", type="integer", description="max results", required=False, default=10
        )
        assert p.default == 10

    def test_enum_param(self) -> None:
        p = ToolParameter(
            name="fmt", type="string", description="format", required=True, enum=["json", "text"]
        )
        assert p.enum == ["json", "text"]


class TestToolDefinition:
    def _make_tool(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web",
            parameters=[
                ToolParameter(name="query", type="string", description="query", required=True),
                ToolParameter(
                    name="limit", type="integer", description="max", required=False, default=5
                ),
            ],
        )

    def test_basic_fields(self) -> None:
        t = self._make_tool()
        assert t.name == "web_search"
        assert len(t.parameters) == 2

    def test_to_function_schema_structure(self) -> None:
        schema = self._make_tool().to_function_schema()
        assert schema["name"] == "web_search"
        assert "description" in schema
        assert "parameters" in schema
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

    def test_to_function_schema_required_list(self) -> None:
        schema = self._make_tool().to_function_schema()
        required = schema["parameters"]["required"]
        assert "query" in required
        assert "limit" not in required

    def test_to_function_schema_properties(self) -> None:
        schema = self._make_tool().to_function_schema()
        props = schema["parameters"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"
        assert "limit" in props

    def test_to_function_schema_is_valid_json(self) -> None:
        schema = self._make_tool().to_function_schema()
        json_str = json.dumps(schema)
        assert json.loads(json_str) == schema

    def test_enum_values_in_schema(self) -> None:
        t = ToolDefinition(
            name="fmt_tool",
            description="formatter",
            parameters=[
                ToolParameter(
                    name="fmt",
                    type="string",
                    description="output format",
                    required=True,
                    enum=["json", "csv"],
                )
            ],
        )
        schema = t.to_function_schema()
        assert schema["parameters"]["properties"]["fmt"]["enum"] == ["json", "csv"]

    def test_no_parameters(self) -> None:
        t = ToolDefinition(name="ping", description="ping", parameters=[])
        schema = t.to_function_schema()
        assert schema["parameters"]["required"] == []
        assert schema["parameters"]["properties"] == {}

    def test_json_round_trip(self) -> None:
        t = self._make_tool()
        restored = ToolDefinition.model_validate_json(t.model_dump_json())
        assert restored.name == t.name


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_basic(self) -> None:
        tu = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            model="claude-3-haiku",
        )
        assert tu.total_tokens == 150

    def test_json_round_trip(self) -> None:
        tu = TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0, model="gpt-4o"
        )
        assert TokenUsage.model_validate_json(tu.model_dump_json()) == tu


# ---------------------------------------------------------------------------
# AgentDefinition
# ---------------------------------------------------------------------------


class TestAgentDefinition:
    def test_minimal_agent(self) -> None:
        a = AgentDefinition(name="my-agent", model="claude-3-haiku")
        assert a.name == "my-agent"
        assert a.model == "claude-3-haiku"
        assert a.tools == []
        assert a.memory_enabled is True
        assert a.max_iterations > 0

    def test_full_agent(self) -> None:
        a = AgentDefinition(
            name="full",
            model="gpt-4o",
            system_prompt="You are helpful.",
            tools=["search", "calc"],
            max_iterations=20,
            temperature=0.7,
            memory_enabled=False,
            cost_budget_usd=5.0,
        )
        assert a.system_prompt == "You are helpful."
        assert a.cost_budget_usd == 5.0
        assert a.temperature == 0.7

    def test_temperature_validation(self) -> None:
        with pytest.raises(ValidationError):
            AgentDefinition(name="a", model="m", temperature=2.5)

    def test_json_round_trip(self) -> None:
        a = AgentDefinition(name="test", model="m", temperature=0.5)
        assert AgentDefinition.model_validate_json(a.model_dump_json()) == a


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_create(self) -> None:
        state = AgentState(
            agent_id="agent-1",
            session_id="session-1",
        )
        assert state.agent_id == "agent-1"
        assert state.status == AgentStatus.IDLE
        assert state.messages == []
        assert state.current_step == 0

    def test_json_round_trip(self) -> None:
        state = AgentState(agent_id="a", session_id="s")
        restored = AgentState.model_validate_json(state.model_dump_json())
        assert restored.agent_id == state.agent_id
        assert restored.status == state.status


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_create(self) -> None:
        tu = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0, model="m")
        result = ExecutionResult(
            output="done",
            messages=[],
            tool_calls_made=2,
            token_usage=tu,
            duration_seconds=1.5,
            steps_taken=3,
            status=AgentStatus.COMPLETED,
        )
        assert result.output == "done"
        assert result.steps_taken == 3
        assert result.status == AgentStatus.COMPLETED

    def test_json_round_trip(self) -> None:
        tu = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="m")
        r = ExecutionResult(
            output="x",
            messages=[],
            tool_calls_made=0,
            token_usage=tu,
            duration_seconds=0.1,
            steps_taken=1,
            status=AgentStatus.COMPLETED,
        )
        assert ExecutionResult.model_validate_json(r.model_dump_json()).output == "x"
