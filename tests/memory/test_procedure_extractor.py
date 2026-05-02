"""Tests for nexus.memory.procedure_extractor — ProcedureExtractor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from nexus.core.models.base import ModelResponse
from nexus.core.types import TokenUsage, ToolCall
from nexus.memory.procedure_extractor import ProcedureExtractor
from nexus.memory.types import Procedure


def make_tool_call(
    call_id: str = "tc-1",
    name: str = "web_search",
    arguments: dict | None = None,
) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments or {"query": "python tips"})


def make_model_response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0, model="test"
        ),
        model="test",
        stop_reason="end_turn",
    )


VALID_PROCEDURE_JSON = json.dumps(
    {
        "name": "web_research",
        "description": "Search the web and summarize findings",
        "steps": [
            {
                "action": "search for information",
                "tool_name": "web_search",
                "parameters_template": {"query": "{topic}"},
                "expected_outcome": "list of relevant results",
            },
            {
                "action": "summarize results",
                "tool_name": "summarizer",
                "parameters_template": {"text": "{results}"},
                "expected_outcome": "concise summary",
            },
        ],
        "trigger_conditions": ["research task", "find information about"],
    }
)

INVALID_JSON = "not valid json"
EMPTY_OBJECT_JSON = "{}"
NON_DICT_JSON = "[]"


@pytest.fixture()
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=make_model_response(VALID_PROCEDURE_JSON))
    return client


@pytest.fixture()
def mock_procedural() -> AsyncMock:
    mem = AsyncMock()
    mem.store = AsyncMock(side_effect=lambda p: p)
    return mem


@pytest.fixture()
def extractor(mock_client: AsyncMock, mock_procedural: AsyncMock) -> ProcedureExtractor:
    return ProcedureExtractor(
        model_client=mock_client,
        procedural_memory=mock_procedural,
        agent_id="agent-1",
    )


class TestProcedureExtractorMinSteps:
    async def test_returns_none_for_empty_tool_calls(
        self, extractor: ProcedureExtractor, mock_client: AsyncMock
    ) -> None:
        result = await extractor.extract([], "do something")
        assert result is None
        mock_client.complete.assert_not_called()

    async def test_returns_none_for_single_tool_call(
        self, extractor: ProcedureExtractor, mock_client: AsyncMock
    ) -> None:
        result = await extractor.extract([make_tool_call()], "do something")
        assert result is None
        mock_client.complete.assert_not_called()

    async def test_proceeds_for_two_tool_calls(
        self, extractor: ProcedureExtractor, mock_client: AsyncMock
    ) -> None:
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2", name="summarizer")]
        result = await extractor.extract(calls, "research task")
        mock_client.complete.assert_called_once()
        assert result is not None


class TestProcedureExtractorParsing:
    async def test_returns_none_for_invalid_json(
        self, extractor: ProcedureExtractor, mock_client: AsyncMock
    ) -> None:
        mock_client.complete.return_value = make_model_response(INVALID_JSON)
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert result is None

    async def test_returns_none_for_non_dict_json(
        self, extractor: ProcedureExtractor, mock_client: AsyncMock
    ) -> None:
        mock_client.complete.return_value = make_model_response(NON_DICT_JSON)
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert result is None

    async def test_returns_procedure_for_valid_json(
        self, extractor: ProcedureExtractor, mock_client: AsyncMock
    ) -> None:
        mock_client.complete.return_value = make_model_response(VALID_PROCEDURE_JSON)
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert isinstance(result, Procedure)

    async def test_extracted_procedure_has_correct_name(
        self, extractor: ProcedureExtractor
    ) -> None:
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert result is not None
        assert result.name == "web_research"

    async def test_extracted_procedure_has_correct_description(
        self, extractor: ProcedureExtractor
    ) -> None:
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert result is not None
        assert "search" in result.description.lower()

    async def test_extracted_procedure_has_steps(self, extractor: ProcedureExtractor) -> None:
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert result is not None
        assert len(result.steps) >= 1

    async def test_extracted_procedure_has_trigger_conditions(
        self, extractor: ProcedureExtractor
    ) -> None:
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert result is not None
        assert isinstance(result.trigger_conditions, list)


class TestProcedureExtractorStorage:
    async def test_stores_extracted_procedure_in_memory(
        self,
        extractor: ProcedureExtractor,
        mock_procedural: AsyncMock,
    ) -> None:
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        await extractor.extract(calls, "research task")
        mock_procedural.store.assert_called_once()

    async def test_does_not_store_when_result_is_none(
        self,
        extractor: ProcedureExtractor,
        mock_client: AsyncMock,
        mock_procedural: AsyncMock,
    ) -> None:
        mock_client.complete.return_value = make_model_response(INVALID_JSON)
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        await extractor.extract(calls, "research task")
        mock_procedural.store.assert_not_called()

    async def test_stored_procedure_has_agent_id(
        self,
        extractor: ProcedureExtractor,
        mock_procedural: AsyncMock,
    ) -> None:
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        await extractor.extract(calls, "research task")
        stored: Procedure = mock_procedural.store.call_args[0][0]
        assert stored.agent_id == "agent-1"

    async def test_stored_procedure_has_unique_id(
        self,
        extractor: ProcedureExtractor,
        mock_procedural: AsyncMock,
    ) -> None:
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert result is not None
        assert result.id != ""

    async def test_returns_none_when_steps_list_is_empty_in_json(
        self, extractor: ProcedureExtractor, mock_client: AsyncMock
    ) -> None:
        no_steps = json.dumps(
            {
                "name": "empty_proc",
                "description": "no steps",
                "steps": [],
                "trigger_conditions": [],
            }
        )
        mock_client.complete.return_value = make_model_response(no_steps)
        calls = [make_tool_call("tc-1"), make_tool_call("tc-2")]
        result = await extractor.extract(calls, "research task")
        assert result is None
