"""Tests for grampus.memory.consolidation — ConsolidationPipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from grampus.core.models.base import ModelResponse
from grampus.core.types import TokenUsage
from grampus.memory.consolidation import ConsolidationPipeline, ConsolidationResult
from grampus.memory.types import EpisodicRecord, SemanticFact


def make_episode(
    episode_id: str = "ep-1",
    content: str = "The user likes Python.",
    consolidated: bool = False,
) -> EpisodicRecord:
    return EpisodicRecord(
        id=episode_id,
        agent_id="agent-1",
        session_id="s1",
        content=content,
        trust_score=0.9,
        importance_score=0.5,
        metadata={"consolidated": True} if consolidated else {},
    )


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


SINGLE_FACT_JSON = json.dumps(
    [{"subject": "user", "predicate": "likes", "object_value": "Python", "confidence": 0.9}]
)
EMPTY_JSON = "[]"
INVALID_JSON = "not valid json"


@pytest.fixture()
def mock_episodic() -> AsyncMock:
    mem = AsyncMock()
    mem.list_all = AsyncMock(return_value=[])
    mem.update_metadata = AsyncMock(return_value=None)
    return mem


@pytest.fixture()
def mock_semantic() -> AsyncMock:
    sem = AsyncMock()
    sem.store = AsyncMock(side_effect=lambda f: f)
    return sem


@pytest.fixture()
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=make_model_response(SINGLE_FACT_JSON))
    return client


@pytest.fixture()
def pipeline(
    mock_episodic: AsyncMock, mock_semantic: AsyncMock, mock_client: AsyncMock
) -> ConsolidationPipeline:
    return ConsolidationPipeline(
        episodic_memory=mock_episodic,
        semantic_memory=mock_semantic,
        model_client=mock_client,
        agent_id="agent-1",
    )


class TestConsolidationPipelineResult:
    async def test_empty_memory_returns_zero_result(
        self, pipeline: ConsolidationPipeline, mock_episodic: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = []
        result = await pipeline.run()
        assert isinstance(result, ConsolidationResult)
        assert result.episodes_processed == 0
        assert result.facts_extracted == 0

    async def test_result_is_consolidation_result_type(
        self, pipeline: ConsolidationPipeline
    ) -> None:
        result = await pipeline.run()
        assert isinstance(result, ConsolidationResult)

    async def test_episodes_processed_counts_unconsolidated(
        self, pipeline: ConsolidationPipeline, mock_episodic: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = [
            make_episode("ep-1"),
            make_episode("ep-2"),
        ]
        result = await pipeline.run()
        assert result.episodes_processed == 2

    async def test_facts_extracted_counts_parsed_facts(
        self, pipeline: ConsolidationPipeline, mock_episodic: AsyncMock, mock_client: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = [make_episode("ep-1")]
        two_facts = json.dumps(
            [
                {
                    "subject": "user",
                    "predicate": "likes",
                    "object_value": "Python",
                    "confidence": 0.9,
                },
                {"subject": "user", "predicate": "uses", "object_value": "vim", "confidence": 0.7},
            ]
        )
        mock_client.complete.return_value = make_model_response(two_facts)
        result = await pipeline.run()
        assert result.facts_extracted == 2


class TestConsolidationPipelineExtraction:
    async def test_calls_model_client_for_unconsolidated_episodes(
        self, pipeline: ConsolidationPipeline, mock_episodic: AsyncMock, mock_client: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = [make_episode("ep-1")]
        await pipeline.run()
        mock_client.complete.assert_called_once()

    async def test_stores_extracted_facts_in_semantic_memory(
        self,
        pipeline: ConsolidationPipeline,
        mock_episodic: AsyncMock,
        mock_semantic: AsyncMock,
    ) -> None:
        mock_episodic.list_all.return_value = [make_episode("ep-1")]
        await pipeline.run()
        mock_semantic.store.assert_called_once()
        stored: SemanticFact = mock_semantic.store.call_args[0][0]
        assert stored.subject == "user"
        assert stored.predicate == "likes"
        assert stored.object_value == "Python"

    async def test_episode_id_added_to_source_ids(
        self,
        pipeline: ConsolidationPipeline,
        mock_episodic: AsyncMock,
        mock_semantic: AsyncMock,
    ) -> None:
        mock_episodic.list_all.return_value = [make_episode("ep-42")]
        await pipeline.run()
        stored: SemanticFact = mock_semantic.store.call_args[0][0]
        assert "ep-42" in stored.source_episode_ids

    async def test_marks_processed_episodes_consolidated(
        self, pipeline: ConsolidationPipeline, mock_episodic: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = [make_episode("ep-1")]
        await pipeline.run()
        mock_episodic.update_metadata.assert_called_once_with("ep-1", {"consolidated": True})

    async def test_skips_already_consolidated_episodes(
        self, pipeline: ConsolidationPipeline, mock_episodic: AsyncMock, mock_client: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = [
            make_episode("ep-old", consolidated=True),
        ]
        result = await pipeline.run()
        mock_client.complete.assert_not_called()
        assert result.episodes_processed == 0

    async def test_batch_size_limits_processing(
        self, mock_episodic: AsyncMock, mock_semantic: AsyncMock, mock_client: AsyncMock
    ) -> None:
        pipeline = ConsolidationPipeline(
            episodic_memory=mock_episodic,
            semantic_memory=mock_semantic,
            model_client=mock_client,
            agent_id="agent-1",
            batch_size=2,
        )
        mock_episodic.list_all.return_value = [make_episode(f"ep-{i}") for i in range(5)]
        result = await pipeline.run()
        assert result.episodes_processed == 2

    async def test_invalid_json_from_model_is_handled_gracefully(
        self, pipeline: ConsolidationPipeline, mock_episodic: AsyncMock, mock_client: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = [make_episode("ep-1")]
        mock_client.complete.return_value = make_model_response(INVALID_JSON)
        result = await pipeline.run()
        assert result.facts_extracted == 0
        assert result.episodes_processed == 1

    async def test_empty_json_array_is_valid(
        self,
        pipeline: ConsolidationPipeline,
        mock_episodic: AsyncMock,
        mock_client: AsyncMock,
        mock_semantic: AsyncMock,
    ) -> None:
        mock_episodic.list_all.return_value = [make_episode("ep-1")]
        mock_client.complete.return_value = make_model_response(EMPTY_JSON)
        result = await pipeline.run()
        assert result.facts_extracted == 0
        mock_semantic.store.assert_not_called()

    async def test_no_model_call_when_no_episodes(
        self, pipeline: ConsolidationPipeline, mock_episodic: AsyncMock, mock_client: AsyncMock
    ) -> None:
        mock_episodic.list_all.return_value = []
        await pipeline.run()
        mock_client.complete.assert_not_called()
