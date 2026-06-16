"""Tests for grampus.memory.procedure_matcher — ProcedureMatcher."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from grampus.memory.procedure_matcher import ProcedureMatcher
from grampus.memory.types import Procedure, ProcedureStep


def make_step(action: str = "do something") -> ProcedureStep:
    return ProcedureStep(action=action, tool_name="tool_x")


def make_procedure(
    procedure_id: str | None = None,
    name: str = "research_topic",
    description: str = "Research a topic",
    embedding: list[float] | None = None,
    agent_id: str = "agent-1",
) -> Procedure:
    return Procedure(
        id=procedure_id or str(uuid.uuid4()),
        name=name,
        description=description,
        steps=[make_step(), make_step("step 2")],
        agent_id=agent_id,
        embedding=embedding,
    )


@pytest.fixture()
def mock_procedural() -> AsyncMock:
    mem = AsyncMock()
    mem.list_all = AsyncMock(return_value=[])
    return mem


@pytest.fixture()
def mock_embeddings() -> AsyncMock:
    svc = AsyncMock()
    svc.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
    return svc


@pytest.fixture()
def matcher(mock_procedural: AsyncMock, mock_embeddings: AsyncMock) -> ProcedureMatcher:
    return ProcedureMatcher(
        procedural_memory=mock_procedural,
        embedding_service=mock_embeddings,
    )


class TestProcedureMatcherEmpty:
    async def test_returns_empty_list_for_no_procedures(
        self, matcher: ProcedureMatcher, mock_procedural: AsyncMock
    ) -> None:
        mock_procedural.list_all.return_value = []
        results = await matcher.find_matches("research task")
        assert results == []

    async def test_returns_empty_list_when_all_procedures_lack_embedding(
        self, matcher: ProcedureMatcher, mock_procedural: AsyncMock
    ) -> None:
        mock_procedural.list_all.return_value = [make_procedure(embedding=None)]
        results = await matcher.find_matches("research task")
        assert results == []


class TestProcedureMatcherResults:
    async def test_returns_list_of_tuples(
        self,
        matcher: ProcedureMatcher,
        mock_procedural: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mock_procedural.list_all.return_value = [make_procedure(embedding=[1.0, 0.0, 0.0])]
        results = await matcher.find_matches("research task")
        assert len(results) == 1
        procedure, score = results[0]
        assert isinstance(procedure, Procedure)
        assert isinstance(score, float)

    async def test_score_is_between_0_and_1(
        self,
        matcher: ProcedureMatcher,
        mock_procedural: AsyncMock,
    ) -> None:
        mock_procedural.list_all.return_value = [make_procedure(embedding=[1.0, 0.0, 0.0])]
        results = await matcher.find_matches("research task")
        _, score = results[0]
        assert 0.0 <= score <= 1.0

    async def test_sorted_by_score_descending(
        self,
        mock_procedural: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]
        matcher = ProcedureMatcher(
            procedural_memory=mock_procedural, embedding_service=mock_embeddings
        )
        similar = make_procedure(procedure_id="sim", embedding=[1.0, 0.0])
        dissimilar = make_procedure(procedure_id="diff", embedding=[0.0, 1.0])
        mock_procedural.list_all.return_value = [dissimilar, similar]
        results = await matcher.find_matches("task")
        assert results[0][0].id == "sim"
        assert results[1][0].id == "diff"

    async def test_higher_similarity_yields_higher_score(
        self,
        mock_procedural: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]
        matcher = ProcedureMatcher(
            procedural_memory=mock_procedural, embedding_service=mock_embeddings
        )
        similar = make_procedure(procedure_id="sim", embedding=[1.0, 0.0])
        dissimilar = make_procedure(procedure_id="diff", embedding=[0.0, 1.0])
        mock_procedural.list_all.return_value = [similar, dissimilar]
        results = await matcher.find_matches("task")
        scores = {p.id: s for p, s in results}
        assert scores["sim"] > scores["diff"]

    async def test_top_k_limits_results(
        self,
        mock_procedural: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0, 0.0]
        matcher = ProcedureMatcher(
            procedural_memory=mock_procedural, embedding_service=mock_embeddings
        )
        procs = [make_procedure(embedding=[1.0, 0.0, 0.0]) for _ in range(10)]
        mock_procedural.list_all.return_value = procs
        results = await matcher.find_matches("task", top_k=3)
        assert len(results) <= 3

    async def test_skips_procedures_without_embedding(
        self,
        mock_procedural: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]
        matcher = ProcedureMatcher(
            procedural_memory=mock_procedural, embedding_service=mock_embeddings
        )
        with_emb = make_procedure(procedure_id="has-emb", embedding=[1.0, 0.0])
        without_emb = make_procedure(procedure_id="no-emb", embedding=None)
        mock_procedural.list_all.return_value = [with_emb, without_emb]
        results = await matcher.find_matches("task")
        ids = [p.id for p, _ in results]
        assert "has-emb" in ids
        assert "no-emb" not in ids

    async def test_embeds_query_once(
        self,
        mock_procedural: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        matcher = ProcedureMatcher(
            procedural_memory=mock_procedural, embedding_service=mock_embeddings
        )
        procs = [make_procedure(embedding=[1.0, 0.0]) for _ in range(3)]
        mock_procedural.list_all.return_value = procs
        await matcher.find_matches("my query")
        mock_embeddings.embed.assert_called_once_with("my query")

    async def test_identical_embeddings_score_is_one(
        self,
        mock_procedural: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mock_embeddings.embed.return_value = [1.0, 0.0]
        matcher = ProcedureMatcher(
            procedural_memory=mock_procedural, embedding_service=mock_embeddings
        )
        proc = make_procedure(embedding=[1.0, 0.0])
        mock_procedural.list_all.return_value = [proc]
        results = await matcher.find_matches("task")
        _, score = results[0]
        assert abs(score - 1.0) < 1e-9
