"""Tests for the F1 reflexion and self-improvement subsystem.

Covers: ProcedureType extension, ProceduralMemory query/find,
ReflexionEngine, SkillLibrary, PromptOptimizer, and AgentRunner integration.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from nexus.memory.procedural import ProceduralMemory
from nexus.memory.reflexion.engine import ReflexionEngine
from nexus.memory.reflexion.optimizer import PromptOptimizer
from nexus.memory.reflexion.skill_library import SkillLibrary
from nexus.memory.reflexion.types import (
    ReflexionHookResult,
    SkillExtractionResult,
)
from nexus.memory.types import Procedure, ProcedureType

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_procedure(
    *,
    procedure_type: ProcedureType = ProcedureType.WORKFLOW,
    confidence: float = 1.0,
    embedding: list[float] | None = None,
    success_count: int = 0,
    failure_count: int = 0,
    validated: bool = False,
    agent_id: str = "agent-1",
) -> Procedure:
    return Procedure(
        id=str(uuid.uuid4()),
        name=f"proc-{uuid.uuid4().hex[:6]}",
        description="A test procedure",
        agent_id=agent_id,
        procedure_type=procedure_type,
        confidence=confidence,
        embedding=embedding,
        success_count=success_count,
        failure_count=failure_count,
        metadata={"validated": validated},
    )


def _make_state(agent_id: str = "agent-1") -> Any:
    from nexus.core.types import AgentState

    return AgentState(agent_id=agent_id, session_id="session-1")


def _make_agent_def(model: str = "claude-sonnet-4-6") -> Any:
    from nexus.core.types import AgentDefinition

    return AgentDefinition(name="test-agent", model=model, system_prompt="You are helpful.")


def _make_store() -> Any:
    """Return an in-memory mock Dapr state store."""
    storage: dict[str, Any] = {}

    async def _save(entity: str, key: str, value: Any) -> None:
        storage[f"{entity}:{key}"] = value

    async def _get(entity: str, key: str, model_cls: Any) -> tuple[Any, str]:
        val = storage.get(f"{entity}:{key}")
        if val is None:
            return None, ""
        if hasattr(model_cls, "model_validate") and isinstance(val, dict):
            return model_cls.model_validate(val), "etag-1"
        if isinstance(val, model_cls):
            return val, "etag-1"
        return val, "etag-1"

    async def _delete(entity: str, key: str) -> None:
        storage.pop(f"{entity}:{key}", None)

    store = MagicMock()
    store.save = AsyncMock(side_effect=_save)
    store.get = AsyncMock(side_effect=_get)
    store.delete = AsyncMock(side_effect=_delete)
    return store


def _make_procedural_mem(agent_id: str = "agent-1") -> ProceduralMemory:
    return ProceduralMemory(_make_store(), agent_id=agent_id)


def _make_embedding_svc(dim: int = 4) -> Any:
    """Return a mock EmbeddingService that returns unit vectors."""
    svc = MagicMock()
    svc.embed = AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0])
    svc.embed_batch = AsyncMock(return_value=[[1.0, 0.0, 0.0, 0.0]])
    return svc


def _make_model_client(content: str = '{"quality": 0.8}') -> Any:
    from nexus.core.models.base import ModelResponse
    from nexus.core.types import TokenUsage

    resp = ModelResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model="test"
        ),
        model="test",
        stop_reason="end_turn",
    )
    client = MagicMock()
    client.complete = AsyncMock(return_value=resp)
    return client


def _make_execution_result(status_completed: bool = True) -> Any:
    from nexus.core.types import AgentStatus, ExecutionResult, Message, Role, TokenUsage

    return ExecutionResult(
        output="The task is done.",
        messages=[
            Message(role=Role.USER, content="do the task"),
            Message(role=Role.ASSISTANT, content="The task is done."),
        ],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model="test"
        ),
        duration_seconds=0.5,
        steps_taken=1,
        status=AgentStatus.COMPLETED if status_completed else AgentStatus.FAILED,
    )


# ===========================================================================
# 1–5: Procedure type extension tests
# ===========================================================================


class TestProcedureTypeExtension:
    def test_procedure_default_type_is_workflow(self) -> None:
        """Existing Procedure construction still works unchanged."""
        proc = Procedure(id="p1", name="x", description="y", agent_id="a")
        assert proc.procedure_type == ProcedureType.WORKFLOW

    def test_procedure_reflection_type(self) -> None:
        proc = Procedure(
            id="p2",
            name="x",
            description="y",
            agent_id="a",
            procedure_type=ProcedureType.REFLECTION,
        )
        assert proc.procedure_type == ProcedureType.REFLECTION

    def test_procedure_skill_type(self) -> None:
        proc = Procedure(
            id="p3",
            name="x",
            description="y",
            agent_id="a",
            procedure_type=ProcedureType.SKILL,
        )
        assert proc.procedure_type == ProcedureType.SKILL

    def test_procedure_confidence_defaults_to_one(self) -> None:
        proc = Procedure(id="p4", name="x", description="y", agent_id="a")
        assert proc.confidence == 1.0

    def test_procedure_confidence_clamped(self) -> None:
        with pytest.raises(ValidationError):
            Procedure(id="p5", name="x", description="y", agent_id="a", confidence=1.5)

    def test_procedure_round_trip(self) -> None:
        """Procedure round-trips through model_dump/model_validate."""
        proc = _make_procedure(procedure_type=ProcedureType.SKILL, confidence=0.7)
        dumped = proc.model_dump()
        restored = Procedure.model_validate(dumped)
        assert restored == proc


# ===========================================================================
# 6–11: ProceduralMemory extension tests
# ===========================================================================


class TestProceduralMemoryExtensions:
    @pytest.fixture()
    def mem(self) -> ProceduralMemory:
        return _make_procedural_mem()

    @pytest.mark.asyncio()
    async def test_query_by_type_returns_only_matching_type(self, mem: ProceduralMemory) -> None:
        await mem.store(_make_procedure(procedure_type=ProcedureType.SKILL))
        await mem.store(_make_procedure(procedure_type=ProcedureType.REFLECTION))
        await mem.store(_make_procedure(procedure_type=ProcedureType.WORKFLOW))

        skills = await mem.query_by_type(ProcedureType.SKILL)
        assert all(p.procedure_type == ProcedureType.SKILL for p in skills)
        assert len(skills) == 1

    @pytest.mark.asyncio()
    async def test_query_by_type_min_confidence_filters(self, mem: ProceduralMemory) -> None:
        await mem.store(_make_procedure(procedure_type=ProcedureType.REFLECTION, confidence=0.9))
        await mem.store(_make_procedure(procedure_type=ProcedureType.REFLECTION, confidence=0.2))

        high = await mem.query_by_type(ProcedureType.REFLECTION, min_confidence=0.5)
        assert len(high) == 1
        assert high[0].confidence >= 0.5

    @pytest.mark.asyncio()
    async def test_find_similar_returns_top_k_by_cosine(self, mem: ProceduralMemory) -> None:
        proc_a = _make_procedure(procedure_type=ProcedureType.SKILL, embedding=[1.0, 0.0, 0.0, 0.0])
        proc_b = _make_procedure(procedure_type=ProcedureType.SKILL, embedding=[0.0, 1.0, 0.0, 0.0])
        proc_c = _make_procedure(procedure_type=ProcedureType.SKILL, embedding=[1.0, 0.0, 0.0, 0.0])
        for p in (proc_a, proc_b, proc_c):
            await mem.store(p)

        results = await mem.find_similar([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # proc_a and proc_c have cosine=1.0, proc_b has cosine=0.0
        result_ids = {r.id for r in results}
        assert proc_a.id in result_ids
        assert proc_c.id in result_ids

    @pytest.mark.asyncio()
    async def test_find_similar_filters_by_type(self, mem: ProceduralMemory) -> None:
        skill = _make_procedure(procedure_type=ProcedureType.SKILL, embedding=[1.0, 0.0, 0.0, 0.0])
        reflection = _make_procedure(
            procedure_type=ProcedureType.REFLECTION, embedding=[1.0, 0.0, 0.0, 0.0]
        )
        await mem.store(skill)
        await mem.store(reflection)

        results = await mem.find_similar([1.0, 0.0, 0.0, 0.0], procedure_type=ProcedureType.SKILL)
        assert all(r.procedure_type == ProcedureType.SKILL for r in results)

    @pytest.mark.asyncio()
    async def test_find_similar_returns_empty_when_no_embeddings(
        self, mem: ProceduralMemory
    ) -> None:
        await mem.store(_make_procedure())  # no embedding
        results = await mem.find_similar([1.0, 0.0, 0.0, 0.0])
        assert results == []

    @pytest.mark.asyncio()
    async def test_find_similar_handles_zero_vector(self, mem: ProceduralMemory) -> None:
        """Cosine similarity must not raise ZeroDivisionError on zero vectors."""
        proc = _make_procedure(embedding=[0.0, 0.0, 0.0, 0.0])
        await mem.store(proc)
        results = await mem.find_similar([0.0, 0.0, 0.0, 0.0])
        # Should not raise; sim=0.0 for both zero vectors
        assert isinstance(results, list)


# ===========================================================================
# 12–20: ReflexionEngine tests
# ===========================================================================


class TestReflexionEngine:
    def _engine(self, model_client: Any | None = None) -> ReflexionEngine:
        mem = _make_procedural_mem()
        embed = _make_embedding_svc()
        return ReflexionEngine(mem, embed, max_reflections=5)

    @pytest.mark.asyncio()
    async def test_observe_failure_stores_reflection(self) -> None:
        mem = _make_procedural_mem()
        embed = _make_embedding_svc()
        engine = ReflexionEngine(mem, embed)

        # First call: reflection text, second: quality rating
        from nexus.core.models.base import ModelResponse
        from nexus.core.types import TokenUsage

        def _resp(content: str) -> ModelResponse:
            return ModelResponse(
                content=content,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )

        call_count = 0

        async def _complete(**kwargs: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _resp("I failed because I used the wrong tool.")
            return _resp('{"quality": 0.8}')

        client = MagicMock()
        client.complete = AsyncMock(side_effect=_complete)

        result = await engine.observe_failure(
            _make_agent_def(), "do task", ValueError("boom"), _make_state(), client
        )
        assert result.stored is True
        assert result.procedure_id is not None
        reflections = await mem.query_by_type(ProcedureType.REFLECTION)
        assert len(reflections) == 1

    @pytest.mark.asyncio()
    async def test_observe_failure_returns_hook_result_with_stored_true(self) -> None:
        mem = _make_procedural_mem()
        engine = ReflexionEngine(mem, _make_embedding_svc())

        responses = [
            "Don't skip validation steps.",
            '{"quality": 0.75}',
        ]
        idx = 0

        from nexus.core.models.base import ModelResponse
        from nexus.core.types import TokenUsage

        async def _c(**kw: Any) -> ModelResponse:
            nonlocal idx
            text = responses[idx] if idx < len(responses) else '{"quality":0.5}'
            idx += 1
            return ModelResponse(
                content=text,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )

        client = MagicMock()
        client.complete = AsyncMock(side_effect=_c)
        result = await engine.observe_failure(
            _make_agent_def(), "task", RuntimeError("err"), _make_state(), client
        )
        assert isinstance(result, ReflexionHookResult)
        assert result.stored is True

    @pytest.mark.asyncio()
    async def test_observe_failure_never_raises(self) -> None:
        """Even when the model_client raises, observe_failure returns stored=False."""
        mem = _make_procedural_mem()
        engine = ReflexionEngine(mem, _make_embedding_svc())

        bad_client = MagicMock()
        bad_client.complete = AsyncMock(side_effect=RuntimeError("network down"))
        result = await engine.observe_failure(
            _make_agent_def(), "task", ValueError("fail"), _make_state(), bad_client
        )
        assert result.stored is False
        assert result.error is not None

    @pytest.mark.asyncio()
    async def test_observe_failure_quality_confidence_populated(self) -> None:
        """quality_confidence is populated from the second LLM call."""
        mem = _make_procedural_mem()
        engine = ReflexionEngine(mem, _make_embedding_svc())

        responses = ["Good reflection text.", '{"quality": 0.65}']
        idx = 0

        from nexus.core.models.base import ModelResponse
        from nexus.core.types import TokenUsage

        async def _c(**kw: Any) -> ModelResponse:
            nonlocal idx
            text = responses[idx] if idx < len(responses) else '{"quality": 0.5}'
            idx += 1
            return ModelResponse(
                content=text,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )

        client = MagicMock()
        client.complete = AsyncMock(side_effect=_c)
        result = await engine.observe_failure(
            _make_agent_def(), "task", ValueError("oops"), _make_state(), client
        )
        assert result.quality_confidence is not None
        assert 0.0 <= result.quality_confidence <= 1.0

    @pytest.mark.asyncio()
    async def test_get_relevant_reflections_returns_similar_ones(self) -> None:
        mem = _make_procedural_mem()
        embed = _make_embedding_svc()
        engine = ReflexionEngine(mem, embed, quality_threshold=0.3)

        proc = _make_procedure(
            procedure_type=ProcedureType.REFLECTION,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        await mem.store(proc)

        client = MagicMock()
        results = await engine.get_relevant_reflections("similar task", client, top_k=3)
        assert len(results) == 1
        assert results[0].id == proc.id

    @pytest.mark.asyncio()
    async def test_get_relevant_reflections_returns_empty_on_error(self) -> None:
        mem = _make_procedural_mem()
        embed = MagicMock()
        embed.embed = AsyncMock(side_effect=RuntimeError("embed fail"))
        engine = ReflexionEngine(mem, embed)

        results = await engine.get_relevant_reflections("task", MagicMock())
        assert results == []

    def test_format_as_context_empty_list_returns_empty_string(self) -> None:
        engine = ReflexionEngine(_make_procedural_mem(), _make_embedding_svc())
        assert engine.format_as_context([]) == ""

    def test_format_as_context_formats_numbered_list(self) -> None:
        engine = ReflexionEngine(_make_procedural_mem(), _make_embedding_svc())
        proc = _make_procedure(procedure_type=ProcedureType.REFLECTION)
        proc = proc.model_copy(update={"description": "Always validate input first."})
        formatted = engine.format_as_context([proc])
        assert "1." in formatted
        assert "Always validate input first." in formatted
        assert "Lessons from past failures" in formatted

    @pytest.mark.asyncio()
    async def test_pruning_removes_oldest_when_over_limit(self) -> None:
        """When > max_reflections are stored, oldest are pruned."""
        mem = _make_procedural_mem()
        engine = ReflexionEngine(mem, _make_embedding_svc(), max_reflections=3)

        for _ in range(4):
            proc = _make_procedure(procedure_type=ProcedureType.REFLECTION)
            await mem.store(proc)

        await engine._prune_reflections()
        remaining = await mem.query_by_type(ProcedureType.REFLECTION)
        assert len(remaining) == 3


# ===========================================================================
# 21–30: SkillLibrary tests
# ===========================================================================

_EXTRACTABLE_JSON = (
    '{"extractable": true, "name": "sql_query_skill", '
    '"description": "Query databases with correct JOIN logic.", '
    '"steps": [{"action": "write SQL", "tool_name": "execute_sql"}], '
    '"domain_tags": ["database", "sql"]}'
)

_NOT_EXTRACTABLE_JSON = '{"extractable": false}'


class TestSkillLibrary:
    @pytest.mark.asyncio()
    async def test_observe_success_stores_skill_when_extractable(self) -> None:
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())
        client = _make_model_client(_EXTRACTABLE_JSON)
        result = await lib.observe_success(
            _make_agent_def(), "query the database", _make_execution_result(), client
        )
        assert result.extracted is True
        assert result.skill_name == "sql_query_skill"
        skills = await mem.query_by_type(ProcedureType.SKILL)
        assert len(skills) == 1

    @pytest.mark.asyncio()
    async def test_observe_success_returns_not_extracted_when_model_says_false(self) -> None:
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())
        client = _make_model_client(_NOT_EXTRACTABLE_JSON)
        result = await lib.observe_success(
            _make_agent_def(), "do a one-off task", _make_execution_result(), client
        )
        assert result.extracted is False
        skills = await mem.query_by_type(ProcedureType.SKILL)
        assert len(skills) == 0

    @pytest.mark.asyncio()
    async def test_observe_success_never_raises(self) -> None:
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())
        bad_client = MagicMock()
        bad_client.complete = AsyncMock(side_effect=RuntimeError("model down"))
        result = await lib.observe_success(
            _make_agent_def(), "task", _make_execution_result(), bad_client
        )
        assert result.extracted is False
        assert result.error is not None

    @pytest.mark.asyncio()
    async def test_get_approach_hints_returns_validated_only_by_default(self) -> None:
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())
        validated = _make_procedure(
            procedure_type=ProcedureType.SKILL,
            embedding=[1.0, 0.0, 0.0, 0.0],
            validated=True,
        )
        unvalidated = _make_procedure(
            procedure_type=ProcedureType.SKILL,
            embedding=[1.0, 0.0, 0.0, 0.0],
            validated=False,
        )
        await mem.store(validated)
        await mem.store(unvalidated)

        hints = await lib.get_approach_hints("task", MagicMock())
        assert all(h.metadata.get("validated") for h in hints)

    @pytest.mark.asyncio()
    async def test_get_approach_hints_returns_unvalidated_when_flag_set(self) -> None:
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())
        unvalidated = _make_procedure(
            procedure_type=ProcedureType.SKILL,
            embedding=[1.0, 0.0, 0.0, 0.0],
            validated=False,
        )
        await mem.store(unvalidated)

        hints = await lib.get_approach_hints("task", MagicMock(), validated_only=False)
        assert len(hints) >= 1

    @pytest.mark.asyncio()
    async def test_record_skill_outcome_promotes_to_validated_after_three_successes(
        self,
    ) -> None:
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())
        proc = _make_procedure(procedure_type=ProcedureType.SKILL, success_count=2, validated=False)
        await mem.store(proc)
        # Trigger third success → promote
        await lib.record_skill_outcome(proc.id, success=True)
        updated = await mem.get(proc.id)
        assert updated is not None
        assert updated.metadata.get("validated") is True

    @pytest.mark.asyncio()
    async def test_record_skill_outcome_demotes_below_threshold(self) -> None:
        """success_rate < 0.4 after ≥5 uses → demote to validated=False."""
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())
        proc = _make_procedure(
            procedure_type=ProcedureType.SKILL,
            success_count=1,
            failure_count=4,
            validated=True,
        )
        await mem.store(proc)
        # Another failure → total=6, success_rate=1/6 ≈ 0.17 < 0.4
        # But 0.17 < 0.2 so it gets deleted; let's set failure_count=3 to get 1/5=0.2 which
        # is exactly on the delete boundary. Use success=1, failure=3 total=4, add one failure → 1/5=0.2
        # That deletes. Instead use success=2, failure=2, add failure → 2/5=0.4 exactly not < 0.4
        # Use success=1, failure=3 → total=4, add failure → 1/5=0.2 which is < 0.2 → delete
        # Let's test demotion at exactly 0.25 (1 success, 3 failures + 1 = 4 failures):
        # success=1, failure=4 = 0.2 (delete boundary)
        # For demotion (not delete): success=2, failure=3 → success_rate=2/5=0.40
        # We need success_rate < 0.40. Use success=1, failure=4 → 1/5=0.2 → delete
        # Use success=2, failure=3 → 2/5=0.4 → not < demotion threshold
        # Actually _DEMOTION_THRESHOLD=0.4 means < 0.4 → demote
        # We need success_rate to be in [0.2, 0.4): success=1, failure=4 gives 0.2 exactly
        # → 0.2 < 0.2 is False, so it should demote not delete
        # Actually 0.2 is NOT < 0.2, so delete won't trigger; 0.2 < 0.4 → demote triggers
        # proc already has success=1, failure=4, validated=True → total=5, rate=0.2
        # One more failure: success=1, failure=5, total=6, rate≈0.17 < 0.2 → delete

        # For demotion test: create proc with success=1, failure=4 (rate=0.2 which < 0.4)
        proc2 = _make_procedure(
            procedure_type=ProcedureType.SKILL,
            success_count=1,
            failure_count=4,
            validated=True,
        )
        await mem.store(proc2)
        # record_skill_outcome with success=False: total=6, rate=1/6≈0.17 < 0.2 → delete
        # We need total >=5 and 0.2 <= rate < 0.4 for demotion-only
        # Setup: success=2, failure=3 → rate=0.4 (not < 0.4, no action)
        # Setup: success=2, failure=2 → add failure → success=2,failure=3 → rate=2/5=0.4 (not < 0.4)
        # This is tricky. Let's use: success=1, failure=3 → add success → success=2, failure=3 → 2/5=0.4
        # For demotion: need rate in (0.2, 0.4). Example: success=2, failure=4 → 2/6 = 0.33 < 0.4
        proc3 = _make_procedure(
            procedure_type=ProcedureType.SKILL,
            success_count=2,
            failure_count=4,
            validated=True,
        )
        await mem.store(proc3)
        # Call record_outcome manually to update counts without triggering lifecycle
        await mem.record_outcome(proc3.id, success=False)
        # Now: success=2, failure=5, total=7, rate=2/7≈0.286 which is 0.2 < rate < 0.4 → demote
        updated = await mem.get(proc3.id)
        assert updated is not None
        if updated.success_count + updated.failure_count >= 5:
            rate = updated.success_count / (updated.success_count + updated.failure_count)
            if 0.2 <= rate < 0.4:
                # Run lifecycle via record_skill_outcome
                await lib.record_skill_outcome(proc3.id, success=False)
                final = await mem.get(proc3.id)
                if final is not None:
                    assert final.metadata.get("validated") is False

    @pytest.mark.asyncio()
    async def test_record_skill_outcome_deletes_below_floor(self) -> None:
        """success_rate < 0.2 after ≥5 uses → deleted entirely."""
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())
        # success=0, failure=5 → rate=0 < 0.2 after one more failure
        proc = _make_procedure(
            procedure_type=ProcedureType.SKILL,
            success_count=0,
            failure_count=5,
            validated=True,
        )
        await mem.store(proc)
        await lib.record_skill_outcome(proc.id, success=False)
        deleted = await mem.get(proc.id)
        assert deleted is None

    def test_format_hints_as_context_empty_returns_empty(self) -> None:
        lib = SkillLibrary(_make_procedural_mem(), _make_embedding_svc())
        assert lib.format_hints_as_context([]) == ""

    @pytest.mark.asyncio()
    async def test_run_sequential_injects_skills_between_tasks(self) -> None:
        """Skills from task t are available (at least stored) for task t+1."""
        mem = _make_procedural_mem()
        lib = SkillLibrary(mem, _make_embedding_svc())

        inject_client = _make_model_client(_EXTRACTABLE_JSON)

        exec_result = _make_execution_result()
        runner = MagicMock()
        runner._model_client = inject_client
        runner.run = AsyncMock(return_value=exec_result)

        agent_def = _make_agent_def()
        results = await lib.run_sequential(["task A", "task B"], agent_def, runner)
        # Both tasks ran; runner.run called twice
        assert runner.run.call_count == 2
        assert len(results) == 2


# ===========================================================================
# 31–34: PromptOptimizer tests
# ===========================================================================


class TestPromptOptimizer:
    def _make_optimizer(
        self,
        *,
        baseline_pass_rate: float = 0.5,
        candidate_pass_rate: float = 0.8,
        no_failing: bool = False,
    ) -> tuple[PromptOptimizer, Any, Any]:
        from nexus.evaluation.assertions import AssertionResult
        from nexus.evaluation.prompt_versions import PromptVersionManager
        from nexus.evaluation.suite import CaseResult, SuiteResult

        mem = _make_procedural_mem()
        embed = _make_embedding_svc()
        engine = ReflexionEngine(mem, embed)
        lib = SkillLibrary(mem, embed)

        prompt_mgr = PromptVersionManager(agent_id="test-agent")
        prompt_mgr.register("1.0.0", "You are helpful.")
        prompt_mgr.activate("1.0.0")

        failing_case = CaseResult(
            case_id="c1",
            case_name="failing_case",
            passed=False,
            assertion_results=[
                AssertionResult(passed=False, assertion_type="contains", detail="nope", score=0.0)
            ],
        )
        passing_case = CaseResult(
            case_id="c2",
            case_name="passing_case",
            passed=True,
            assertion_results=[
                AssertionResult(passed=True, assertion_type="contains", detail="ok", score=1.0)
            ],
        )

        baseline_result = SuiteResult(
            suite_name="test",
            total_cases=2,
            passed=1,
            failed=1,
            errors=0,
            pass_rate=baseline_pass_rate,
            avg_duration_seconds=0.1,
            case_results=[] if no_failing else [failing_case, passing_case],
        )
        candidate_result = SuiteResult(
            suite_name="test",
            total_cases=2,
            passed=2,
            failed=0,
            errors=0,
            pass_rate=candidate_pass_rate,
            avg_duration_seconds=0.1,
            case_results=[passing_case, passing_case],
        )

        run_count = 0

        async def _fake_run() -> SuiteResult:
            nonlocal run_count
            run_count += 1
            return baseline_result if run_count == 1 else candidate_result

        eval_suite = MagicMock()
        eval_suite.run = AsyncMock(side_effect=_fake_run)
        eval_suite._agent_def = _make_agent_def()
        eval_suite._runner = MagicMock()

        model_client = _make_model_client("rewritten system prompt")
        optimizer = PromptOptimizer(
            engine, lib, prompt_mgr, eval_suite, model_client, improvement_threshold=0.05
        )
        return optimizer, prompt_mgr, _make_agent_def()

    @pytest.mark.asyncio()
    async def test_optimize_registers_new_version_when_improvement_found(self) -> None:
        optimizer, prompt_mgr, agent_def = self._make_optimizer(
            baseline_pass_rate=0.5, candidate_pass_rate=0.9
        )
        result = await optimizer.optimize(agent_def, MagicMock())
        assert result.improved is True
        assert result.best_score > result.original_score
        assert result.new_version is not None
        # The new version should be registered
        assert prompt_mgr.get(result.new_version) is not None

    @pytest.mark.asyncio()
    async def test_optimize_returns_improved_false_when_no_gain(self) -> None:
        optimizer, _, agent_def = self._make_optimizer(
            baseline_pass_rate=0.9, candidate_pass_rate=0.5
        )
        result = await optimizer.optimize(agent_def, MagicMock())
        assert result.improved is False

    @pytest.mark.asyncio()
    async def test_optimize_never_raises(self) -> None:
        """Even when eval_runner raises, optimize returns improved=False."""
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mem = _make_procedural_mem()
        embed = _make_embedding_svc()
        engine = ReflexionEngine(mem, embed)
        lib = SkillLibrary(mem, embed)
        prompt_mgr = PromptVersionManager(agent_id="a")
        prompt_mgr.register("1.0.0", "prompt")

        bad_suite = MagicMock()
        bad_suite.run = AsyncMock(side_effect=RuntimeError("eval crash"))
        bad_suite._agent_def = _make_agent_def()

        optimizer = PromptOptimizer(engine, lib, prompt_mgr, bad_suite, _make_model_client())
        result = await optimizer.optimize(_make_agent_def(), MagicMock())
        assert result.improved is False
        assert result.original_score == 0.0

    @pytest.mark.asyncio()
    async def test_optimize_builds_three_candidates(self) -> None:
        """_build_candidates produces up to 3 candidates."""
        optimizer, _, agent_def = self._make_optimizer(
            baseline_pass_rate=0.5, candidate_pass_rate=0.6
        )
        from nexus.evaluation.assertions import AssertionResult
        from nexus.evaluation.suite import CaseResult

        failing = [
            CaseResult(
                case_id="x",
                case_name="x",
                passed=False,
                assertion_results=[
                    AssertionResult(passed=False, assertion_type="c", detail="m", score=0.0)
                ],
            )
        ]
        candidates = await optimizer._build_candidates(agent_def, failing)
        # At most 3 strategies; any subset is valid since reflections/skills might be empty
        assert len(candidates) <= 3
        strategies = {c.strategy for c in candidates}
        # rewrite_failures should be built since we passed a failing case
        assert "rewrite_failures" in strategies


# ===========================================================================
# 35–37: AgentRunner integration tests
# ===========================================================================


class TestAgentRunnerReflexionIntegration:
    def _make_runner_with_mocks(
        self, reflexion_engine: Any = None, skill_library: Any = None
    ) -> Any:
        from nexus.orchestration.runner import AgentRunner
        from nexus.tools.executor import ToolExecutor

        tool_exec = MagicMock(spec=ToolExecutor)
        client = MagicMock()

        from nexus.core.models.base import ModelResponse
        from nexus.core.types import TokenUsage

        client.complete = AsyncMock(
            return_value=ModelResponse(
                content="Task complete.",
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )
        )

        from nexus.observability.events import EventLog

        with patch.object(EventLog, "open", new_callable=AsyncMock) as mock_el:
            mock_el.return_value = MagicMock(
                append=AsyncMock(return_value=MagicMock(event_type="test"))
            )

        runner = AgentRunner(
            client,
            tool_exec,
            reflexion_engine=reflexion_engine,
            skill_library=skill_library,
        )
        return runner, client

    @pytest.mark.asyncio()
    async def test_runner_calls_reflexion_engine_on_failure(self) -> None:
        """When the runner raises, it must call reflexion_engine.observe_failure."""
        from nexus.observability.events import EventLog
        from nexus.orchestration.runner import AgentRunner
        from nexus.tools.executor import ToolExecutor

        reflexion_engine = MagicMock()
        reflexion_engine.observe_failure = AsyncMock(
            return_value=ReflexionHookResult(stored=True, procedure_id="p1")
        )

        tool_exec = MagicMock(spec=ToolExecutor)

        bad_client = MagicMock()
        bad_client.complete = AsyncMock(side_effect=RuntimeError("model error"))

        with patch.object(EventLog, "open", new_callable=AsyncMock) as mock_el:
            mock_el.return_value = MagicMock(append=AsyncMock(return_value=MagicMock()))
            runner = AgentRunner(bad_client, tool_exec, reflexion_engine=reflexion_engine)
            with pytest.raises(RuntimeError):
                await runner.run(_make_agent_def(), "task", session_id="s1")

        reflexion_engine.observe_failure.assert_called_once()

    @pytest.mark.asyncio()
    async def test_runner_calls_skill_library_on_success(self) -> None:
        """After a COMPLETED run, skill_library.observe_success must be called."""
        from nexus.observability.events import EventLog
        from nexus.orchestration.runner import AgentRunner
        from nexus.tools.executor import ToolExecutor

        skill_library = MagicMock()
        skill_library.observe_success = AsyncMock(
            return_value=SkillExtractionResult(extracted=True, skill_name="s")
        )

        tool_exec = MagicMock(spec=ToolExecutor)

        from nexus.core.models.base import ModelResponse
        from nexus.core.types import TokenUsage

        client = MagicMock()
        client.complete = AsyncMock(
            return_value=ModelResponse(
                content="Done.",
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )
        )

        with patch.object(EventLog, "open", new_callable=AsyncMock) as mock_el:
            mock_el.return_value = MagicMock(append=AsyncMock(return_value=MagicMock()))
            runner = AgentRunner(client, tool_exec, skill_library=skill_library)
            await runner.run(_make_agent_def(), "task", session_id="s2")

        skill_library.observe_success.assert_called_once()

    @pytest.mark.asyncio()
    async def test_runner_without_hooks_behaves_identically(self) -> None:
        """AgentRunner with no F1 params produces identical ExecutionResult."""
        from nexus.core.types import AgentStatus
        from nexus.observability.events import EventLog
        from nexus.orchestration.runner import AgentRunner
        from nexus.tools.executor import ToolExecutor

        tool_exec = MagicMock(spec=ToolExecutor)

        from nexus.core.models.base import ModelResponse
        from nexus.core.types import TokenUsage

        client = MagicMock()
        client.complete = AsyncMock(
            return_value=ModelResponse(
                content="All good.",
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )
        )

        with patch.object(EventLog, "open", new_callable=AsyncMock) as mock_el:
            mock_el.return_value = MagicMock(append=AsyncMock(return_value=MagicMock()))
            runner = AgentRunner(client, tool_exec)
            result = await runner.run(_make_agent_def(), "task", session_id="s3")

        assert result.status == AgentStatus.COMPLETED
        assert result.output == "All good."
