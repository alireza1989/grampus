"""CausalWorldModel: persistent SCM populated by extractor, queried by inference."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any

from nexus.causal.extractor import CausalRelationExtractor
from nexus.causal.inference import SimpleCausalInference
from nexus.causal.types import (
    CausalDiagnosis,
    CausalRelation,
    InterventionQuery,
    InterventionResult,
    WorldModelGraph,
)
from nexus.core.logging import get_logger

_log = get_logger(__name__)
_WORLD_MODEL_ENTITY = "causal_world_model"


class CausalWorldModel:
    """Persistent SCM for one agent — populated by LLM labeling, queried by code.

    Follows the Causal-aware LLMs pattern (IJCAI 2025, arXiv 2505.24710):
    the LLM extracts causal claims; SimpleCausalInference answers queries.

    The world model grows over sessions. Verified causal edges from
    CausalTracer.diagnose() can be fed back via absorb_diagnosis() to
    strengthen the model with structurally validated evidence.

    Storage: single Dapr key per agent (same pattern as F3 MemoryGraph).

    Args:
        state_store: DaprStateStore for persistence.
        extractor: CausalRelationExtractor for LLM labeling.
        agent_id: Scopes to this agent.
    """

    def __init__(
        self,
        state_store: Any,
        extractor: CausalRelationExtractor,
        *,
        agent_id: str,
    ) -> None:
        self._store = state_store
        self._extractor = extractor
        self._agent_id = agent_id

    async def observe(self, text: str, *, session_id: str) -> list[CausalRelation]:
        """Extract causal relations from text and add them to the world model.

        Returns extracted relations (may be empty). Never raises.
        """
        if not text or not text.strip():
            return []
        try:
            relations = await self._extractor.extract(
                text, session_id=session_id, agent_id=self._agent_id
            )
            if relations:
                graph = await self.load()
                for rel in relations:
                    self._add_relation(graph, rel)
                graph.version += 1
                graph.last_updated = datetime.now(UTC)
                await self.save(graph)
            return relations
        except Exception:
            _log.warning("causal_world_model_observe_failed", agent_id=self._agent_id)
            return []

    async def query_intervention(self, query: InterventionQuery) -> InterventionResult:
        """Answer P(outcome | do(target = value)).

        Loads the current world model, runs SimpleCausalInference.
        Returns low-confidence result on error. Never raises.
        """
        try:
            graph = await self.load()
            inference = SimpleCausalInference(graph)
            return inference.intervene(query)
        except Exception:
            _log.warning("causal_world_model_query_failed", agent_id=self._agent_id)
            return InterventionResult(
                query=query,
                answer="Query failed — world model unavailable.",
                causal_path=[],
                confidence=0.0,
                explanation="Internal error loading world model.",
                is_identifiable=False,
            )

    async def absorb_diagnosis(self, diagnosis: CausalDiagnosis) -> None:
        """Add structurally validated causal edges from a failure diagnosis.

        Each step in a root cause candidate's causal_chain becomes a
        CausalRelation in the world model. This strengthens the model
        with evidence that bypasses LLM extraction uncertainty.
        Never raises.
        """
        try:
            if not diagnosis.root_causes:
                return
            graph = await self.load()
            for candidate in diagnosis.root_causes[:1]:
                chain = candidate.causal_chain
                for i in range(len(chain) - 1):
                    src_meta = diagnosis.causal_graph.nodes.get(chain[i], {})
                    tgt_meta = diagnosis.causal_graph.nodes.get(chain[i + 1], {})
                    rel = CausalRelation(
                        relation_id=str(uuid.uuid4()),
                        cause_description=str(src_meta.get("description", chain[i])),
                        effect_description=str(tgt_meta.get("description", chain[i + 1])),
                        relation_type="caused",
                        confidence=candidate.composite_score,
                        evidence_text=(
                            f"Structurally derived from failure diagnosis "
                            f"session={diagnosis.session_id}"
                        ),
                        session_id=diagnosis.session_id,
                        agent_id=self._agent_id,
                    )
                    self._add_relation(graph, rel)
            graph.version += 1
            graph.last_updated = datetime.now(UTC)
            await self.save(graph)
            _log.debug(
                "causal_world_model_absorbed_diagnosis",
                agent_id=self._agent_id,
                session_id=diagnosis.session_id,
            )
        except Exception:
            _log.warning("causal_world_model_absorb_failed", agent_id=self._agent_id)

    async def load(self) -> WorldModelGraph:
        """Load from Dapr or return new empty WorldModelGraph."""
        result, _ = await self._store.get(_WORLD_MODEL_ENTITY, self._agent_id, WorldModelGraph)
        if result is None:
            return WorldModelGraph(agent_id=self._agent_id)
        return WorldModelGraph.model_validate(result) if not isinstance(result, WorldModelGraph) else result

    async def save(self, graph: WorldModelGraph) -> None:
        """Persist WorldModelGraph to Dapr."""
        await self._store.save(_WORLD_MODEL_ENTITY, self._agent_id, graph)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_relation(self, graph: WorldModelGraph, rel: CausalRelation) -> None:
        """Add a CausalRelation and update the adjacency map. Mutates graph."""
        cause_id = _slugify(rel.cause_description)
        effect_id = _slugify(rel.effect_description)
        graph.variables.setdefault(cause_id, rel.cause_description)
        graph.variables.setdefault(effect_id, rel.effect_description)
        existing = graph.adjacency.setdefault(cause_id, [])
        if effect_id not in existing:
            existing.append(effect_id)
        graph.relations.append(rel)


def _slugify(text: str) -> str:
    """Convert description to a stable variable ID slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return slug[:48] or "var"
