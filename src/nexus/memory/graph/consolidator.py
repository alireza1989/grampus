"""SemanticConsolidator — merges session EventGraph into persistent MemoryGraph."""

from __future__ import annotations

import contextlib
import json
import math
import uuid
from datetime import UTC, datetime
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.graph.types import (
    ConceptNode,
    EventGraph,
    MemoryGraph,
    RelationshipEdge,
)

_log = get_logger(__name__)

_GRAPH_ENTITY = "memory_graph"
_MERGE_SIMILARITY_THRESHOLD = 0.85
_EXTRACT_MAX_EVENTS_CHARS = 2000

EXTRACT_NODES_SYSTEM_PROMPT = (
    "You are a knowledge graph builder. "
    "Given a set of interaction events, extract key concepts and their relationships. "
    "Reply with only valid JSON: "
    '{"concepts": [{"label": "...", "description": "...", "relations": '
    '[{"target": "...", "type": "related_to|caused_by|precedes|contradicts"}]}]}. '
    "Extract only concepts that are clearly central to the interactions. "
    'Maximum 5 concepts per extraction. If nothing extractable, reply {"concepts": []}.'
)

EXTRACT_NODES_USER_TEMPLATE = (
    "Session events ({count} total):\n{events_summary}\n\nExtract concepts:"
)


class SemanticConsolidator:
    """Merges the session EventGraph into the persistent MemoryGraph (GAM).

    Only runs when a SemanticShiftEvent is detected or at session end.
    Prevents transient noise from contaminating stable knowledge.

    Args:
        state_store: DaprStateStore for persisting MemoryGraph.
        embedding_service: For embedding new nodes.
        model_client: LLM for concept extraction from event summaries.
    """

    def __init__(
        self,
        state_store: Any,
        embedding_service: Any,
        model_client: Any,
    ) -> None:
        self._store = state_store
        self._embeddings = embedding_service
        self._model_client = model_client

    async def consolidate(
        self,
        event_graph: EventGraph,
        agent_id: str,
    ) -> MemoryGraph:
        """Run consolidation. Returns updated MemoryGraph.

        Never raises — returns current (possibly unchanged) MemoryGraph on error.
        """
        graph = await self.load_graph(agent_id)
        if not event_graph.events:
            return graph

        with contextlib.suppress(Exception):
            concepts = await self._extract_concepts(event_graph, self._model_client)
            source_ids = [e.event_id for e in event_graph.events]

            for concept in concepts:
                label = concept.get("label", "")
                description = concept.get("description", "")
                if not label:
                    continue

                embedding: list[float] | None = None
                with contextlib.suppress(Exception):
                    embedding = await self._embeddings.embed(f"{label}: {description}")

                if embedding is None:
                    continue

                node_id = await self._merge_or_add_node(
                    graph, label, description, embedding, source_ids
                )

                for rel in concept.get("relations", []):
                    target_label = rel.get("target", "")
                    rel_type = rel.get("type", "related_to")
                    target_node_id = _find_node_by_label(graph, target_label)
                    if target_node_id and target_node_id != node_id:
                        edge = RelationshipEdge(
                            edge_id=str(uuid.uuid4()),
                            source_node_id=node_id,
                            target_node_id=target_node_id,
                            relation_type=rel_type,
                        )
                        graph.edges.append(edge)

            graph.version += 1
            graph.last_consolidated = datetime.now(UTC)
            await self.save_graph(graph)
            _log.debug(
                "graph_consolidated",
                agent_id=agent_id,
                version=graph.version,
                nodes=len(graph.nodes),
            )

        return graph

    async def load_graph(self, agent_id: str) -> MemoryGraph:
        """Load from Dapr or create empty MemoryGraph."""
        with contextlib.suppress(Exception):
            result, _ = await self._store.get(_GRAPH_ENTITY, agent_id, MemoryGraph)
            if result is not None:
                return result  # type: ignore[no-any-return]
        return MemoryGraph(graph_id=agent_id)

    async def save_graph(self, graph: MemoryGraph) -> None:
        """Persist MemoryGraph to Dapr. Key: graph_id (= agent_id)."""
        await self._store.save(_GRAPH_ENTITY, graph.graph_id, graph)

    async def _extract_concepts(
        self, event_graph: EventGraph, model_client: Any
    ) -> list[dict[str, Any]]:
        """LLM call to extract concepts from event summaries. Returns [] on error."""
        summary = self._build_events_summary(event_graph)
        if not summary.strip():
            return []

        from nexus.core.types import Message, Role  # noqa: PLC0415

        user_prompt = EXTRACT_NODES_USER_TEMPLATE.format(
            count=len(event_graph.events),
            events_summary=summary,
        )
        messages = [
            Message(role=Role.SYSTEM, content=EXTRACT_NODES_SYSTEM_PROMPT),
            Message(role=Role.USER, content=user_prompt),
        ]

        with contextlib.suppress(Exception):
            response = await model_client.complete(
                messages=messages,
                model=None,
                temperature=0.2,
                max_tokens=400,
            )
            content = response.content or ""
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                return parsed.get("concepts", [])  # type: ignore[no-any-return]

        return []

    async def _merge_or_add_node(
        self,
        graph: MemoryGraph,
        label: str,
        description: str,
        embedding: list[float],
        source_episode_ids: list[str],
    ) -> str:
        """Return node_id of merged-into or newly created node."""
        best_id: str | None = None
        best_sim = 0.0

        for existing_id, existing_node in graph.nodes.items():
            if existing_node.embedding is None:
                continue
            sim = _cosine_similarity(embedding, existing_node.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = existing_id

        if best_id is not None and best_sim >= _MERGE_SIMILARITY_THRESHOLD:
            node = graph.nodes[best_id]
            merged = node.model_copy(
                update={
                    "frequency": node.frequency + 1,
                    "description": description if description else node.description,
                    "last_updated": datetime.now(UTC),
                    "source_episode_ids": list(set(node.source_episode_ids + source_episode_ids)),
                }
            )
            graph.nodes[best_id] = merged
            return best_id

        new_id = str(uuid.uuid4())
        new_node = ConceptNode(
            node_id=new_id,
            label=label,
            description=description,
            embedding=embedding,
            source_episode_ids=source_episode_ids,
        )
        if new_node.frequency >= 5:
            new_node.metadata["category"] = "schematic"
        graph.nodes[new_id] = new_node
        return new_id

    def _build_events_summary(self, event_graph: EventGraph) -> str:
        """Concatenate event content_summaries for LLM prompt. Max 2000 chars."""
        parts = [e.content_summary for e in event_graph.events]
        summary = "\n".join(parts)
        return summary[:_EXTRACT_MAX_EVENTS_CHARS]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _find_node_by_label(graph: MemoryGraph, label: str) -> str | None:
    label_lower = label.lower()
    for node_id, node in graph.nodes.items():
        if node.label.lower() == label_lower:
            return node_id
    return None
