"""GraphRetriever — traversal-based retrieval from the topic-associative-network."""

from __future__ import annotations

import contextlib
import math
from collections import deque
from typing import TYPE_CHECKING, Any

from grampus.core.logging import get_logger
from grampus.memory.graph.types import GraphQueryResult, MemoryGraph

if TYPE_CHECKING:
    from grampus.memory.graph.consolidator import SemanticConsolidator

_log = get_logger(__name__)

_DEFAULT_TRAVERSAL_DEPTH = 2
_DEFAULT_TOP_K_NODES = 5


class GraphRetriever:
    """Traversal-based retrieval from the topic-associative-network.

    Algorithm:
    1. Embed query
    2. Find top-k seed nodes by cosine similarity (entry points)
    3. BFS traversal up to traversal_depth hops
    4. Score each reached node: seed_score * edge_weight * (1 / (hop_distance + 1))
    5. Return top-k nodes by final score

    Args:
        consolidator: SemanticConsolidator for loading MemoryGraph.
        embedding_service: For embedding queries.
    """

    def __init__(
        self,
        consolidator: SemanticConsolidator,
        embedding_service: Any,
    ) -> None:
        self._consolidator = consolidator
        self._embeddings = embedding_service

    async def query(
        self,
        agent_id: str,
        query: str,
        *,
        top_k: int = _DEFAULT_TOP_K_NODES,
        traversal_depth: int = _DEFAULT_TRAVERSAL_DEPTH,
    ) -> GraphQueryResult:
        """Retrieve relevant concept nodes by graph traversal.

        Returns empty GraphQueryResult on error or empty graph.
        """
        with contextlib.suppress(Exception):
            graph = await self._consolidator.load_graph(agent_id)
            if not graph.nodes:
                return GraphQueryResult(nodes=[], query=query, traversal_depth=traversal_depth)

            query_embedding: list[float] | None = None
            with contextlib.suppress(Exception):
                query_embedding = await self._embeddings.embed(query)

            if query_embedding is None:
                return GraphQueryResult(nodes=[], query=query, traversal_depth=traversal_depth)

            seed_scores = _find_seed_nodes(graph, query_embedding, top_k=top_k)
            if not seed_scores:
                return GraphQueryResult(nodes=[], query=query, traversal_depth=traversal_depth)

            all_scores = self._bfs_from_seeds(graph, list(seed_scores.keys()), traversal_depth)
            for node_id, seed_score in seed_scores.items():
                if node_id in all_scores:
                    hop_score, hop_dist = all_scores[node_id]
                    all_scores[node_id] = (max(hop_score, seed_score), hop_dist)
                else:
                    all_scores[node_id] = (seed_score, 0)

            sorted_ids = sorted(all_scores.keys(), key=lambda nid: all_scores[nid][0], reverse=True)
            result_nodes = [graph.nodes[nid] for nid in sorted_ids[:top_k] if nid in graph.nodes]

            _log.debug(
                "graph_retrieved",
                agent_id=agent_id,
                query_len=len(query),
                returned=len(result_nodes),
            )
            return GraphQueryResult(
                nodes=result_nodes,
                query=query,
                traversal_depth=traversal_depth,
            )

        return GraphQueryResult(nodes=[], query=query, traversal_depth=traversal_depth)

    def _bfs_from_seeds(
        self,
        graph: MemoryGraph,
        seed_node_ids: list[str],
        max_depth: int,
    ) -> dict[str, tuple[float, int]]:
        """BFS traversal. Returns {node_id: (score, hop_distance)}."""
        visited: dict[str, tuple[float, int]] = {}
        queue: deque[tuple[str, int, float]] = deque()
        for node_id in seed_node_ids:
            queue.append((node_id, 0, 1.0))

        while queue:
            node_id, depth, score = queue.popleft()
            if node_id in visited or depth > max_depth:
                continue
            visited[node_id] = (score, depth)
            for edge in graph.edges:
                if edge.source_node_id == node_id:
                    child_score = score * edge.weight * (1.0 / (depth + 2))
                    queue.append((edge.target_node_id, depth + 1, child_score))

        return visited

    def format_as_context(self, result: GraphQueryResult) -> str:
        """Format graph query result as a readable string for context injection.

        Returns empty string if result.nodes is empty.
        """
        if not result.nodes:
            return ""

        lines = ["Knowledge graph context:"]
        for node in result.nodes:
            lines.append(f"- {node.label}: {node.description}")
            if node.metadata.get("category") == "schematic":
                lines[-1] += " [core concept]"

        return "\n".join(lines)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _find_seed_nodes(
    graph: MemoryGraph,
    query_embedding: list[float],
    top_k: int,
) -> dict[str, float]:
    """Return top-k nodes by cosine similarity to query_embedding."""
    scores: list[tuple[str, float]] = []
    for node_id, node in graph.nodes.items():
        if node.embedding is None:
            continue
        sim = _cosine_similarity(query_embedding, node.embedding)
        scores.append((node_id, sim))

    scores.sort(key=lambda t: t[1], reverse=True)
    return {node_id: score for node_id, score in scores[:top_k]}
