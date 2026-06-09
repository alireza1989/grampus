"""CausalTracer: event log → causal graph → root cause diagnosis.

Based on AgentTrace (arXiv 2603.14688) and CHIEF (arXiv 2602.23701).
No LLM inference required — purely structural analysis.
"""

from __future__ import annotations

import uuid
from collections import deque
from typing import Any

from nexus.causal.types import (
    CausalDiagnosis,
    CausalEdge,
    CausalGraph,
    EdgeType,
    RootCauseCandidate,
)
from nexus.core.logging import get_logger

_log = get_logger(__name__)
_MAX_BACKWARD_DEPTH = 8
# Minimum shared prefix length to infer a data dependency edge.
# Intentionally simple substring heuristic; token-level overlap is a future improvement.
_DATA_DEP_MIN_OVERLAP_CHARS = 20


class CausalTracer:
    """Reconstructs causal graphs from execution event logs.

    Implements the AgentTrace three-edge-type model:
    - SEQUENTIAL: consecutive events in the same session
    - DATA_DEPENDENCY: A's output_snippet prefix appears in B's input_snippet
    - FAILURE_CASCADE: error in A precedes failure in B within 3 steps

    Root cause composite score:
        structural = 1 / (1 + downstream_count)
        positional = 1 - (step_index / max(total_steps - 1, 1))
        composite  = 0.6 * structural + 0.4 * positional

    Args:
        event_store: Must expose
            ``get_events_for_session(session_id, agent_id) -> list[dict]``.
    """

    def __init__(self, event_store: Any) -> None:
        self._events = event_store

    async def trace_session(self, session_id: str, agent_id: str) -> CausalGraph:
        """Build a full causal graph from all events in this session.

        Returns empty CausalGraph on error or empty event log. Never raises.
        """
        try:
            events = await self._events.get_events_for_session(session_id, agent_id)
            return self._build_graph(session_id, agent_id, events)
        except Exception:
            _log.warning("causal_trace_failed", session_id=session_id)
            return CausalGraph(graph_id=session_id, agent_id=agent_id)

    async def diagnose(
        self,
        session_id: str,
        agent_id: str,
        *,
        failure_event_id: str,
    ) -> CausalDiagnosis:
        """Diagnose the root cause of a known failure event.

        Algorithm:
        1. Build the full session causal graph.
        2. BFS backward from failure_event_id up to _MAX_BACKWARD_DEPTH.
        3. Score and rank all reached ancestors as root cause candidates.

        Returns empty CausalDiagnosis on error. Never raises.
        """
        try:
            graph = await self.trace_session(session_id, agent_id)
            candidate_ids = self._backward_bfs(graph, failure_event_id)
            candidates = self._rank_candidates(graph, failure_event_id, candidate_ids)
            candidates.sort(key=lambda c: c.composite_score, reverse=True)
            return CausalDiagnosis(
                session_id=session_id,
                agent_id=agent_id,
                failure_event_id=failure_event_id,
                root_causes=candidates,
                causal_graph=graph,
            )
        except Exception:
            _log.warning("causal_diagnose_failed", session_id=session_id)
            empty_graph = CausalGraph(graph_id=session_id, agent_id=agent_id)
            return CausalDiagnosis(
                session_id=session_id,
                agent_id=agent_id,
                failure_event_id=failure_event_id,
                causal_graph=empty_graph,
            )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(
        self, session_id: str, agent_id: str, events: list[dict[str, Any]]
    ) -> CausalGraph:
        nodes = {
            e["event_id"]: {
                "event_type": e.get("event_type", "unknown"),
                "description": e.get("output_snippet") or e.get("event_type", ""),
                "failed": bool(e.get("failed", False)),
                "step_index": int(e.get("step_index", 0)),
            }
            for e in events
        }
        edges: list[CausalEdge] = []
        edges.extend(self._build_sequential_edges(events))
        edges.extend(self._build_data_dependency_edges(events))
        edges.extend(self._build_failure_cascade_edges(events))
        return CausalGraph(
            graph_id=session_id,
            agent_id=agent_id,
            nodes=nodes,
            edges=edges,
        )

    def _build_sequential_edges(self, events: list[dict[str, Any]]) -> list[CausalEdge]:
        sorted_events = sorted(events, key=lambda e: e.get("step_index", 0))
        return [
            CausalEdge(
                edge_id=str(uuid.uuid4()),
                source_event_id=sorted_events[i]["event_id"],
                target_event_id=sorted_events[i + 1]["event_id"],
                edge_type=EdgeType.SEQUENTIAL,
                evidence="consecutive execution steps",
            )
            for i in range(len(sorted_events) - 1)
        ]

    def _build_data_dependency_edges(self, events: list[dict[str, Any]]) -> list[CausalEdge]:
        edges: list[CausalEdge] = []
        sorted_events = sorted(events, key=lambda e: e.get("step_index", 0))
        for i, a in enumerate(sorted_events):
            a_out = (a.get("output_snippet") or "").strip()
            if len(a_out) < _DATA_DEP_MIN_OVERLAP_CHARS:
                continue
            prefix = a_out[:_DATA_DEP_MIN_OVERLAP_CHARS]
            for b in sorted_events[i + 1 :]:
                b_in = (b.get("input_snippet") or "").strip()
                if prefix in b_in:
                    edges.append(
                        CausalEdge(
                            edge_id=str(uuid.uuid4()),
                            source_event_id=a["event_id"],
                            target_event_id=b["event_id"],
                            edge_type=EdgeType.DATA_DEPENDENCY,
                            weight=1.5,
                            evidence=(
                                f"output of step {a.get('step_index')} "
                                f"consumed by step {b.get('step_index')}"
                            ),
                        )
                    )
        return edges

    def _build_failure_cascade_edges(self, events: list[dict[str, Any]]) -> list[CausalEdge]:
        edges: list[CausalEdge] = []
        sorted_events = sorted(events, key=lambda e: e.get("step_index", 0))
        for i, a in enumerate(sorted_events):
            if not a.get("error_message"):
                continue
            window = sorted_events[i + 1 : i + 4]
            for b in window:
                if b.get("failed"):
                    edges.append(
                        CausalEdge(
                            edge_id=str(uuid.uuid4()),
                            source_event_id=a["event_id"],
                            target_event_id=b["event_id"],
                            edge_type=EdgeType.FAILURE_CASCADE,
                            weight=2.0,
                            evidence=(
                                f"error in step {a.get('step_index')} "
                                f"preceded failure in step {b.get('step_index')}"
                            ),
                        )
                    )
        return edges

    # ------------------------------------------------------------------
    # Root cause tracing
    # ------------------------------------------------------------------

    def _backward_bfs(self, graph: CausalGraph, failure_event_id: str) -> list[str]:
        """BFS backward from failure_event_id. Returns ancestor event_ids (excluding failure)."""
        reverse_adj: dict[str, list[str]] = {}
        for edge in graph.edges:
            reverse_adj.setdefault(edge.target_event_id, []).append(edge.source_event_id)

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(failure_event_id, 0)])
        ancestors: list[str] = []

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited or depth > _MAX_BACKWARD_DEPTH:
                continue
            visited.add(node_id)
            if node_id != failure_event_id:
                ancestors.append(node_id)
            for parent in reverse_adj.get(node_id, []):
                if parent not in visited:
                    queue.append((parent, depth + 1))
        return ancestors

    def _rank_candidates(
        self,
        graph: CausalGraph,
        failure_event_id: str,
        candidate_ids: list[str],
    ) -> list[RootCauseCandidate]:
        total_steps = max((n.get("step_index", 0) for n in graph.nodes.values()), default=1)
        forward_adj: dict[str, list[str]] = {}
        for edge in graph.edges:
            forward_adj.setdefault(edge.source_event_id, []).append(edge.target_event_id)

        candidates: list[RootCauseCandidate] = []
        for cid in candidate_ids:
            node_meta = graph.nodes.get(cid, {})
            downstream = len(forward_adj.get(cid, []))
            step_index = int(node_meta.get("step_index", 0))
            structural = 1.0 / (1.0 + downstream)
            positional = 1.0 - (step_index / max(total_steps - 1, 1))
            composite = 0.6 * structural + 0.4 * positional
            chain = self._find_path(graph, cid, failure_event_id)
            candidates.append(
                RootCauseCandidate(
                    event_id=cid,
                    event_type=node_meta.get("event_type", "unknown"),
                    description=str(node_meta.get("description", "")),
                    structural_score=round(structural, 4),
                    positional_score=round(positional, 4),
                    composite_score=round(composite, 4),
                    causal_chain=chain,
                )
            )
        return candidates

    def _find_path(self, graph: CausalGraph, source: str, target: str) -> list[str]:
        """DFS to find one directed path from source to target. Returns [] if none."""
        forward_adj: dict[str, list[str]] = {}
        for edge in graph.edges:
            forward_adj.setdefault(edge.source_event_id, []).append(edge.target_event_id)

        stack = [(source, [source])]
        while stack:
            node, path = stack.pop()
            if node == target:
                return path
            for neighbor in forward_adj.get(node, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return []
