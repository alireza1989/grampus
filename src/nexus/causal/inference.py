"""SimpleCausalInference: pure-Python do-calculus for small agent world models.

Implements the backdoor criterion and adjustment for P(Y|do(X)) queries.
Suitable for agent world models with < 200 variables. No external libraries.
"""

from __future__ import annotations

import math
from collections import deque

from nexus.causal.types import InterventionQuery, InterventionResult, WorldModelGraph
from nexus.core.logging import get_logger

_log = get_logger(__name__)


class SimpleCausalInference:
    """Backdoor adjustment for interventional queries over simple DAGs.

    Given a WorldModelGraph with adjacency list, answers:
        P(outcome_variable | do(target_variable = value))

    For identifiable queries (a valid backdoor set exists), returns a
    qualitative answer with the adjustment set and causal path.
    For non-identifiable queries, returns is_identifiable=False.

    Args:
        world_model: WorldModelGraph to reason over. Treated as read-only.
    """

    def __init__(self, world_model: WorldModelGraph) -> None:
        self._graph = world_model

    def intervene(self, query: InterventionQuery) -> InterventionResult:
        """Answer a do-calculus intervention query.

        Returns InterventionResult. Never raises — returns low-confidence
        non-identifiable result on any error.
        """
        try:
            return self._compute_intervention(query)
        except Exception:
            _log.warning("causal_intervene_failed", query=query.natural_language)
            return InterventionResult(
                query=query,
                answer="Unable to determine causal effect.",
                causal_path=[],
                confidence=0.0,
                explanation="Inference failed; check world model structure.",
                is_identifiable=False,
            )

    def find_causal_paths(self, source: str, target: str) -> list[list[str]]:
        """DFS to find all directed paths from source to target. Returns []."""
        adj = self._graph.adjacency
        all_paths: list[list[str]] = []
        stack = [(source, [source])]
        while stack:
            node, path = stack.pop()
            if node == target:
                all_paths.append(path)
                continue
            for neighbor in adj.get(node, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return all_paths

    def find_backdoor_adjustment_set(self, treatment: str, outcome: str) -> list[str] | None:
        """Find a valid backdoor adjustment set Z for P(outcome|do(treatment)).

        A set Z satisfies the backdoor criterion if:
        1. Z blocks all backdoor paths (paths into treatment via parents)
        2. Z contains no descendants of treatment

        Returns list of variable IDs (may be empty — direct path, no backdoor),
        or None if the effect is not identifiable (no valid set found).
        """
        treatment_parents = self._get_parents(treatment)
        treatment_descendants = self._get_descendants(treatment)
        valid_z = [p for p in treatment_parents if p not in treatment_descendants and p != outcome]
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)
        if not backdoor_paths:
            return []
        if valid_z or not backdoor_paths:
            return valid_z
        return None

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm. Returns [] if a cycle is detected."""
        in_degree: dict[str, int] = {v: 0 for v in self._graph.variables}
        for sources in self._graph.adjacency.values():
            for target in sources:
                if target in in_degree:
                    in_degree[target] += 1
        queue: deque[str] = deque(v for v, d in in_degree.items() if d == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self._graph.adjacency.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        return order if len(order) == len(self._graph.variables) else []

    def is_dag(self) -> bool:
        """Return True if the world model has no directed cycles."""
        return len(self.topological_sort()) == len(self._graph.variables)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_intervention(self, query: InterventionQuery) -> InterventionResult:
        paths = self.find_causal_paths(query.target_variable, query.outcome_variable)
        if not paths:
            return InterventionResult(
                query=query,
                answer=(
                    f"No causal path found from '{query.target_variable}' "
                    f"to '{query.outcome_variable}'."
                ),
                causal_path=[],
                confidence=0.1,
                explanation="The world model contains no directed path between these variables.",
                is_identifiable=False,
            )
        shortest = min(paths, key=len)
        adjustment_set = self.find_backdoor_adjustment_set(
            query.target_variable, query.outcome_variable
        )
        if adjustment_set is None:
            return InterventionResult(
                query=query,
                answer="Causal effect is not identifiable from the current world model.",
                causal_path=shortest,
                confidence=0.2,
                explanation=(
                    "Backdoor criterion cannot be satisfied — the world model may be incomplete."
                ),
                is_identifiable=False,
            )
        path_desc = " → ".join(self._graph.variables.get(v, v) for v in shortest)
        target_desc = self._graph.variables.get(query.target_variable, query.target_variable)
        outcome_desc = self._graph.variables.get(query.outcome_variable, query.outcome_variable)
        answer = (
            f"Setting {target_desc} to '{query.intervention_value}' "
            f"causally affects {outcome_desc} via: {path_desc}."
        )
        explanation = f"Direct causal path exists ({len(shortest) - 1} hop(s)). " + (
            f"Backdoor adjustment applied for: {', '.join(adjustment_set)}."
            if adjustment_set
            else "No backdoor adjustment needed (no confounders found)."
        )
        confidence = round(1.0 / math.sqrt(len(shortest)), 3)
        return InterventionResult(
            query=query,
            answer=answer,
            causal_path=shortest,
            backdoor_vars=adjustment_set,
            backdoor_adjustment_applied=bool(adjustment_set),
            confidence=confidence,
            explanation=explanation,
            is_identifiable=True,
        )

    def _get_parents(self, var_id: str) -> list[str]:
        return [src for src, targets in self._graph.adjacency.items() if var_id in targets]

    def _get_descendants(self, var_id: str) -> set[str]:
        visited: set[str] = set()
        queue: deque[str] = deque([var_id])
        while queue:
            node = queue.popleft()
            for neighbor in self._graph.adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def _find_backdoor_paths(self, treatment: str, outcome: str) -> list[list[str]]:
        parents = self._get_parents(treatment)
        backdoor: list[list[str]] = []
        for parent in parents:
            sub_paths = self._find_paths_ignoring_direct(parent, outcome, blocked={treatment})
            backdoor.extend([[treatment] + p for p in sub_paths])
        return backdoor

    def _find_paths_ignoring_direct(
        self, source: str, target: str, blocked: set[str]
    ) -> list[list[str]]:
        adj = self._graph.adjacency
        all_paths: list[list[str]] = []
        stack = [(source, [source])]
        while stack:
            node, path = stack.pop()
            if node == target:
                all_paths.append(path)
                continue
            for neighbor in adj.get(node, []):
                if neighbor not in path and neighbor not in blocked:
                    stack.append((neighbor, path + [neighbor]))
        return all_paths
