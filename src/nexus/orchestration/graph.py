"""Graph execution engine with checkpoint/restore and parallel branch support."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from nexus.core.errors import OrchestrationError
from nexus.core.logging import get_logger
from nexus.core.types import AgentState

_log = get_logger(__name__)

NodeHandler = Callable[[AgentState], Coroutine[Any, Any, AgentState]]
EdgeCondition = Callable[[AgentState], Coroutine[Any, Any, str]]

_CHECKPOINT_ENTITY = "orchestration"
_MAX_STEPS_DEFAULT = 100


class GraphNode(BaseModel):
    """A named node in the execution graph."""

    name: str
    handler: NodeHandler
    is_entry: bool = False

    model_config = {"arbitrary_types_allowed": True}


class GraphEdge(BaseModel):
    """A directed edge between two nodes, optionally guarded by a condition."""

    from_node: str
    to_node: str | None
    condition: EdgeCondition | None = None
    routes: dict[str, str | None] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class GraphCheckpoint(BaseModel):
    """Persisted mid-execution snapshot of a graph run."""

    graph_id: str
    current_node: str
    state: AgentState
    completed_nodes: list[str] = Field(default_factory=list)
    created_at: datetime


class Graph:
    """Directed graph of async node handlers with checkpoint/restore support.

    Build with the builder methods, then call execute().
    """

    def __init__(
        self,
        *,
        graph_id: str,
        state_store: Any | None = None,
        max_steps: int = _MAX_STEPS_DEFAULT,
    ) -> None:
        self._graph_id = graph_id
        self._state_store = state_store
        self._max_steps = max_steps
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_node(self, name: str, handler: NodeHandler, *, entry: bool = False) -> Graph:
        """Register a node. Returns self for chaining."""
        self._nodes[name] = GraphNode(name=name, handler=handler, is_entry=entry)
        return self

    def add_edge(self, from_node: str, to_node: str | None) -> Graph:
        """Add an unconditional edge. to_node=None marks the terminal."""
        self._edges.append(GraphEdge(from_node=from_node, to_node=to_node))
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        condition: EdgeCondition,
        routes: dict[str, str | None],
    ) -> Graph:
        """Add a conditional edge. condition() returns a key into routes."""
        self._edges.append(
            GraphEdge(from_node=from_node, to_node=None, condition=condition, routes=routes)
        )
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, initial_state: AgentState) -> AgentState:
        """Run the graph from the entry node to completion.

        Raises:
            OrchestrationError: On missing entry, unknown node, or step limit.
        """
        entry = self._find_entry_node()
        return await self._run(entry, initial_state, completed_nodes=[])

    async def restore_and_execute(self, graph_id: str) -> AgentState | None:
        """Load the latest checkpoint and resume execution from where it stopped.

        Returns:
            Final AgentState, or None if no checkpoint found.
        """
        if self._state_store is None:
            return None
        checkpoint_id = f"graph:{graph_id}:checkpoint"
        checkpoint_obj, _ = await self._state_store.get(
            _CHECKPOINT_ENTITY, checkpoint_id, GraphCheckpoint
        )
        if checkpoint_obj is None:
            return None
        return await self._run(
            checkpoint_obj.current_node,
            checkpoint_obj.state,
            completed_nodes=list(checkpoint_obj.completed_nodes),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_entry_node(self) -> str:
        for name, node in self._nodes.items():
            if node.is_entry:
                return name
        raise OrchestrationError(
            "No entry node defined — call add_node(..., entry=True)",
            code="NO_ENTRY_NODE",
            hint="Register the node with graph.add_node() before adding edges that reference it.",
        )

    async def _run(
        self,
        start_node: str,
        state: AgentState,
        completed_nodes: list[str],
    ) -> AgentState:
        """Execute linearly from start_node until a terminal edge or max_steps."""
        current: str | None = start_node
        steps = 0

        while current is not None:
            if steps >= self._max_steps:
                raise OrchestrationError(
                    f"Execution exceeded {self._max_steps} steps (MAX_STEPS_EXCEEDED)",
                    code="MAX_STEPS_EXCEEDED",
                    hint="Ensure all edge targets are valid node names already added to the graph.",
                )
            if current not in self._nodes:
                raise OrchestrationError(
                    f"Unknown node '{current}'",
                    code="UNKNOWN_NODE",
                    hint="Ensure all edge targets are valid node names already added to the graph.",
                )

            node = self._nodes[current]
            _log.debug("graph.node_start", graph_id=self._graph_id, node=current, step=steps)
            state = await node.handler(state)
            completed_nodes.append(current)
            steps += 1

            if self._state_store is not None:
                await self._save_checkpoint(current, state, completed_nodes)

            current, state = await self._resolve_next(current, state, completed_nodes)

        return state

    async def _resolve_next(
        self,
        node_name: str,
        state: AgentState,
        completed_nodes: list[str],
    ) -> tuple[str | None, AgentState]:
        """Return (next_node_name, updated_state). None means terminal."""
        outgoing = [e for e in self._edges if e.from_node == node_name]
        unconditional = [e for e in outgoing if e.condition is None]
        conditional = [e for e in outgoing if e.condition is not None]

        if len(unconditional) > 1:
            state = await self._run_parallel(unconditional, state, completed_nodes)
            return None, state

        if unconditional:
            return unconditional[0].to_node, state

        if conditional:
            edge = conditional[0]
            cond = edge.condition
            assert cond is not None
            route_key = await cond(state)
            next_node = edge.routes.get(route_key)
            return next_node, state

        return None, state

    async def _run_parallel(
        self,
        edges: list[GraphEdge],
        state: AgentState,
        _completed_nodes: list[str],
    ) -> AgentState:
        """Run multiple branch targets concurrently and merge their results."""
        tasks = [
            self._run(edge.to_node, state.model_copy(deep=True), [])
            for edge in edges
            if edge.to_node is not None
        ]
        if not tasks:
            return state
        results = await asyncio.gather(*tasks)
        return _merge_states(state, list(results))

    async def _save_checkpoint(
        self,
        current_node: str,
        state: AgentState,
        completed_nodes: list[str],
    ) -> None:
        checkpoint = GraphCheckpoint(
            graph_id=self._graph_id,
            current_node=current_node,
            state=state,
            completed_nodes=list(completed_nodes),
            created_at=datetime.now(UTC),
        )
        checkpoint_id = f"graph:{self._graph_id}:checkpoint"
        store = self._state_store
        assert store is not None
        await store.save(_CHECKPOINT_ENTITY, checkpoint_id, checkpoint)


def _merge_states(base: AgentState, results: list[AgentState]) -> AgentState:
    """Merge parallel branch results: all messages appended, metadata merged, latest status."""
    merged = base.model_copy(deep=True)
    for result in results:
        merged.messages.extend(result.messages)
        merged.metadata.update(result.metadata)
        merged.status = result.status
    return merged
