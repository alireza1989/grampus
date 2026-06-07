"""In-process Prometheus-compatible metrics collector."""

from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from nexus.core.logging import get_logger

logger = get_logger(__name__)

_HISTOGRAM_BUCKETS = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]


class MetricsSnapshot(BaseModel):
    """Point-in-time snapshot of all counters and gauges."""

    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_tool_calls: int = 0
    total_errors: int = 0
    active_agents: int = 0
    llm_call_count: int = 0
    per_model_tokens: dict[str, int] = Field(default_factory=dict)
    per_agent_cost: dict[str, float] = Field(default_factory=dict)


class CostSummary(BaseModel):
    """Cost breakdown by model and agent.

    Attributes:
        total_cost_usd: Aggregate USD cost recorded.
        total_tokens: Aggregate token count.
        total_llm_calls: Total LLM API calls.
        by_model: Per-model rows sorted by cost descending.
        by_agent: Per-agent rows sorted by cost descending.
    """

    total_cost_usd: float
    total_tokens: int
    total_llm_calls: int
    by_model: list[dict[str, Any]] = Field(default_factory=list)
    by_agent: list[dict[str, Any]] = Field(default_factory=list)


class _Histogram:
    """Latency histogram with fixed bucket boundaries.

    Keeps only the most recent ``_WINDOW`` samples to bound memory. A separate
    ``_total_count`` accumulates all-time call counts so the Prometheus
    ``_count`` metric remains accurate even after the window rolls over.
    """

    _WINDOW = 10_000

    def __init__(self) -> None:
        self._samples: deque[float] = deque(maxlen=self._WINDOW)
        self._total_count: int = 0
        self._total_sum: float = 0.0

    def observe(self, value: float) -> None:
        self._samples.append(value)
        self._total_count += 1
        self._total_sum += value

    @property
    def count(self) -> int:
        return self._total_count

    @property
    def total(self) -> float:
        return self._total_sum

    def bucket_counts(self) -> list[tuple[float, int]]:
        """Return (upper_bound, cumulative_count) pairs for each bucket."""
        result = []
        for bound in _HISTOGRAM_BUCKETS:
            cnt = sum(1 for s in self._samples if s <= bound)
            result.append((bound, cnt))
        result.append((float("inf"), self._total_count))
        return result

    def reset(self) -> None:
        self._samples.clear()
        self._total_count = 0
        self._total_sum = 0.0


def _prom_label(agent_id: str) -> str:
    return f'{{agent_id="{agent_id}"}}'


def _emit_counter(lines: list[str], name: str, help_text: str, label: str, value: float) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    lines.append(f"{name}{label} {value}")


def _emit_histogram(
    lines: list[str], name: str, help_text: str, label: str, hist: _Histogram
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} histogram")
    inner = label[1:-1]  # strip outer braces
    for bound, cnt in hist.bucket_counts():
        le = "+Inf" if bound == float("inf") else str(int(bound) if bound == int(bound) else bound)
        bucket_label = "{" + inner + f',le="{le}"' + "}"
        lines.append(f"{name}_bucket{bucket_label} {cnt}")
    lines.append(f"{name}_count{label} {hist.count}")
    lines.append(f"{name}_sum{label} {hist.total}")


class NexusMetrics:
    """In-process metrics collector with Prometheus-compatible text exposition.

    Does NOT require a running Prometheus server — stores everything in memory
    and exports to Prometheus text format on demand.

    Args:
        agent_id: Scopes per-agent metrics.
    """

    def __init__(self, *, agent_id: str) -> None:
        self._agent_id = agent_id
        self._label = _prom_label(agent_id)
        self.reset()

    def reset(self) -> None:
        """Reset all counters. Useful for testing."""
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._total_tool_calls: int = 0
        self._total_errors: int = 0
        self._active_agents: int = 0
        self._llm_call_count: int = 0
        self._per_model_tokens: dict[str, int] = {}
        self._per_model_cost: dict[str, float] = {}
        self._llm_latency = _Histogram()
        self._tool_latency = _Histogram()

    def record_llm_call(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """Increment token/cost/call counters. Record latency in histogram.

        Args:
            model: Model identifier.
            input_tokens: Prompt token count.
            output_tokens: Completion token count.
            cost_usd: Estimated USD cost.
            latency_ms: Round-trip latency in milliseconds.
        """
        total = input_tokens + output_tokens
        self._total_tokens += total
        self._total_cost += cost_usd
        self._llm_call_count += 1
        self._per_model_tokens[model] = self._per_model_tokens.get(model, 0) + total
        self._per_model_cost[model] = self._per_model_cost.get(model, 0.0) + cost_usd
        self._llm_latency.observe(latency_ms)
        logger.debug("llm_call_recorded", model=model, tokens=total, cost_usd=cost_usd)

    def record_tool_call(self, *, tool_name: str, success: bool, latency_ms: float) -> None:
        """Increment tool call counter. Record latency in histogram.

        Args:
            tool_name: Name of the invoked tool.
            success: Whether execution succeeded.
            latency_ms: Execution time in milliseconds.
        """
        self._total_tool_calls += 1
        self._tool_latency.observe(latency_ms)
        logger.debug("tool_call_recorded", tool_name=tool_name, success=success)

    def record_error(self, *, error_type: str) -> None:
        """Increment error counter.

        Args:
            error_type: Short class name of the error.
        """
        self._total_errors += 1
        logger.debug("error_recorded", error_type=error_type)

    def set_active_agents(self, count: int) -> None:
        """Update active agent gauge.

        Args:
            count: Current number of concurrently running agents.
        """
        self._active_agents = count

    def snapshot(self) -> MetricsSnapshot:
        """Return current accumulated metrics. Pure computation, no I/O."""
        per_agent_cost = {self._agent_id: self._total_cost} if self._total_cost else {}
        return MetricsSnapshot(
            total_tokens=self._total_tokens,
            total_cost_usd=self._total_cost,
            total_tool_calls=self._total_tool_calls,
            total_errors=self._total_errors,
            active_agents=self._active_agents,
            llm_call_count=self._llm_call_count,
            per_model_tokens=dict(self._per_model_tokens),
            per_agent_cost=per_agent_cost,
        )

    def get_cost_summary(self) -> CostSummary:
        """Return a structured cost breakdown by model and agent.

        Returns:
            CostSummary with by_model and by_agent rows sorted by cost descending.
        """
        total = self._total_cost
        by_model: list[dict[str, Any]] = sorted(
            [
                {
                    "model": model,
                    "tokens": self._per_model_tokens.get(model, 0),
                    "cost_usd": cost,
                    "pct": (cost / total * 100.0) if total > 0 else 0.0,
                }
                for model, cost in self._per_model_cost.items()
            ],
            key=lambda r: -r["cost_usd"],
        )
        by_agent: list[dict[str, Any]] = (
            [
                {
                    "agent_id": self._agent_id,
                    "cost_usd": total,
                    "pct": 100.0,
                }
            ]
            if total > 0
            else []
        )
        return CostSummary(
            total_cost_usd=total,
            total_tokens=self._total_tokens,
            total_llm_calls=self._llm_call_count,
            by_model=by_model,
            by_agent=by_agent,
        )

    def to_prometheus_text(self) -> str:
        """Export metrics in Prometheus text exposition format.

        Returns:
            Multiline string with # HELP, # TYPE, and metric lines.
        """
        lines: list[str] = []
        lbl = self._label
        _emit_counter(lines, "nexus_total_tokens", "Total tokens consumed", lbl, self._total_tokens)
        _emit_counter(lines, "nexus_total_cost_usd", "Total cost in USD", lbl, self._total_cost)
        _emit_counter(
            lines,
            "nexus_total_tool_calls",
            "Total tool calls executed",
            lbl,
            self._total_tool_calls,
        )
        _emit_counter(lines, "nexus_total_errors", "Total errors recorded", lbl, self._total_errors)
        _emit_counter(
            lines, "nexus_llm_call_count", "Total LLM calls made", lbl, self._llm_call_count
        )

        lines.append("# HELP nexus_active_agents Currently active agents")
        lines.append("# TYPE nexus_active_agents gauge")
        lines.append(f"nexus_active_agents{lbl} {self._active_agents}")

        _emit_histogram(
            lines, "nexus_llm_latency_ms", "LLM call latency in ms", lbl, self._llm_latency
        )
        _emit_histogram(
            lines, "nexus_tool_latency_ms", "Tool call latency in ms", lbl, self._tool_latency
        )
        return "\n".join(lines) + "\n"
