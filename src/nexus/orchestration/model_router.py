"""Model router — selects the cheapest capable model per step with fallback."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from nexus.core.errors import ModelError
from nexus.core.logging import get_logger
from nexus.core.models.base import ModelResponse
from nexus.core.types import Message

_log = get_logger(__name__)


class ModelTier(StrEnum):
    """Cost/capability tier for model selection."""

    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


class ModelSpec(BaseModel):
    """Descriptor for a registered model including cost and capability metadata."""

    model_id: str
    tier: ModelTier
    provider: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    context_window: int
    supports_tools: bool = True


class RoutingRule(BaseModel):
    """Maps a step name pattern to a preferred tier.

    Matching uses substring search (case-insensitive). First matching rule wins.
    """

    step_pattern: str
    preferred_tier: ModelTier
    require_tools: bool = False


class ModelRouter:
    """Routes execution steps to the cheapest capable model.

    Args:
        models: Registered model specs; must cover at least one tier in use.
        clients: Map of model_id -> ModelClient instance (duck-typed as Any).
        routing_rules: Ordered rules; first substring match wins.
            Falls back to BALANCED when no rule matches.
        fallback_chain: Ordered tiers to attempt on failure.
            Defaults to [BALANCED, FAST, POWERFUL].
    """

    _DEFAULT_FALLBACK: list[ModelTier] = [ModelTier.BALANCED, ModelTier.FAST, ModelTier.POWERFUL]

    def __init__(
        self,
        models: list[ModelSpec],
        clients: dict[str, Any],
        *,
        routing_rules: list[RoutingRule] | None = None,
        fallback_chain: list[ModelTier] | None = None,
    ) -> None:
        self._models = models
        self._clients = clients
        self._rules = routing_rules or []
        self._fallback_chain = fallback_chain or list(self._DEFAULT_FALLBACK)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, step_name: str, *, require_tools: bool = False) -> ModelSpec:
        """Return the cheapest ModelSpec for this step.

        Applies routing rules (first substring match), then picks the
        cheapest model in the resolved tier. If ``require_tools=True``,
        only models with ``supports_tools=True`` are considered.

        Raises:
            ModelError: code="NO_MODEL_AVAILABLE" when no model satisfies constraints.
        """
        tier = self._resolve_tier(step_name)
        return self._cheapest_in_tier(tier, require_tools=require_tools)

    def get_client(self, model_spec: ModelSpec) -> Any:
        """Return the ModelClient registered for *model_spec*.

        Raises:
            ModelError: code="NO_CLIENT_FOR_MODEL" when not registered.
        """
        client = self._clients.get(model_spec.model_id)
        if client is None:
            raise ModelError(
                f"No client registered for model '{model_spec.model_id}'",
                code="NO_CLIENT_FOR_MODEL",
                hint="Register the provider with ModelRouter or use 'anthropic' / 'openai' as the provider name.",
            )
        return client

    async def complete_with_fallback(
        self,
        step_name: str,
        messages: list[Message],
        *,
        require_tools: bool = False,
        **kwargs: Any,
    ) -> tuple[ModelResponse, ModelSpec]:
        """Attempt completion, falling back through the tier chain on ModelError.

        Returns:
            Tuple of (ModelResponse, ModelSpec) for the model that succeeded.

        Raises:
            ModelError: When all tiers in the fallback chain fail or have no models.
        """
        last_error: ModelError | None = None
        tried_any = False

        for tier in self._fallback_chain:
            spec = self._cheapest_in_tier_or_none(tier, require_tools=require_tools)
            if spec is None:
                continue
            tried_any = True
            try:
                client = self.get_client(spec)
                response: ModelResponse = await client.complete(
                    messages=messages, model=spec.model_id, **kwargs
                )
                _log.debug("model_router_success", model=spec.model_id, tier=tier, step=step_name)
                return response, spec
            except ModelError as exc:
                _log.warning(
                    "model_router_fallback", model=spec.model_id, tier=tier, error=str(exc)
                )
                last_error = exc

        if not tried_any:
            raise ModelError(
                f"No models registered for any tier in fallback chain for step '{step_name}'",
                code="NO_MODEL_AVAILABLE",
                hint="All models in the fallback chain failed. Check API keys and rate limits.",
            )
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_tier(self, step_name: str) -> ModelTier:
        """Return the tier from the first matching rule, or BALANCED."""
        lower = step_name.lower()
        for rule in self._rules:
            if rule.step_pattern.lower() in lower:
                return rule.preferred_tier
        return ModelTier.BALANCED

    def _cheapest_in_tier(self, tier: ModelTier, *, require_tools: bool) -> ModelSpec:
        """Return the cheapest model in *tier*, raising ModelError if none found."""
        spec = self._cheapest_in_tier_or_none(tier, require_tools=require_tools)
        if spec is None:
            raise ModelError(
                f"No model available in tier '{tier}' (require_tools={require_tools})",
                hint="Add a model spec for this tier in ModelRouter or choose a different ModelTier.",
                code="NO_MODEL_AVAILABLE",
            )
        return spec

    def _cheapest_in_tier_or_none(
        self, tier: ModelTier, *, require_tools: bool
    ) -> ModelSpec | None:
        """Return the cheapest model in *tier*, or None if none available."""
        candidates = [
            m for m in self._models if m.tier == tier and (not require_tools or m.supports_tools)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.input_cost_per_1k_tokens)
