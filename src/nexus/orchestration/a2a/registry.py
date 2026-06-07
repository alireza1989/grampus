"""A2A AgentRegistry — tracks local runners and remote agent URLs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from a2a.types.a2a_pb2 import (
        AgentCard,
        AgentSkill,
    )

    _HAS_A2A = True
except ImportError:  # pragma: no cover
    _HAS_A2A = False

from pydantic import BaseModel, ConfigDict

from nexus.core.logging import get_logger
from nexus.core.types import AgentDefinition

if TYPE_CHECKING:
    from nexus.orchestration.runner import AgentRunner

_log = get_logger(__name__)


class AgentEntry(BaseModel):
    """A registered agent — local runner, Dapr sibling service, or remote A2A URL."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    card: Any  # AgentCard proto (typed Any to avoid import at class-definition time)
    runner: Any | None = None
    agent_def: Any | None = None  # AgentDefinition for local runners
    dapr_app_id: str | None = None  # path 2: sibling Nexus service via Dapr invocation
    dapr_method: str = "a2a"  # Dapr method path (default: the /a2a endpoint)
    remote_url: str | None = None  # path 3: external non-Nexus agent via A2A HTTP
    client: Any | None = None  # A2AAgentClient, lazily created from remote_url


class AgentRegistry:
    """Registry for local AgentRunner instances and remote A2A agent URLs.

    Replaces the D3 AgentRegistry but retains backward compatibility for
    HandoffExecutor consumers.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentEntry] = {}

    def register_local(
        self,
        name: str,
        runner: AgentRunner,
        description: str,
        skills: list[AgentSkill] | None = None,
        base_url: str = "http://localhost:8000",
        agent_def: AgentDefinition | None = None,
    ) -> None:
        """Register a local AgentRunner as an A2A agent.

        Args:
            name: Logical agent name.
            runner: AgentRunner instance.
            description: Human-readable description for the AgentCard.
            skills: Optional list of A2A AgentSkill protos to advertise.
            base_url: Base URL used to build the AgentCard interface URL.
            agent_def: Optional AgentDefinition; built from name/model defaults if omitted.
        """
        card = self._build_card(
            name=name,
            description=description,
            base_url=base_url,
            version="1.0.0",
            skills=skills,
        )
        effective_def = agent_def or AgentDefinition(name=name, model="claude-3-5-haiku-20241022")
        self._agents[name] = AgentEntry(
            name=name,
            card=card,
            runner=runner,
            agent_def=effective_def,
            remote_url=None,
        )

    def register_remote(
        self,
        name: str,
        url: str,
        api_key: str | None = None,
        _client: Any | None = None,
    ) -> None:
        """Register an external A2A agent by URL.

        The AgentCard is fetched lazily on first use. A pre-built client may
        be injected for testing via ``_client``.

        Args:
            name: Logical agent name.
            url: Base URL of the remote A2A agent.
            api_key: Optional Bearer token for auth.
            _client: Pre-built A2AAgentClient (for testing only).
        """
        # Build a placeholder card using the URL
        card = self._build_card(
            name=name,
            description=f"Remote agent at {url}",
            base_url=url,
        )

        client = _client
        if client is None and _HAS_A2A:
            from nexus.orchestration.a2a.client import A2AAgentClient

            client = A2AAgentClient(base_url=url, api_key=api_key)

        self._agents[name] = AgentEntry(
            name=name,
            card=card,
            runner=None,
            remote_url=url,
            client=client,
        )

    def register_dapr_service(
        self,
        name: str,
        dapr_app_id: str,
        description: str = "",
        dapr_method: str = "a2a",
        skills: list[AgentSkill] | None = None,
        dapr_http_port: int = 3500,
    ) -> None:
        """Register another Nexus service in the same cluster, reachable via Dapr service invocation.

        Use this instead of register_remote() when the target is a Nexus service
        running alongside this one. Traffic goes through the Dapr sidecar, gaining
        automatic mTLS, retries, and distributed tracing.

        Args:
            name: Logical agent name used in handoff tool names.
            dapr_app_id: Dapr application ID of the target service (matches --app-id at startup).
            description: Human-readable description for AgentCard generation.
            dapr_method: HTTP path on the target service to invoke (default: "a2a").
            skills: Optional skills list for the generated AgentCard.
            dapr_http_port: Dapr sidecar HTTP port (default: 3500).
        """
        invoke_url = (
            f"http://localhost:{dapr_http_port}/v1.0/invoke/{dapr_app_id}/method/{dapr_method}"
        )
        card = self._build_card(
            name=name,
            description=description or f"Dapr service: {dapr_app_id}",
            base_url=invoke_url,
            skills=skills,
        )
        self._agents[name] = AgentEntry(
            name=name,
            card=card,
            runner=None,
            dapr_app_id=dapr_app_id,
            dapr_method=dapr_method,
            remote_url=None,
            client=None,
        )

    def get(self, name: str) -> AgentEntry | None:
        """Return the AgentEntry for ``name``, or None if not registered."""
        return self._agents.get(name)

    def list_agents(self) -> list[AgentCard]:
        """Return all registered AgentCard protos."""
        return [e.card for e in self._agents.values()]

    def list_agent_names(self) -> list[str]:
        """Return all registered agent names (backward-compat helper)."""
        return list(self._agents.keys())

    def generate_server_card(
        self,
        name: str,
        description: str,
        base_url: str,
        version: str = "1.0.0",
        skills: list[AgentSkill] | None = None,
        api_key_scheme: bool = False,
    ) -> AgentCard:
        """Build the AgentCard advertising THIS Nexus server instance.

        Args:
            name: Agent name shown to callers.
            description: Human-readable description.
            base_url: Public base URL (used to construct the /a2a endpoint).
            version: Semantic version string.
            skills: Optional skills to include.
            api_key_scheme: When True, add an API-key security scheme.

        Returns:
            An AgentCard proto ready for serialization at /.well-known/agent-card.json.
        """
        return self._build_card(
            name=name,
            description=description,
            base_url=base_url,
            version=version,
            skills=skills,
            api_key_scheme=api_key_scheme,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _build_card(
        name: str,
        description: str,
        base_url: str,
        version: str = "1.0.0",
        skills: list[AgentSkill] | None = None,
        api_key_scheme: bool = False,
    ) -> AgentCard:
        card = AgentCard()
        card.name = name
        card.description = description
        card.version = version

        iface = card.supported_interfaces.add()
        iface.url = f"{base_url.rstrip('/')}/a2a"
        iface.protocol_binding = "JSONRPC"
        iface.protocol_version = "1.0"

        card.capabilities.streaming = True
        card.capabilities.push_notifications = False

        card.default_input_modes.append("text")
        card.default_output_modes.append("text")

        for skill in skills or []:
            card.skills.append(skill)

        if api_key_scheme:
            scheme = card.security_schemes["nexus-api-key"]
            scheme.api_key_security_scheme.name = "Authorization"
            scheme.api_key_security_scheme.location = "header"
            req = card.security_requirements.add()
            req.schemes["nexus-api-key"].list.extend([])

        return card
