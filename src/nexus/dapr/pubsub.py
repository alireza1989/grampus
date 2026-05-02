"""Typed Dapr pub/sub with handler registry."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from nexus.core.logging import get_logger
from nexus.dapr.client import DaprGateway
from nexus.dapr.serialization import from_dapr_bytes, to_dapr_bytes

_log = get_logger(__name__)

_HandlerFn = Callable[[Any], Awaitable[None]]


class DaprPubSub:
    """Typed publish/subscribe backed by Dapr pub/sub building block.

    Handlers are registered per topic and invoked when ``handle_incoming``
    is called (e.g. from an HTTP subscription endpoint).
    """

    def __init__(self, gateway: DaprGateway, pubsub_name: str) -> None:
        self._gw = gateway
        self._pubsub = pubsub_name
        self._handlers: dict[str, list[tuple[type[BaseModel], _HandlerFn]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(
        self,
        topic: str,
        event: BaseModel,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Serialize *event* and publish it to *topic*."""
        data = to_dapr_bytes(event)
        kwargs: dict[str, Any] = {
            "pubsub_name": self._pubsub,
            "topic_name": topic,
            "data": data,
        }
        if metadata is not None:
            kwargs["publish_metadata"] = metadata
        await self._gw.publish_event(**kwargs)
        _log.debug("pubsub_published", pubsub=self._pubsub, topic=topic)

    # ------------------------------------------------------------------
    # Handler registry
    # ------------------------------------------------------------------

    def register_handler(
        self,
        topic: str,
        event_cls: type[BaseModel],
        handler: _HandlerFn,
    ) -> None:
        """Register *handler* to be called for events on *topic*."""
        self._handlers[topic].append((event_cls, handler))
        _log.debug("pubsub_handler_registered", topic=topic)

    def handler(
        self,
        topic: str,
        event_cls: type[BaseModel],
    ) -> Callable[[_HandlerFn], _HandlerFn]:
        """Decorator that registers the decorated coroutine as a topic handler."""

        def decorator(fn: _HandlerFn) -> _HandlerFn:
            self.register_handler(topic, event_cls, fn)
            return fn

        return decorator

    def has_handler(self, topic: str) -> bool:
        """Return True if at least one handler is registered for *topic*."""
        return bool(self._handlers.get(topic))

    # ------------------------------------------------------------------
    # Incoming dispatch
    # ------------------------------------------------------------------

    async def handle_incoming(self, topic: str, payload: bytes) -> None:
        """Deserialize *payload* and call all handlers registered for *topic*.

        If no handler is registered the call is a no-op.
        """
        topic_handlers = self._handlers.get(topic)
        if not topic_handlers:
            return

        coros = []
        for event_cls, fn in topic_handlers:
            event = from_dapr_bytes(payload, event_cls)
            coros.append(fn(event))

        await asyncio.gather(*coros)
        _log.debug("pubsub_handled", topic=topic, handler_count=len(coros))
