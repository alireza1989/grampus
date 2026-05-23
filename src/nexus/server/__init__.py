"""Nexus REST API server package."""

from nexus.server.app import create_app
from nexus.server.models import (
    HealthResponse,
    MemoryRecallRequest,
    MemoryRecallResponse,
    RunRequest,
    RunResponse,
    StreamChunkResponse,
)
from nexus.server.openai_compat import (
    OAIChatChunk,
    OAIChatRequest,
    OAIChatResponse,
    OAIMessage,
    OAIModelList,
    OAIModelObject,
    OAIUsage,
    create_openai_router,
)

__all__ = [
    "create_app",
    "HealthResponse",
    "MemoryRecallRequest",
    "MemoryRecallResponse",
    "RunRequest",
    "RunResponse",
    "StreamChunkResponse",
    "OAIChatChunk",
    "OAIChatRequest",
    "OAIChatResponse",
    "OAIMessage",
    "OAIModelList",
    "OAIModelObject",
    "OAIUsage",
    "create_openai_router",
]
