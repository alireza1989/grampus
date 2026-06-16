"""Pydantic ↔ Dapr bytes conversion and content hashing utilities."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ValidationError

from grampus.core.errors import StateSerializationError


def to_dapr_bytes(model: BaseModel) -> bytes:
    """Serialize a Pydantic model to UTF-8 JSON bytes for Dapr state storage."""
    return model.model_dump_json().encode("utf-8")


def from_dapr_bytes[T: BaseModel](data: bytes | None, cls: type[T]) -> T:
    """Deserialize Dapr state bytes into a Pydantic model instance.

    Raises:
        StateSerializationError: If data is empty, not valid UTF-8, not valid
            JSON, or does not match the expected model schema.
    """
    if not data:
        raise StateSerializationError(
            f"Cannot deserialize empty bytes into {cls.__name__}",
            code="STATE_SERIALIZATION_ERROR",
            details={"target_class": cls.__name__},
        )
    try:
        text = data.decode("utf-8")
    except (UnicodeDecodeError, AttributeError) as exc:
        raise StateSerializationError(
            f"State bytes are not valid UTF-8 for {cls.__name__}",
            code="STATE_SERIALIZATION_ERROR",
            details={"target_class": cls.__name__},
        ) from exc
    try:
        return cls.model_validate_json(text)
    except (ValidationError, ValueError) as exc:
        raise StateSerializationError(
            f"Failed to deserialize state into {cls.__name__}: {exc}",
            code="STATE_SERIALIZATION_ERROR",
            details={"target_class": cls.__name__},
        ) from exc


def compute_content_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def empty_response(response: Any) -> bool:
    """Return True if a Dapr StateResponse contains no data."""
    data = getattr(response, "data", None)
    return not data
