"""Bounds for the free-form JSON fields of the public API.

Every request field that carries caller-defined JSON is bounded here, so the worst-case
cost of a request is a property of the contract rather than of the caller's goodwill.
The bounds are checked at the API edge only; the domain models keep accepting whatever the
pipeline itself produces.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any

from pydantic import AfterValidator, Field

#: ``custom_metadata`` everywhere it appears: scope, observation, thread, message, file.
METADATA_MAX_KEYS = 32
METADATA_MAX_BYTES = 8 * 1024
METADATA_MAX_DEPTH = 2

#: Tool arguments and schemas; the output has its own, larger, limit because the service
#: already spills large outputs to the blob store.
TOOL_JSON_MAX_BYTES = 64 * 1024
TOOL_OUTPUT_MAX_BYTES = 256 * 1024


def serialised_size(value: Any) -> int:
    """Bytes of the compact JSON form; the same measure the storage layer pays for."""
    return len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"))


def container_depth(value: Any) -> int:
    """Nesting depth of dicts and lists: a flat object is 1, an object of objects is 2."""
    if isinstance(value, dict):
        return 1 + max((container_depth(v) for v in value.values()), default=0)
    if isinstance(value, list | tuple):
        return 1 + max((container_depth(v) for v in value), default=0)
    return 0


def _bounded_metadata(value: dict[str, Any]) -> dict[str, Any]:
    if len(value) > METADATA_MAX_KEYS:
        raise ValueError(f"custom_metadata may carry at most {METADATA_MAX_KEYS} keys")
    depth = container_depth(value)
    if depth > METADATA_MAX_DEPTH:
        raise ValueError(
            f"custom_metadata may nest at most {METADATA_MAX_DEPTH} levels deep (got {depth})"
        )
    size = serialised_size(value)
    if size > METADATA_MAX_BYTES:
        raise ValueError(
            f"custom_metadata may serialise to at most {METADATA_MAX_BYTES} bytes (got {size})"
        )
    return value


def bounded_json(limit_bytes: int) -> Callable[[Any], Any]:
    """Validator rejecting any JSON value whose compact serialisation exceeds ``limit_bytes``."""

    def check(value: Any) -> Any:
        size = serialised_size(value)
        if size > limit_bytes:
            raise ValueError(f"serialises to {size} bytes; the limit is {limit_bytes}")
        return value

    return check


CustomMetadata = Annotated[
    dict[str, Any],
    AfterValidator(_bounded_metadata),
    Field(
        description=(
            f"Caller-defined JSON object: at most {METADATA_MAX_KEYS} keys, "
            f"{METADATA_MAX_DEPTH} levels of nesting and {METADATA_MAX_BYTES} bytes serialised."
        )
    ),
]

ToolJson = Annotated[dict[str, Any], AfterValidator(bounded_json(TOOL_JSON_MAX_BYTES))]
ToolOutput = Annotated[Any, AfterValidator(bounded_json(TOOL_OUTPUT_MAX_BYTES))]
