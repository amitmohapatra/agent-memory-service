"""Tool-output working memory (TOOL_MEMORY.md §30.2).

A lookup answers "has this exact call already been made, in a scope I am allowed to reuse,
recently enough to still be true?". It is deliberately narrow: only a tool whose registered
policy says *deterministic, cacheable and free of write side effects* may ever be replayed,
because serving a cached answer suppresses a call the agent would otherwise make. Every hit
says it is a hit and how old it is, so an agent can decide to call anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.tools import CacheScope, ToolDescriptor, stable_hash
from memory_service.observability.logging import get_logger
from memory_service.ports.cache import CacheProvider

log = get_logger(__name__)


@dataclass(frozen=True)
class CachedOutput:
    output_summary: str
    output_digest: str | None
    output_blob_ref: str | None
    output_fields: dict[str, Any]
    invocation_id: str
    stored_at: datetime

    @property
    def age_seconds(self) -> float:
        return max(0.0, (datetime.now(UTC) - self.stored_at).total_seconds())


def scope_value(scope: CacheScope, ctx: MemoryExecutionContext) -> str | None:
    """The identifier a cached entry is confined to. ``None`` means the scope is not available
    in this context, and the entry is then neither written nor read (never silently widened)."""
    if scope == "run":
        return ctx.agent_run_id
    if scope == "thread":
        return ctx.thread_id
    if scope == "user":
        return ctx.user_id
    return ctx.tenant_id


class ToolOutputCache:
    def __init__(self, cache: CacheProvider | None) -> None:
        self.cache = cache

    def _key(self, tool: ToolDescriptor, args_hash: str, ctx: MemoryExecutionContext) -> str | None:
        anchor = scope_value(tool.policy.cache_scope, ctx)
        if anchor is None:
            return None
        return (
            f"toolout:{ctx.tenant_id}:{tool.policy.cache_scope}:{anchor}"
            f":{tool.name}:v{tool.version}:{args_hash}"
        )

    async def get(
        self, tool: ToolDescriptor, args_hash: str, ctx: MemoryExecutionContext
    ) -> CachedOutput | None:
        if self.cache is None or not tool.policy.replayable:
            return None
        key = self._key(tool, args_hash, ctx)
        if key is None:
            return None
        try:
            raw = await self.cache.get(key)
        except Exception as exc:
            log.warning("tools.cache.unavailable", error=type(exc).__name__)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return CachedOutput(
                output_summary=data["output_summary"],
                output_digest=data.get("output_digest"),
                output_blob_ref=data.get("output_blob_ref"),
                output_fields=data.get("output_fields") or {},
                invocation_id=data["invocation_id"],
                stored_at=datetime.fromisoformat(data["stored_at"]),
            )
        except (ValueError, KeyError):
            return None

    async def put(
        self,
        tool: ToolDescriptor,
        args_hash: str,
        ctx: MemoryExecutionContext,
        *,
        output_summary: str,
        output_digest: str | None,
        output_blob_ref: str | None,
        output_fields: dict[str, Any],
        invocation_id: str,
    ) -> bool:
        if self.cache is None or not tool.policy.replayable or tool.policy.cache_ttl_seconds <= 0:
            return False
        key = self._key(tool, args_hash, ctx)
        if key is None:
            return False
        payload = json.dumps(
            {
                "output_summary": output_summary,
                "output_digest": output_digest,
                "output_blob_ref": output_blob_ref,
                "output_fields": output_fields,
                "invocation_id": invocation_id,
                "stored_at": datetime.now(UTC).isoformat(),
            },
            default=str,
        ).encode()
        try:
            await self.cache.set(key, payload, ttl_seconds=tool.policy.cache_ttl_seconds)
        except Exception as exc:
            log.warning("tools.cache.write_failed", error=type(exc).__name__)
            return False
        return True


def args_hash_for(tool: ToolDescriptor, args: dict[str, Any]) -> str:
    """Hash the *full* arguments, redacted ones included.

    Only the redacted copy is ever persisted, but the hash must still distinguish calls that
    differ solely in a redacted field: two callers with different credentials can legitimately
    get different results, and hashing the redacted form would collapse them onto one cache
    entry and serve one caller's output to the other. The hash is a one-way digest, so the
    secret is not recoverable from it.
    """
    del tool  # identity is already in the cache key; the hash covers arguments only
    return stable_hash(args)
