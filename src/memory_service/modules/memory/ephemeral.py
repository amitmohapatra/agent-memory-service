"""Working memory: EPHEMERAL candidates live only in the cache (Dragonfly TTL), per thread
or agent run. They are never persisted and never indexed; the ContextBuilder folds them into
the bundle for the same scope while they last."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime

from memory_service.domain.context import MemoryExecutionContext
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.intelligence import MemoryCandidate


class EphemeralMemory:
    def __init__(
        self, cache: CacheProvider | None, *, ttl_seconds: int = 3600, max_items: int = 50
    ):
        self.cache = cache
        self.ttl = ttl_seconds
        self.max_items = max_items

    @staticmethod
    def key(ctx: MemoryExecutionContext) -> str | None:
        anchor = ctx.agent_run_id or ctx.thread_id
        return f"wm:{ctx.tenant_id}:{anchor}" if anchor else None

    async def remember(self, ctx: MemoryExecutionContext, candidate: MemoryCandidate) -> None:
        key = self.key(ctx)
        if self.cache is None or key is None:
            return
        item = json.dumps(
            {
                "content": candidate.content,
                "memory_type": candidate.memory_type.value,
                "principal": ctx.principal_id,
                "at": datetime.now(UTC).isoformat(),
            }
        ).encode()
        with contextlib.suppress(CacheUnavailable):
            await self.cache.list_push(key, item, max_len=self.max_items, ttl_seconds=self.ttl)

    async def recall(self, ctx: MemoryExecutionContext) -> list[dict[str, str]]:
        key = self.key(ctx)
        if self.cache is None or key is None:
            return []
        try:
            raw = await self.cache.list_range(key)
        except CacheUnavailable:
            return []
        out = []
        for item in raw:
            with contextlib.suppress(ValueError):
                d = json.loads(item)
                # private to the principal that wrote it, or to the user the agent works for
                if d.get("principal") == ctx.principal_id or (
                    ctx.user_id and d.get("principal") == f"user:{ctx.user_id}"
                ):
                    out.append(d)
        return out
