"""Hot thread cache and working memory in Dragonfly. Never the source of truth.

- ``hot:thread:{tenant}:{thread}:{revision}``   bounded list of recent visible messages
- ``wm:{tenant}:{scope}:{key}``                  ephemeral working-memory values (TTL)
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from memory_service.domain.conversation import Message
from memory_service.observability.logging import get_logger
from memory_service.ports.cache import CacheProvider, CacheUnavailable

log = get_logger(__name__)


def _dump(message: Message) -> bytes:
    return message.model_dump_json(exclude={"attachments", "system_metadata"}).encode()


class HotThreadCache:
    def __init__(
        self, cache: CacheProvider | None, *, max_messages: int = 200, ttl_seconds: int = 6 * 3600
    ) -> None:
        self.cache = cache
        self.max_messages = max_messages
        self.ttl = ttl_seconds

    @staticmethod
    def key(tenant_id: str, thread_id: str) -> str:
        return f"hot:thread:{tenant_id}:{thread_id}"

    async def append(self, message: Message) -> None:
        if self.cache is None or message.kind.value != "VISIBLE":
            return
        try:
            await self.cache.list_push(
                self.key(message.tenant_id, message.thread_id),
                _dump(message),
                max_len=self.max_messages,
                ttl_seconds=self.ttl,
            )
        except CacheUnavailable:
            log.debug("hot_thread.append_skipped")

    async def recent(self, tenant_id: str, thread_id: str, *, limit: int) -> list[Message] | None:
        """Return the newest ``limit`` visible messages or None on a miss/outage."""
        if self.cache is None:
            return None
        try:
            raw = await self.cache.list_range(self.key(tenant_id, thread_id), -limit, -1)
        except CacheUnavailable:
            return None
        if not raw:
            return None
        messages = [Message.model_validate_json(item) for item in raw]
        # a gap (message purged from the bounded list) means the cache cannot answer fully
        seqs = [m.sequence for m in messages]
        if seqs != list(range(seqs[0], seqs[0] + len(seqs))):
            return None
        return messages

    async def invalidate(self, tenant_id: str, thread_id: str) -> None:
        if self.cache is None:
            return
        with contextlib.suppress(CacheUnavailable):
            await self.cache.delete(self.key(tenant_id, thread_id))


class WorkingMemory:
    """Ephemeral per-scope key/value state (Lifetime.EPHEMERAL)."""

    def __init__(self, cache: CacheProvider | None, *, ttl_seconds: int = 1800) -> None:
        self.cache = cache
        self.ttl = ttl_seconds

    @staticmethod
    def key(tenant_id: str, scope_key: str, name: str) -> str:
        return f"wm:{tenant_id}:{scope_key}:{name}"

    async def set(self, tenant_id: str, scope_key: str, name: str, value: Any) -> None:
        if self.cache is None:
            return
        with contextlib.suppress(CacheUnavailable):
            await self.cache.set(
                self.key(tenant_id, scope_key, name),
                json.dumps(value, default=str).encode(),
                ttl_seconds=self.ttl,
            )

    async def get(self, tenant_id: str, scope_key: str, name: str) -> Any | None:
        if self.cache is None:
            return None
        try:
            raw = await self.cache.get(self.key(tenant_id, scope_key, name))
        except CacheUnavailable:
            return None
        return json.loads(raw) if raw is not None else None
