"""Hot thread cache in Dragonfly. Never the source of truth.

- ``hot:thread:{tenant}:{thread}``       bounded list of recent visible messages
- ``hot:thread:{tenant}:{thread}:rev``   thread revision the list is current for
"""

from __future__ import annotations

import contextlib

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

    @classmethod
    def rev_key(cls, tenant_id: str, thread_id: str) -> str:
        return cls.key(tenant_id, thread_id) + ":rev"

    async def append(self, message: Message, *, revision: int = 0) -> None:
        """Push a committed message; ``revision`` is the thread revision after it. A push
        that fails (outage) leaves the stored revision behind, so the next read misses."""
        if self.cache is None:
            return
        try:
            if message.kind.value == "VISIBLE":
                await self.cache.list_push(
                    self.key(message.tenant_id, message.thread_id),
                    _dump(message),
                    max_len=self.max_messages,
                    ttl_seconds=self.ttl,
                )
            await self.cache.set(
                self.rev_key(message.tenant_id, message.thread_id),
                str(revision).encode(),
                ttl_seconds=self.ttl,
            )
        except CacheUnavailable:
            log.debug("hot_thread.append_skipped")

    async def refill(
        self, tenant_id: str, thread_id: str, messages: list[Message], *, revision: int
    ) -> None:
        """Rebuild the list from the canonical read (after a miss)."""
        if self.cache is None or not messages:
            return
        visible = [m for m in messages if m.kind.value == "VISIBLE"][-self.max_messages :]
        with contextlib.suppress(CacheUnavailable):
            await self.cache.delete(self.key(tenant_id, thread_id))
            for m in visible:
                await self.cache.list_push(
                    self.key(tenant_id, thread_id),
                    _dump(m),
                    max_len=self.max_messages,
                    ttl_seconds=self.ttl,
                )
            await self.cache.set(
                self.rev_key(tenant_id, thread_id), str(revision).encode(), ttl_seconds=self.ttl
            )

    async def recent(
        self, tenant_id: str, thread_id: str, *, limit: int, revision: int | None = None
    ) -> list[Message] | None:
        """Return the newest ``limit`` visible messages, or None on a miss, an outage, or
        when the cache is not current for ``revision`` (the thread's canonical revision)."""
        if self.cache is None:
            return None
        try:
            if revision is not None:
                stored = await self.cache.get(self.rev_key(tenant_id, thread_id))
                if stored is None or int(stored) != revision:
                    return None
            raw = await self.cache.list_range(self.key(tenant_id, thread_id), -limit, -1)
        except CacheUnavailable:
            return None
        if not raw:
            return None
        messages = [Message.model_validate_json(item) for item in raw]
        # a gap (a push that failed during an outage) means the cache cannot answer fully;
        # neither can a short list that does not start at the thread's first message
        seqs = [m.sequence for m in messages]
        if seqs != list(range(seqs[0], seqs[0] + len(seqs))):
            return None
        if len(messages) < limit and seqs[0] != 1:
            return None
        return messages

    async def invalidate(self, tenant_id: str, thread_id: str) -> None:
        if self.cache is None:
            return
        with contextlib.suppress(CacheUnavailable):
            await self.cache.delete(self.key(tenant_id, thread_id))
            await self.cache.delete(self.rev_key(tenant_id, thread_id))
