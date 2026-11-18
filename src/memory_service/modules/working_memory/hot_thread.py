"""Hot thread cache in Dragonfly. Never the source of truth.

- ``hot:thread:{tenant}:{thread}``       bounded list of recent visible messages
- ``hot:thread:{tenant}:{thread}:rev``   ``{revision}:{sequence}``: the thread revision the
  list is current for, and the sequence of the message the list must end with

The list and its marker are written in separate commands, so a refill built from an earlier
read can wipe messages appended meanwhile and still find its marker overwritten by a later
append. A read therefore trusts the list only when the marker's revision is the thread's and
the list ends at the marker's message: a list cut short by such a race is a miss, and the
canonical read refills it.
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
        rev_key = self.rev_key(message.tenant_id, message.thread_id)
        try:
            if message.kind.value != "VISIBLE":
                # the list is unchanged, and which visible message ends it is not known here:
                # the next read misses and refills it
                await self.cache.delete(rev_key)
                return
            await self.cache.list_push(
                self.key(message.tenant_id, message.thread_id),
                _dump(message),
                max_len=self.max_messages,
                ttl_seconds=self.ttl,
            )
            await self.cache.set(rev_key, _marker(revision, message.sequence), ttl_seconds=self.ttl)
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
            if visible:
                await self.cache.set(
                    self.rev_key(tenant_id, thread_id),
                    _marker(revision, visible[-1].sequence),
                    ttl_seconds=self.ttl,
                )
            else:
                await self.cache.delete(self.rev_key(tenant_id, thread_id))

    async def recent(
        self, tenant_id: str, thread_id: str, *, limit: int, revision: int | None = None
    ) -> list[Message] | None:
        """Return the newest ``limit`` visible messages, or None on a miss, an outage, or
        when the cache is not current for ``revision`` (the thread's canonical revision)."""
        if self.cache is None:
            return None
        try:
            marker = _read_marker(await self.cache.get(self.rev_key(tenant_id, thread_id)))
            if marker is None or (revision is not None and marker[0] != revision):
                return None
            raw = await self.cache.list_range(self.key(tenant_id, thread_id), -limit, -1)
        except CacheUnavailable:
            return None
        if not raw:
            return None
        messages = [Message.model_validate_json(item) for item in raw]
        if messages[-1].sequence != marker[1]:
            return None  # cut short by a refill racing an append: not the thread's newest
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


def _marker(revision: int, sequence: int) -> bytes:
    """What the list is current for: the thread's revision and the message it ends with."""
    return f"{revision}:{sequence}".encode()


def _read_marker(stored: bytes | None) -> tuple[int, int] | None:
    """``(revision, sequence)`` of a stored marker; ``None`` for none, or one an older release
    wrote (a revision alone): a miss, which the canonical read rewrites."""
    if stored is None:
        return None
    revision, _, sequence = stored.decode().partition(":")
    try:
        return int(revision), int(sequence)
    except ValueError:
        return None
