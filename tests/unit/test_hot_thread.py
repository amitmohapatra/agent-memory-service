"""The hot thread cache: a thread's recent visible messages in the cache, trusted only when
it is provably the thread's newest — current for the thread's revision and ending at the
message its marker names — so a refill racing an append never serves a list cut short."""

from __future__ import annotations

from memory_service.adapters.cache.memory_cache import MemoryCache
from memory_service.domain.conversation import Message, MessageKind, MessageRole
from memory_service.modules.working_memory.hot_thread import HotThreadCache

T, THREAD = "acme", "thread_1"


def said(sequence: int, kind: MessageKind = MessageKind.VISIBLE) -> Message:
    return Message(
        thread_id=THREAD,
        session_id="s",
        turn_id="t",
        tenant_id=T,
        role=MessageRole.USER,
        kind=kind,
        sequence=sequence,
        content=f"m{sequence}",
        content_hash="x",
        author_principal="user:u",
    )


async def test_appended_messages_are_served_while_current() -> None:
    hot = HotThreadCache(MemoryCache())
    for n in (1, 2, 3):
        await hot.append(said(n), revision=n)
    served = await hot.recent(T, THREAD, limit=20, revision=3)
    assert served is not None and [m.sequence for m in served] == [1, 2, 3]
    assert await hot.recent(T, THREAD, limit=20, revision=4) is None  # the thread moved on


async def test_a_refill_racing_appends_never_serves_a_list_cut_short() -> None:
    """A refill built from an earlier read (only m1) deletes the list after m2..m4 were
    pushed, pushes m1 back, and its marker is overwritten by m4's append: the list is [m1]
    with the thread's current revision. It is a miss, and the canonical read refills it."""
    cache = MemoryCache()
    hot = HotThreadCache(cache)
    for n in (2, 3, 4):
        await hot.append(said(n), revision=n)
    await cache.delete(hot.key(T, THREAD))  # the slow refill's delete
    await cache.list_push(hot.key(T, THREAD), said(1).model_dump_json().encode())  # its push
    await cache.set(hot.rev_key(T, THREAD), b"4:4")  # m4's marker, written after it
    assert await hot.recent(T, THREAD, limit=20, revision=4) is None
    await hot.refill(T, THREAD, [said(n) for n in (1, 2, 3, 4)], revision=4)
    served = await hot.recent(T, THREAD, limit=20, revision=4)
    assert served is not None and [m.sequence for m in served] == [1, 2, 3, 4]


async def test_an_internal_message_or_an_old_marker_is_a_miss() -> None:
    cache = MemoryCache()
    hot = HotThreadCache(cache)
    await hot.append(said(1), revision=1)
    await hot.append(said(2, MessageKind.INTERNAL), revision=2)  # which visible ends it: unknown
    assert await hot.recent(T, THREAD, limit=20, revision=2) is None
    await cache.set(hot.rev_key(T, THREAD), b"2")  # the marker an older release wrote
    assert await hot.recent(T, THREAD, limit=20, revision=2) is None
    await hot.refill(T, THREAD, [said(2, MessageKind.INTERNAL)], revision=3)  # nothing visible
    assert await cache.get(hot.rev_key(T, THREAD)) is None
