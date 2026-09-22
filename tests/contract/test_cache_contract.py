"""One contract, every CachePort adapter.

There are 2,073,600 declared provider combinations, so "is this combination tested?" is the
wrong question — the answerable one is "does every adapter honour its port?". A shared suite
parametrised over the implementations is what makes the combinations safe to swap: if
``MemoryCache`` and ``RedisCache`` both pass this, ``cache.provider`` is a free choice.

An adapter whose backend is not reachable **skips with the reason**. It never passes quietly.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

from memory_service.config.settings import CacheSettings

pytestmark = pytest.mark.contract


#: dragonfly / valkey / redis are one adapter speaking one protocol; the dev stack runs
#: Dragonfly, so "redis-protocol" is the wire all three share.
ADAPTERS = ("memory", "redis-protocol")


def _build(name: str):
    if name == "memory":
        from memory_service.adapters.cache.memory_cache import MemoryCache

        return MemoryCache()
    from memory_service.adapters.cache.redis_cache import RedisCache

    return RedisCache(CacheSettings())


# ``loop_scope="function"`` is load-bearing: the project sets
# ``asyncio_default_fixture_loop_scope = "session"``, so by default this fixture would build
# the client on the session loop while the test awaits it on a function loop — a redis pool
# binds to the loop it was created on and fails with "attached to a different loop".
@pytest_asyncio.fixture(params=ADAPTERS, loop_scope="function")
async def cache(request: pytest.FixtureRequest):
    adapter = _build(request.param)
    if not await adapter.ping():
        pytest.skip(f"{request.param} cache backend not reachable — start the dev stack")
    try:
        yield adapter
    finally:
        await adapter.close()


def _key() -> str:
    return f"contract:{uuid.uuid4().hex}"


async def test_a_value_survives_a_round_trip(cache) -> None:
    key = _key()
    await cache.set(key, b"value")
    assert await cache.get(key) == b"value"
    assert await cache.delete(key) == 1
    assert await cache.get(key) is None


async def test_a_missing_key_is_none_not_an_error(cache) -> None:
    assert await cache.get(_key()) is None


async def test_set_if_absent_is_a_lock(cache) -> None:
    """Two callers race for the same key; exactly one may win, or it is not a lock."""
    key = _key()
    first, second = await asyncio.gather(
        cache.set_if_absent(key, b"a", ttl_seconds=30),
        cache.set_if_absent(key, b"b", ttl_seconds=30),
    )
    assert [first, second].count(True) == 1, "set_if_absent must admit exactly one writer"
    await cache.delete(key)


async def test_a_ttl_expires_the_value(cache) -> None:
    key = _key()
    await cache.set(key, b"brief", ttl_seconds=1)
    assert await cache.get(key) == b"brief"
    await asyncio.sleep(1.6)
    assert await cache.get(key) is None, "the TTL must be honoured, not just accepted"


async def test_incr_counts_from_zero_and_accumulates(cache) -> None:
    key = _key()
    assert await cache.incr(key) == 1
    assert await cache.incr(key, amount=4) == 5
    await cache.delete(key)


async def test_incr_window_counts_and_expires_in_one_round_trip(cache) -> None:
    """What the rate limiter runs on every request: the count comes back, and the key is not
    left behind forever when the window is over."""
    key = _key()
    assert await cache.incr_window(key, ttl_seconds=1) == 1
    assert await cache.incr_window(key, ttl_seconds=1) == 2
    await asyncio.sleep(1.6)
    assert await cache.get(key) is None, "a window counter must expire on its own"


async def test_mget_and_mset_preserve_order_and_gaps(cache) -> None:
    keys = [_key() for _ in range(3)]
    await cache.mset({keys[0]: b"0", keys[2]: b"2"})
    assert await cache.mget(keys) == [b"0", None, b"2"]
    await cache.delete(*keys)


async def test_a_list_keeps_insertion_order(cache) -> None:
    key = _key()
    await cache.list_push(key, b"1")
    await cache.list_push(key, b"2")
    assert await cache.list_range(key) == [b"1", b"2"]
    await cache.delete(key)


async def test_scan_finds_keys_by_prefix_and_excludes_others(cache) -> None:
    tag = uuid.uuid4().hex[:8]
    mine = [f"contract:{tag}:{i}" for i in range(3)]
    other = f"contract:{uuid.uuid4().hex[:8]}:x"
    await cache.mset(dict.fromkeys(mine, b"v") | {other: b"v"})
    seen = {k async for k in cache.scan(f"contract:{tag}:*")}
    assert seen == set(mine), "scan must match its pattern exactly"
    await cache.delete(*mine, other)


async def test_delete_reports_how_many_keys_it_removed(cache) -> None:
    keys = [_key(), _key()]
    await cache.mset(dict.fromkeys(keys, b"v"))
    assert await cache.delete(*keys, _key()) == 2
