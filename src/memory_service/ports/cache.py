"""CacheProvider port. Dragonfly in production, Valkey/Redis compatible, fakes in tests.

The cache is never the source of truth. Every method must tolerate the backend being
down: implementations raise :class:`CacheUnavailable` and callers degrade to the
canonical store.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol, runtime_checkable

from memory_service.domain.errors import DependencyUnavailable


class CacheUnavailable(DependencyUnavailable):
    """The cache backend is unreachable. Callers continue without cache."""


@runtime_checkable
class CacheProvider(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None: ...

    async def set_if_absent(
        self, key: str, value: bytes, *, ttl_seconds: int | None = None
    ) -> bool:
        """Atomic SETNX. Returns True when the key was created. O(1)."""
        ...

    async def delete(self, *keys: str) -> int: ...

    async def incr(self, key: str, *, amount: int = 1) -> int: ...

    async def incr_window(self, key: str, *, ttl_seconds: int) -> int:
        """Increment a counter and (re)set its expiry in one round trip.

        A fixed-window counter is INCR plus EXPIRE, and issued separately that is two waits
        on a remote cache for every request that passes through the rate limiter. The key
        already carries its window, so refreshing the expiry on each hit cannot extend a
        window; it only keeps the row alive until the window is over."""
        ...

    async def mget(self, keys: Sequence[str]) -> list[bytes | None]: ...

    async def mset(self, items: Mapping[str, bytes], *, ttl_seconds: int | None = None) -> None: ...

    async def list_push(
        self, key: str, *values: bytes, max_len: int | None = None, ttl_seconds: int | None = None
    ) -> int:
        """Append to a bounded list (recent-thread cache). Trims to ``max_len`` newest."""
        ...

    async def list_range(self, key: str, start: int = 0, stop: int = -1) -> list[bytes]: ...

    def scan(self, pattern: str) -> AsyncIterator[str]:
        """Keys matching ``pattern``; an async generator, so declared without ``async`` here
        (an ``async def`` in a Protocol would promise a coroutine *returning* the iterator,
        and every ``async for`` over it would fail to type-check)."""
        ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...
