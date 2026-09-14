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

    async def mget(self, keys: Sequence[str]) -> list[bytes | None]: ...

    async def mset(self, items: Mapping[str, bytes], *, ttl_seconds: int | None = None) -> None: ...

    async def list_push(
        self, key: str, *values: bytes, max_len: int | None = None, ttl_seconds: int | None = None
    ) -> int:
        """Append to a bounded list (recent-thread cache). Trims to ``max_len`` newest."""
        ...

    async def list_range(self, key: str, start: int = 0, stop: int = -1) -> list[bytes]: ...

    async def scan(self, pattern: str) -> AsyncIterator[str]: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...
