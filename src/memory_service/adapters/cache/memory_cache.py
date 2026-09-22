"""In-process CacheProvider (tests, single-process dev). Supports TTL and a failure switch."""

from __future__ import annotations

import fnmatch
import time
from collections.abc import AsyncIterator, Mapping, Sequence

from memory_service.ports.cache import CacheUnavailable
from memory_service.ports.models import ProviderInfo


class MemoryCache:
    info = ProviderInfo(name="memory", license="Apache-2.0", origin="internal", locality="local")

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, float | None]] = {}
        self._lists: dict[str, tuple[list[bytes], float | None]] = {}
        self.available = True  # flip to False to simulate an outage
        self.ops = 0

    def _check(self) -> None:
        self.ops += 1
        if not self.available:
            raise CacheUnavailable("simulated cache outage")

    def _live(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and expires < time.monotonic():
            del self._data[key]
            return None
        return value

    async def get(self, key: str) -> bytes | None:
        self._check()
        return self._live(key)

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        self._check()
        self._data[key] = (value, time.monotonic() + ttl_seconds if ttl_seconds else None)

    async def set_if_absent(
        self, key: str, value: bytes, *, ttl_seconds: int | None = None
    ) -> bool:
        self._check()
        if self._live(key) is not None:
            return False
        await self.set(key, value, ttl_seconds=ttl_seconds)
        return True

    async def delete(self, *keys: str) -> int:
        self._check()
        n = 0
        for k in keys:
            if k in self._data or k in self._lists:
                self._data.pop(k, None)
                self._lists.pop(k, None)
                n += 1
        return n

    async def incr(self, key: str, *, amount: int = 1) -> int:
        self._check()
        current = int((self._live(key) or b"0").decode())
        current += amount
        expires = self._data.get(key, (b"", None))[1]
        self._data[key] = (str(current).encode(), expires)
        return current

    async def incr_window(self, key: str, *, ttl_seconds: int) -> int:
        self._check()
        current = int((self._live(key) or b"0").decode()) + 1
        self._data[key] = (str(current).encode(), time.monotonic() + ttl_seconds)
        return current

    async def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        self._check()
        return [self._live(k) for k in keys]

    async def mset(self, items: Mapping[str, bytes], *, ttl_seconds: int | None = None) -> None:
        self._check()
        for k, v in items.items():
            await self.set(k, v, ttl_seconds=ttl_seconds)

    async def list_push(
        self, key: str, *values: bytes, max_len: int | None = None, ttl_seconds: int | None = None
    ) -> int:
        self._check()
        items, _ = self._lists.get(key, ([], None))
        items = [*items, *values]
        if max_len:
            items = items[-max_len:]
        self._lists[key] = (items, time.monotonic() + ttl_seconds if ttl_seconds else None)
        return len(items)

    async def list_range(self, key: str, start: int = 0, stop: int = -1) -> list[bytes]:
        self._check()
        entry = self._lists.get(key)
        if entry is None:
            return []
        items, expires = entry
        if expires is not None and expires < time.monotonic():
            del self._lists[key]
            return []
        stop_idx = len(items) if stop == -1 else stop + 1
        return items[start:stop_idx]

    async def scan(self, pattern: str) -> AsyncIterator[str]:
        self._check()
        for key in list(self._data) + list(self._lists):
            if fnmatch.fnmatch(key, pattern):
                yield key

    async def ping(self) -> bool:
        return self.available

    async def close(self) -> None:
        return None
