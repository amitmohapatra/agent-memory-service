"""Redis-protocol CacheProvider: Dragonfly (production), Valkey, Redis.

Every call is guarded: a backend failure raises ``CacheUnavailable`` quickly (short socket
timeouts) so callers degrade to the canonical store instead of hanging.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence

import redis.asyncio as redis_async
from redis.exceptions import RedisError

from memory_service.config.settings import CacheSettings
from memory_service.observability.metrics import cache_ops_total
from memory_service.ports.cache import CacheUnavailable
from memory_service.ports.models import ProviderInfo

_LICENSES = {"dragonfly": "BSL-1.1", "valkey": "BSD-3-Clause", "redis": "RSALv2/SSPL"}


class RedisCache:
    def __init__(self, settings: CacheSettings) -> None:
        self.settings = settings
        self.info = ProviderInfo(
            name=settings.provider,
            license=_LICENSES.get(settings.provider, "unknown"),
            origin="redis-protocol server",
            locality="local",
            data_residency="deployment",
        )
        self._client = redis_async.from_url(
            settings.url,
            socket_connect_timeout=settings.connect_timeout_seconds,
            socket_timeout=settings.socket_timeout_seconds,
            decode_responses=False,
            health_check_interval=30,
        )

    async def _guard(self, op: str, coro):  # type: ignore[no-untyped-def]
        try:
            result = await coro
        except (RedisError, OSError, TimeoutError) as exc:
            cache_ops_total.labels(op, "error").inc()
            raise CacheUnavailable(f"cache {op} failed: {type(exc).__name__}") from exc
        cache_ops_total.labels(op, "ok").inc()
        return result

    async def get(self, key: str) -> bytes | None:
        return await self._guard("get", self._client.get(key))

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        await self._guard("set", self._client.set(key, value, ex=ttl_seconds))

    async def set_if_absent(
        self, key: str, value: bytes, *, ttl_seconds: int | None = None
    ) -> bool:
        return bool(
            await self._guard("setnx", self._client.set(key, value, ex=ttl_seconds, nx=True))
        )

    async def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(await self._guard("delete", self._client.delete(*keys)))

    async def incr(self, key: str, *, amount: int = 1) -> int:
        return int(await self._guard("incr", self._client.incrby(key, amount)))

    async def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        if not keys:
            return []
        return list(await self._guard("mget", self._client.mget(list(keys))))

    async def mset(self, items: Mapping[str, bytes], *, ttl_seconds: int | None = None) -> None:
        if not items:
            return
        pipe = self._client.pipeline(transaction=False)
        for k, v in items.items():
            pipe.set(k, v, ex=ttl_seconds)
        await self._guard("mset", pipe.execute())

    async def list_push(
        self, key: str, *values: bytes, max_len: int | None = None, ttl_seconds: int | None = None
    ) -> int:
        pipe = self._client.pipeline(transaction=True)
        pipe.rpush(key, *values)
        if max_len:
            pipe.ltrim(key, -max_len, -1)
        if ttl_seconds:
            pipe.expire(key, ttl_seconds)
        results = await self._guard("list_push", pipe.execute())
        return int(results[0])

    async def list_range(self, key: str, start: int = 0, stop: int = -1) -> list[bytes]:
        return list(await self._guard("list_range", self._client.lrange(key, start, stop)))

    async def scan(self, pattern: str) -> AsyncIterator[str]:
        try:
            async for key in self._client.scan_iter(match=pattern, count=500):
                yield key.decode() if isinstance(key, bytes) else key
        except (RedisError, OSError) as exc:
            raise CacheUnavailable("cache scan failed") from exc

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except (RedisError, OSError, TimeoutError):
            return False

    async def close(self) -> None:
        await self._client.aclose()
