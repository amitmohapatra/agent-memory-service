"""Idempotency for persistent write APIs.

Protocol
--------
1. ``begin`` looks the key up (cache O(1) fast path, then PostgreSQL). If a completed record
   exists with the same request hash, the stored response is replayed. If it exists with a
   different hash -> ``Conflict``. If it exists but is still in flight -> ``Conflict`` with
   ``retryable=True`` (client retries shortly).
2. The handler performs its Unit of Work; ``complete`` stores the response inside the same
   transaction when possible (the record is reserved in the UoW), then warms the cache.

Keys are scoped per tenant; the window is 24 hours by default.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict

from memory_service.domain.errors import Conflict
from memory_service.domain.ids import content_hash
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.repositories import IdempotencyRecord, IdempotencyRepository


class ReplayedResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: int
    body: dict[str, Any]


class IdempotencyService:
    def __init__(self, cache: CacheProvider | None, *, window_seconds: int = 24 * 3600) -> None:
        self.cache = cache
        self.window_seconds = window_seconds

    @staticmethod
    def request_hash(payload: Any) -> str:
        return content_hash(json.dumps(payload, sort_keys=True, default=str))

    @staticmethod
    def _cache_key(tenant_id: str, key: str) -> str:
        return f"idem:{tenant_id}:{key}"

    async def lookup_cached(
        self, tenant_id: str, key: str, request_hash: str
    ) -> ReplayedResponse | None:
        if self.cache is None:
            return None
        try:
            raw = await self.cache.get(self._cache_key(tenant_id, key))
        except CacheUnavailable:
            return None
        if raw is None:
            return None
        data = json.loads(raw)
        if data.get("request_hash") != request_hash:
            raise Conflict("Idempotency-Key reused with a different payload")
        return ReplayedResponse(status=int(data["status"]), body=data["body"])

    async def begin(
        self, repo: IdempotencyRepository, tenant_id: str, key: str, request_hash: str
    ) -> ReplayedResponse | None:
        """Reserve the key inside the caller's transaction, or replay/conflict.

        Returns a replay when a completed identical request exists; ``None`` when the caller
        should proceed (the key is now reserved in this transaction).
        """
        now = datetime.now(UTC)
        record = IdempotencyRecord(
            tenant_id=tenant_id,
            key=key,
            request_hash=request_hash,
            expires_at=now + timedelta(seconds=self.window_seconds),
        )
        if await repo.reserve(record):
            return None
        existing = await repo.get(tenant_id, key)
        if existing is None:  # expired + purged between reserve and get: retry once
            if await repo.reserve(record):
                return None
            raise Conflict("Idempotency-Key is being processed", details={"retry": True})
        if existing.expires_at < now:
            # expired: allow re-processing by rewriting the record
            await repo.complete(tenant_id, key, status=0, body={})
            raise Conflict("Idempotency-Key expired; retry with a new key")
        if existing.request_hash != request_hash:
            raise Conflict("Idempotency-Key reused with a different payload")
        if existing.response_status is None or existing.response_body is None:
            err = Conflict("Idempotency-Key is being processed", details={"retry": True})
            err.retryable = True
            raise err
        return ReplayedResponse(status=existing.response_status, body=existing.response_body)

    async def complete(
        self,
        repo: IdempotencyRepository,
        tenant_id: str,
        key: str,
        *,
        status: int,
        body: dict[str, Any],
    ) -> None:
        await repo.complete(tenant_id, key, status=status, body=body)

    async def warm_cache(
        self, tenant_id: str, key: str, request_hash: str, *, status: int, body: dict[str, Any]
    ) -> None:
        if self.cache is None:
            return
        try:
            await self.cache.set(
                self._cache_key(tenant_id, key),
                json.dumps(
                    {"request_hash": request_hash, "status": status, "body": body}, default=str
                ).encode(),
                ttl_seconds=self.window_seconds,
            )
        except CacheUnavailable:
            return
