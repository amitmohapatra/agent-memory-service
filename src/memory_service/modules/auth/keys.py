"""API keys checked against their stored hashes, with a short shared cache.

The hot path pays one constant-time comparison; the store is read once per key per
``ttl_seconds``. Revocation and a tenant's suspension do not delete the entry - they replace
it with a tombstone that says "read the store", so a reader that fetched the row a moment
before the change cannot write the stale record back over the deletion (a cache-aside race
that let a revoked key live on for a TTL). Entries are written with set-if-absent for the
same reason. Unknown ids are cached as missing for a few seconds so a flood of well-formed
garbage tokens costs the store nothing; issuing a key clears that marker.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, ValidationError

from memory_service.domain.errors import AuthenticationFailed, AuthorizationFailed
from memory_service.domain.tenancy import (
    ApiKey,
    KeyRole,
    TenantStatus,
    parse_token,
    secret_matches,
)
from memory_service.observability.logging import get_logger
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

#: Cache values that are not records: "re-read the store" and "there is no such key".
TOMBSTONE = b"\x00tombstone"
MISSING = b"\x00missing"
MISSING_TTL_SECONDS = 5
#: Bounds the in-process memo of recent touches; a process serving more distinct keys than
#: this within one window simply touches a little more often.
MAX_TOUCH_MEMO = 10_000
#: Store reads for ids the store does not know, per process per minute. A flood of distinct
#: well-formed garbage tokens is the one shape the per-credential limiter cannot see (each
#: is its own bucket); past this budget an id nothing on this instance recognises - not the
#: registry's list of live keys, not a key this process has served - is refused without a
#: read. A key issued on another instance during such a flood may be refused until the
#: registry's next refresh; every key this instance knows keeps verifying.
UNKNOWN_IDS_PER_MINUTE = 600


@dataclass(frozen=True)
class VerifiedKey:
    key_id: str
    tenant_id: str
    role: KeyRole
    workspace_id: str | None
    may_act_as: tuple[str, ...] = ("*",)


class _Cached(BaseModel):
    """What the cache holds per key: the record and its tenant's status at load time."""

    model_config = ConfigDict(frozen=True)

    key: ApiKey
    tenant_status: TenantStatus


class ApiKeyVerifier:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        cache: CacheProvider | None,
        *,
        ttl_seconds: int = 60,
        touch_every_seconds: int = 60,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        known: Callable[[str], bool] | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.cache = cache
        self.ttl = ttl_seconds
        self.touch_every = timedelta(seconds=touch_every_seconds)
        self.clock = clock
        #: whether an id is a live key as far as this instance knows (the tenant registry);
        #: without it the unknown-id budget refuses nothing, it only caches misses
        self.known = known
        self._touched: dict[str, datetime] = {}
        #: ids the store did not know, with the instant that answer expires (a local copy of
        #: the shared MISSING marker, so a repeated guess costs no cache read either)
        self._missing: dict[str, datetime] = {}
        self._unknown_window: tuple[int, int] = (0, 0)  # (minute, store reads for unknown ids)

    @staticmethod
    def cache_key(key_id: str) -> str:
        return f"apikey:{key_id}"

    async def verify(self, token: str) -> VerifiedKey:
        parsed = parse_token(token)
        if parsed is None:
            raise AuthenticationFailed("Missing or invalid API key")
        key_id, secret = parsed
        cached = await self._load(key_id)
        now = self.clock()
        # One message for every credential failure: a caller must not learn whether an id
        # exists, whether it expired, or whether it was revoked.
        if (
            cached is None
            or not cached.key.usable_at(now)
            or not secret_matches(secret, cached.key.secret_hash)
        ):
            raise AuthenticationFailed("Missing or invalid API key")
        if cached.tenant_status != "active":
            # The credential is genuine; the tenant is not being served. That is a policy
            # answer, so it says so instead of pretending the key is wrong.
            raise AuthorizationFailed(
                "tenant is suspended", details={"tenant_id": cached.key.tenant_id}
            )
        record = cached.key
        last = self._touched.get(key_id) or record.last_used_at
        if last is None or now - last > self.touch_every:
            # Only a successful use is a use: a wrong secret against a real id must not move
            # last_used_at. Bounded to one write per key per window per process.
            await self._touch(key_id, now)
        return VerifiedKey(
            record.key_id,
            record.tenant_id,
            record.role,
            record.workspace_id,
            tuple(record.may_act_as),
        )

    async def invalidate(self, key_id: str, *, strict: bool = False) -> None:
        """Replace whatever the cache holds with a tombstone: the next reads go to the store
        and nothing stale can be written back until the tombstone expires.

        ``strict`` raises ``CacheUnavailable`` instead of swallowing it: a revocation whose
        tombstone could not be written must not be reported as done.
        """
        if self.cache is None:
            return
        # Twice a record's life: written before the commit that makes it true, it must still
        # stand when a reader that fetched the row just before the change tries to cache it.
        ttl = 2 * self.ttl
        if strict:
            await self.cache.set(self.cache_key(key_id), TOMBSTONE, ttl_seconds=ttl)
            return
        with contextlib.suppress(CacheUnavailable):
            await self.cache.set(self.cache_key(key_id), TOMBSTONE, ttl_seconds=ttl)

    async def forget_missing(self, key_id: str) -> None:
        """A key was just issued under this id: drop the "no such key" marker a guess may
        have left, without the minute-long tombstone a revocation needs (the record itself
        is written set-if-absent on first use)."""
        self._missing.pop(key_id, None)
        if self.cache is not None:
            with contextlib.suppress(CacheUnavailable):
                await self.cache.delete(self.cache_key(key_id))

    async def invalidate_tenant(self, tenant_id: str, *, strict: bool = False) -> None:
        """Every key of the tenant re-reads the store on its next use (status changes)."""
        async with self.uow_factory() as uow:
            keys = await uow.api_keys.list(tenant_id)
        for key in keys:
            await self.invalidate(key.key_id, strict=strict)

    async def _load(self, key_id: str) -> _Cached | None:
        now = self.clock()
        if self._missing.get(key_id, now) > now:
            return None
        raw = await self._cached_raw(key_id)
        if raw == MISSING:
            return None
        if raw is not None and raw != TOMBSTONE:
            try:
                return _Cached.model_validate_json(raw)
            except ValidationError:
                # an entry this build cannot read (another build's shape, corruption): a miss
                raw = TOMBSTONE
        if raw is None and self._unknown(key_id) and not self._unknown_budget_left(now):
            # nothing cached, nothing on this instance recognises the id, and the store
            # already answered "unknown" too often this minute
            log.info("api_key.unknown_id_budget_refused", key_id=key_id)
            return None
        return await self._read_store(key_id, now, behind_tombstone=raw == TOMBSTONE)

    def _unknown(self, key_id: str) -> bool:
        if self.known is None or key_id in self._touched:
            return False
        return not self.known(key_id)

    async def _cached_raw(self, key_id: str) -> bytes | None:
        if self.cache is None:
            return None
        with contextlib.suppress(CacheUnavailable):
            return await self.cache.get(self.cache_key(key_id))
        return None

    async def _read_store(
        self, key_id: str, now: datetime, *, behind_tombstone: bool
    ) -> _Cached | None:
        async with self.uow_factory() as uow:
            record = await uow.api_keys.get(key_id)
            tenant = await uow.tenants.get(record.tenant_id) if record is not None else None
        if record is None:
            self._note_unknown(key_id, now)
            await self._put(key_id, MISSING, MISSING_TTL_SECONDS, overwrite=behind_tombstone)
            return None
        cached = _Cached(key=record, tenant_status=tenant.status if tenant else "suspended")
        # A live record is cached set-if-absent, so a tombstone written meanwhile wins. A
        # revoked or expired key cannot come back, so that answer may replace the tombstone
        # and spare the store the repeats; a suspension can be lifted, so it may not.
        irreversible = not record.usable_at(now)
        if not behind_tombstone or irreversible:
            await self._put(
                key_id, cached.model_dump_json().encode(), self.ttl, overwrite=irreversible
            )
        return cached

    def _unknown_budget_left(self, now: datetime) -> bool:
        minute = int(now.timestamp() // 60)
        window_minute, count = self._unknown_window
        return window_minute != minute or count < UNKNOWN_IDS_PER_MINUTE

    def _note_unknown(self, key_id: str, now: datetime) -> None:
        minute = int(now.timestamp() // 60)
        window_minute, count = self._unknown_window
        self._unknown_window = (minute, count + 1 if window_minute == minute else 1)
        if len(self._missing) >= MAX_TOUCH_MEMO:
            self._missing.clear()
        self._missing[key_id] = now + timedelta(seconds=MISSING_TTL_SECONDS)

    async def _put(self, key_id: str, value: bytes, ttl: int, *, overwrite: bool = False) -> None:
        """Set-if-absent by default: never overwrites a tombstone (or a fresher entry) written
        meanwhile. ``overwrite`` is for answers that cannot go stale the wrong way."""
        if self.cache is None:
            return
        with contextlib.suppress(CacheUnavailable):
            if overwrite:
                await self.cache.set(self.cache_key(key_id), value, ttl_seconds=ttl)
            else:
                await self.cache.set_if_absent(self.cache_key(key_id), value, ttl_seconds=ttl)

    async def _touch(self, key_id: str, now: datetime) -> None:
        async with self.uow_factory() as uow:
            await uow.api_keys.touch(key_id, at=now)
            await uow.commit()
        if len(self._touched) >= MAX_TOUCH_MEMO:
            self._touched.clear()
        self._touched[key_id] = now
