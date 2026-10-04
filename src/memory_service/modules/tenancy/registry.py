"""What every request needs to know about tenants, without a store round trip.

The rate-limit middleware runs before authentication and ``build_context`` runs on every
call; neither may pay a database read. So this process keeps small maps in memory:

- ``tenant_id -> rate_limit_per_minute`` for the tenants that override the service default;
- ``key_id -> tenant_id`` for the keys of those tenants, because a caller holding a key
  usually sends no ``X-Trellis-Tenant`` - the key names the tenant. The token's key id is
  readable without verifying it (``mk_<key_id>.<secret>``); a wrong guess only picks a
  quota, authentication still decides who the caller is;
- the suspended tenants, so a suspension bites for every credential kind (``jwt`` and
  ``trusted_dev`` callers have no verifier cache to invalidate);
- the key ids this process has seen live (issued, announced or verified), so the verifier
  can tell a real key from a guess when it bounds store reads for unknown ids
  (``modules/auth/keys.py``). Bounded, and learned lazily, key by key.

How changes arrive (ADR 0031):

1. **At once, on the instance that made them** (``observe``, ``observe_key``,
   ``forget_key``).
2. **Promptly, everywhere else**: the same calls publish an event on the cache's pub/sub
   channel (``CHANNEL``), and every process's listener applies it - a tenant event by
   reading that one tenant's row, a key event as it stands. A suspension or a revocation
   reaches every API worker of every pod within a round trip instead of within a minute.
3. **Eventually, whatever happens to the cache**: the quota, key-tenant and suspension
   maps - a few rows each - are still reloaded every ``refresh_every`` seconds, and again
   whenever the listener has lost its subscription (it may have missed events while the
   cache was away). With the cache down the registry is as current as it always was.

The full list of live key ids is no longer reloaded: every process read every live key
every minute, a read that grows with the customer base and bought only the flood defence's
"known" answer. A key is known here once this process has issued, been told about or
verified it. A key that is in none of those, during a flood of garbage ids, waits for the
next minute's budget or is found through the shared key cache like any other.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections import OrderedDict
from typing import Any, Final

from memory_service.domain.tenancy import Tenant, bare_credential, parse_token
from memory_service.observability.logging import get_logger
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

#: the pub/sub channel administrative changes are announced on
CHANNEL: Final = "trellis:tenancy"
#: how many live key ids one process remembers (the oldest forgotten first)
MAX_KNOWN_KEYS: Final = 100_000
#: the listener's reconnect pause, doubling up to the cap while the cache is away
RECONNECT_SECONDS: Final = (1.0, 30.0)


class TenantRegistry:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        *,
        refresh_every: float = 60.0,
        cache: CacheProvider | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.refresh_every = refresh_every
        self.cache = cache
        self._limits: dict[str, int] = {}
        self._key_tenants: dict[str, str] = {}
        self._suspended: set[str] = set()
        self._live_keys: OrderedDict[str, None] = OrderedDict()
        self._tasks: list[asyncio.Task[None]] = []
        self._pending: set[asyncio.Task[None]] = set()
        #: bumped by every local change; a refresh that started before one is discarded
        self._generation = 0
        #: who this process is on the channel, so it skips its own announcements
        self._origin = f"{os.getpid()}-{id(self):x}"

    # -- reads (O(1), on the request path) ------------------------------------------
    def quota_for(self, tenant_id: str | None, credential: str | None) -> tuple[str, int | None]:
        """``(tenant, override)`` for a request. The tenant the caller's key names wins when
        the registry knows it - a header cannot pick another tenant's quota - else the
        header's tenant, else ``"-"`` and no override."""
        parsed = parse_token(bare_credential(credential)) if credential else None
        keyed = self._key_tenants.get(parsed[0]) if parsed else None
        tenant = keyed or tenant_id
        if not tenant:
            return "-", None
        return tenant, self._limits.get(tenant)

    def is_suspended(self, tenant_id: str) -> bool:
        return tenant_id in self._suspended

    def knows_key(self, key_id: str) -> bool:
        """Whether this process has seen the id live: issued here, announced, verified, or
        metered under a tenant's quota."""
        return key_id in self._live_keys or key_id in self._key_tenants

    def remember_key(self, key_id: str) -> None:
        """A key that just verified is live (the verifier tells the registry)."""
        self._live_keys[key_id] = None
        self._live_keys.move_to_end(key_id)
        while len(self._live_keys) > MAX_KNOWN_KEYS:
            self._live_keys.popitem(last=False)

    # -- writes (the instance that changed something) -------------------------------
    async def observe(self, tenant: Tenant, *, announce: bool = True) -> None:
        """Apply a tenant's current record - quota, suspension, and the keys to meter under
        a quota - then tell the other processes."""
        self._generation += 1
        if tenant.status == "suspended":
            self._suspended.add(tenant.tenant_id)
        else:
            self._suspended.discard(tenant.tenant_id)
        if tenant.rate_limit_per_minute is None:
            self._limits.pop(tenant.tenant_id, None)
            self._key_tenants = {
                k: t for k, t in self._key_tenants.items() if t != tenant.tenant_id
            }
        else:
            self._limits[tenant.tenant_id] = tenant.rate_limit_per_minute
            async with self.uow_factory() as uow:
                self._key_tenants.update(await uow.api_keys.key_tenants([tenant.tenant_id]))
        if announce:
            await self._announce({"type": "tenant", "tenant_id": tenant.tenant_id})

    def observe_key(self, key_id: str, tenant_id: str, *, announce: bool = True) -> None:
        """A key issued on this instance is known, and counts toward its tenant's quota, at
        once; the other processes hear of it on the channel."""
        self._generation += 1
        self.remember_key(key_id)
        if tenant_id in self._limits:
            self._key_tenants[key_id] = tenant_id
        if announce:
            self._announce_soon({"type": "key_issued", "key_id": key_id, "tenant_id": tenant_id})

    def forget_key(self, key_id: str, *, announce: bool = True) -> None:
        self._generation += 1
        self._live_keys.pop(key_id, None)
        self._key_tenants.pop(key_id, None)
        if announce:
            self._announce_soon({"type": "key_revoked", "key_id": key_id})

    async def refresh(self, *, attempts: int = 3) -> bool:
        """Reload the quotas, their keys and the suspensions from the store. A local change
        that lands during the read (an administrator acting on this instance) would be
        erased by a read that started before its commit, so such a read is retried; after
        ``attempts`` the latest read is applied anyway - by then it postdates the commit
        that change came from. Returns whether a read was applied without a retry."""
        clean = True
        limits: dict[str, int] = {}
        key_tenants: dict[str, str] = {}
        suspended: set[str] = set()
        for _ in range(max(1, attempts)):
            started = self._generation
            async with self.uow_factory() as uow:
                limits = await uow.tenants.rate_limits()
                suspended = set(await uow.tenants.suspended_tenants())
                key_tenants = await uow.api_keys.key_tenants(list(limits)) if limits else {}
            if self._generation == started:
                break
            clean = False
        self._limits, self._key_tenants, self._suspended = limits, key_tenants, suspended
        return clean

    # -- the channel -----------------------------------------------------------------
    async def _announce(self, event: dict[str, Any]) -> None:
        """Publish ``event``; with the cache away the others catch up on their refresh."""
        if self.cache is None:
            return
        body = json.dumps({**event, "origin": self._origin}).encode()
        try:
            await self.cache.publish(CHANNEL, body)
        except CacheUnavailable:
            log.info("tenant_registry.announce_failed", kind=event.get("type"))

    def _announce_soon(self, event: dict[str, Any]) -> None:
        if self.cache is None:
            return
        task = asyncio.get_running_loop().create_task(self._announce(event))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def apply(self, raw: bytes) -> None:
        """Apply one event another process announced."""
        try:
            event = json.loads(raw)
        except ValueError:
            return
        if not isinstance(event, dict) or event.get("origin") == self._origin:
            return
        kind, key_id = event.get("type"), event.get("key_id")
        if kind == "tenant" and isinstance(event.get("tenant_id"), str):
            async with self.uow_factory() as uow:
                tenant = await uow.tenants.get(event["tenant_id"])
            if tenant is not None:
                await self.observe(tenant, announce=False)
        elif kind == "key_issued" and isinstance(key_id, str):
            self.observe_key(key_id, str(event.get("tenant_id", "")), announce=False)
        elif kind == "key_revoked" and isinstance(key_id, str):
            self.forget_key(key_id, announce=False)

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        if self._tasks:
            return
        self._tasks.append(asyncio.create_task(self._run(), name="tenant-registry-refresh"))
        if self.cache is not None:
            self._tasks.append(asyncio.create_task(self._listen(), name="tenant-registry-events"))

    async def close(self) -> None:
        tasks = [*self._tasks, *self._pending]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_every)
            try:
                await self.refresh()
            except Exception as exc:  # the store being away must not stop the loop
                log.warning("tenant_registry.refresh_failed", error=str(exc))

    async def _listen(self) -> None:
        """Apply announced changes. When the subscription is lost, reload what may have been
        missed meanwhile, then subscribe again after a growing pause."""
        assert self.cache is not None
        pause, cap = RECONNECT_SECONDS
        while True:
            try:
                async for raw in self.cache.subscribe(CHANNEL):
                    pause = RECONNECT_SECONDS[0]
                    try:
                        await self.apply(raw)
                    except Exception as exc:  # one bad event must not end the listener
                        log.warning("tenant_registry.event_failed", error=str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.info("tenant_registry.listener_down", error=type(exc).__name__)
            with contextlib.suppress(Exception):
                await self.refresh()
            await asyncio.sleep(pause)
            pause = min(cap, pause * 2)
