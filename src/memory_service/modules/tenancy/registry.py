"""What every request needs to know about tenants, without a store round trip.

The rate-limit middleware runs before authentication and ``build_context`` runs on every
call; neither may pay a database read. So this process keeps three small maps, refreshed
from the store every ``refresh_every`` seconds by its own loop (the API has no worker), and
updated at once by the instance that made an administrative change:

- ``tenant_id -> rate_limit_per_minute`` for the tenants that override the service default;
- ``key_id -> tenant_id`` for the keys of those tenants, because a caller holding a key
  usually sends no ``X-Trellis-Tenant`` - the key names the tenant. The token's key id is
  readable without verifying it (``mk_<key_id>.<secret>``); a wrong guess only picks a
  quota, authentication still decides who the caller is;
- the suspended tenants, so a suspension bites for every credential kind (``jwt`` and
  ``trusted_dev`` callers have no verifier cache to invalidate);
- every live key id, so the verifier can tell a real key whose cache entry lapsed from a
  guess when it bounds store reads for unknown ids (``modules/auth/keys.py``).

Other instances see a change within one refresh interval; the one that made it, at once.
"""

from __future__ import annotations

import asyncio
import contextlib

from memory_service.domain.tenancy import Tenant, bare_credential, parse_token
from memory_service.observability.logging import get_logger
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)


class TenantRegistry:
    def __init__(self, uow_factory: UnitOfWorkFactory, *, refresh_every: float = 60.0) -> None:
        self.uow_factory = uow_factory
        self.refresh_every = refresh_every
        self._limits: dict[str, int] = {}
        self._key_tenants: dict[str, str] = {}
        self._suspended: set[str] = set()
        self._live_keys: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        #: bumped by every local change; a refresh that started before one is discarded
        self._generation = 0

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
        """Whether the id is a live key as of the last refresh or an issue on this instance."""
        return key_id in self._live_keys

    # -- writes (the instance that changed something) -------------------------------
    async def observe(self, tenant: Tenant) -> None:
        """Apply a tenant's current record: quota, suspension, and - when the quota is set -
        the keys that must be metered under it."""
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
            return
        self._limits[tenant.tenant_id] = tenant.rate_limit_per_minute
        async with self.uow_factory() as uow:
            self._key_tenants.update(await uow.api_keys.key_tenants([tenant.tenant_id]))

    def observe_key(self, key_id: str, tenant_id: str) -> None:
        """A key issued on this instance is known, and counts toward its tenant's quota, at
        once."""
        self._generation += 1
        self._live_keys.add(key_id)
        if tenant_id in self._limits:
            self._key_tenants[key_id] = tenant_id

    def forget_key(self, key_id: str) -> None:
        self._generation += 1
        self._live_keys.discard(key_id)
        self._key_tenants.pop(key_id, None)

    async def refresh(self, *, attempts: int = 3) -> bool:
        """Reload from the store. A local change that lands during the read (an administrator
        acting on this instance) would be erased by a read that started before its commit,
        so such a read is retried; after ``attempts`` the latest read is applied anyway - by
        then it postdates the commit that change came from. Returns whether a read was
        applied without a retry."""
        clean = True
        limits: dict[str, int] = {}
        key_tenants: dict[str, str] = {}
        suspended: set[str] = set()
        live_keys: set[str] = set()
        for _ in range(max(1, attempts)):
            started = self._generation
            async with self.uow_factory() as uow:
                limits = await uow.tenants.rate_limits()
                suspended = set(await uow.tenants.suspended_tenants())
                key_tenants = await uow.api_keys.key_tenants(list(limits)) if limits else {}
                live_keys = set(await uow.api_keys.live_key_ids())
            if self._generation == started:
                break
            clean = False
        self._limits, self._key_tenants = limits, key_tenants
        self._suspended, self._live_keys = suspended, live_keys
        return clean

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="tenant-registry-refresh")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_every)
            try:
                await self.refresh()
            except Exception as exc:  # the store being away must not stop the loop
                log.warning("tenant_registry.refresh_failed", error=str(exc))
