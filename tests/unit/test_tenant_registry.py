"""The in-process tenant registry the limiter and the context builder read: quotas by tenant
or by the key a caller presents, suspensions, and how administrative changes reach it."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from memory_service.api.middleware import RateLimitMiddleware
from memory_service.domain.tenancy import Tenant, mint_token
from memory_service.modules.tenancy.registry import TenantRegistry

pytestmark = pytest.mark.unit


class _Tenants:
    def __init__(self, tenants: list[Tenant]) -> None:
        self.rows = {t.tenant_id: t for t in tenants}

    async def rate_limits(self) -> dict[str, int]:
        return {
            t.tenant_id: t.rate_limit_per_minute
            for t in self.rows.values()
            if t.rate_limit_per_minute is not None
        }

    async def suspended_tenants(self) -> list[str]:
        return [t.tenant_id for t in self.rows.values() if t.status == "suspended"]


class _Keys:
    def __init__(self, key_tenants: dict[str, str]) -> None:
        self.rows = key_tenants

    async def key_tenants(self, tenant_ids):  # type: ignore[no-untyped-def]
        return {k: t for k, t in self.rows.items() if t in set(tenant_ids)}

    async def live_key_ids(self) -> list[str]:
        return list(self.rows)


class _Uow:
    def __init__(self, tenants: _Tenants, keys: _Keys) -> None:
        self.tenants, self.api_keys = tenants, keys


def _registry(tenants: list[Tenant], key_tenants: dict[str, str] | None = None) -> TenantRegistry:
    uow = _Uow(_Tenants(tenants), _Keys(key_tenants or {}))

    @asynccontextmanager
    async def factory():
        yield uow

    return TenantRegistry(factory)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _frozen_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    """The limiter buckets by wall-clock minute; a test must not straddle two."""
    monkeypatch.setattr("memory_service.api.middleware.time.time", lambda: 1_800_000_000.0)


# -- the registry itself --------------------------------------------------------------


async def test_refresh_loads_quotas_their_keys_and_suspensions() -> None:
    registry = _registry(
        [
            Tenant(tenant_id="acme", name="Acme", rate_limit_per_minute=5),
            Tenant(tenant_id="globex", name="Globex", status="suspended"),
            Tenant(tenant_id="initech", name="Initech"),
        ],
        {"k1": "acme", "k2": "globex"},
    )
    assert registry.quota_for("acme", None) == ("acme", None), "empty until refreshed"
    await registry.refresh()
    assert registry.quota_for("acme", None) == ("acme", 5)
    assert registry.quota_for(None, f"mk_{'k1':<16}.s".replace(" ", "x")) == ("-", None), (
        "an id the store does not know is nobody"
    )
    assert registry.is_suspended("globex") and not registry.is_suspended("acme")
    assert registry.quota_for("initech", None) == ("initech", None)
    assert registry.knows_key("k1") and registry.knows_key("k2") and not registry.knows_key("k9")


async def test_observe_applies_a_tenant_s_record_and_loads_its_keys_at_once() -> None:
    key_id, token, _ = mint_token()
    registry = _registry([], {key_id: "acme"})
    await registry.observe(Tenant(tenant_id="acme", name="Acme", rate_limit_per_minute=3))
    assert registry.quota_for(None, token) == ("acme", 3), "the key is metered under its tenant"
    assert registry.quota_for(None, f"Bearer {token}") == ("acme", 3), "whichever carrier"
    await registry.observe(Tenant(tenant_id="acme", name="Acme"))
    assert registry.quota_for(None, token) == ("-", None), "clearing the quota forgets the keys"
    await registry.observe(Tenant(tenant_id="acme", name="Acme", status="suspended"))
    assert registry.is_suspended("acme")
    await registry.observe(Tenant(tenant_id="acme", name="Acme", status="active"))
    assert not registry.is_suspended("acme")


async def test_keys_issued_and_revoked_on_this_instance_are_known_at_once() -> None:
    registry = _registry([])
    await registry.observe(Tenant(tenant_id="acme", name="Acme", rate_limit_per_minute=3))
    key_id, token, _ = mint_token()
    registry.observe_key(key_id, "acme")
    registry.observe_key("other", "globex")  # globex has no override: no quota to remember
    assert registry.knows_key(key_id) and registry.knows_key("other"), "but both are live keys"
    assert registry.quota_for(None, token) == ("acme", 3)
    assert registry.quota_for("globex", token) == ("acme", 3), "a header cannot pick another quota"
    registry.forget_key(key_id)
    assert registry.quota_for(None, token) == ("-", None)
    assert registry.quota_for(None, "mk_k1.s") == ("-", None), "a malformed token is nobody"


async def test_a_refresh_that_raced_a_local_change_is_discarded() -> None:
    """The administrator's change must not be erased by a read that started before it."""
    key_id, token, _ = mint_token()
    registry = _registry(
        [Tenant(tenant_id="acme", name="Acme", rate_limit_per_minute=5)], {key_id: "acme"}
    )
    original = registry.uow_factory
    fired = False

    @asynccontextmanager
    async def racing():
        nonlocal fired
        async with original() as uow:
            if not fired:
                # the store is being read; meanwhile an administrator suspends the tenant:
                # the row commits, then observe() applies it locally
                fired = True
                suspended = Tenant(
                    tenant_id="acme", name="Acme", status="suspended", rate_limit_per_minute=5
                )
                uow.tenants.rows["acme"] = suspended
                await registry.observe(suspended)
            yield uow

    registry.uow_factory = racing  # type: ignore[assignment]
    assert await registry.refresh() is False, "the first read raced and was retried"
    assert registry.is_suspended("acme"), "the suspension survived the stale read"
    assert registry.quota_for(None, token) == ("acme", 5)
    registry.uow_factory = original  # type: ignore[assignment]
    assert await registry.refresh() is True


async def test_a_busy_instance_still_applies_its_last_read() -> None:
    """Retries are bounded: past them the latest read is applied, since by then it
    postdates the commit every local change came from."""
    registry = _registry([Tenant(tenant_id="acme", name="Acme", rate_limit_per_minute=5)])
    await registry.refresh()  # acme's quota is known, so a key observed for it is a change
    original = registry.uow_factory

    @asynccontextmanager
    async def always_racing():
        async with original() as uow:
            registry.observe_key("k", "acme")  # a local change on every read
            yield uow

    registry.uow_factory = always_racing  # type: ignore[assignment]
    assert await registry.refresh(attempts=2) is False
    assert registry.quota_for("acme", None) == ("acme", 5), "the read was applied anyway"


# -- the middleware reading it ----------------------------------------------------------


class _Cache:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def incr_window(self, key: str, *, ttl_seconds: int) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


class _Container:
    def __init__(self, registry: TenantRegistry) -> None:
        self.cache = _Cache()
        self.services = {"tenant_registry": registry}


class _State:
    def __init__(self, container: _Container) -> None:
        self.container = container


class _App:
    def __init__(self, container: _Container) -> None:
        self.state = _State(container)


async def _status(
    middleware: RateLimitMiddleware, app: _App, tenant: str | None, key: str = "k"
) -> tuple[int, str | None]:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request"}

    async def send(message: dict) -> None:
        sent.append(message)

    headers = [(b"x-api-key", key.encode())]
    if tenant is not None:
        headers.append((b"x-trellis-tenant", tenant.encode()))
    scope = {"type": "http", "path": "/v1/recall", "headers": headers, "app": app, "state": {}}
    await middleware(scope, receive, send)
    start = next(m for m in sent if m.get("type") == "http.response.start")
    limit = dict(start.get("headers", [])).get(b"x-ratelimit-limit")
    return start["status"], limit.decode() if limit else None


async def _middleware(
    limits: dict[str, int], per_minute: int = 2
) -> tuple[RateLimitMiddleware, _App, TenantRegistry]:
    async def ok(scope, receive, send):  # type: ignore[no-untyped-def]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    registry = _registry([])
    for tenant_id, quota in limits.items():
        await registry.observe(
            Tenant(tenant_id=tenant_id, name=tenant_id, rate_limit_per_minute=quota)
        )
    return (
        RateLimitMiddleware(ok, per_minute=per_minute, burst=0),
        _App(_Container(registry)),
        registry,
    )


async def test_the_default_applies_to_a_tenant_without_an_override() -> None:
    mw, app, _ = await _middleware({})
    assert [(await _status(mw, app, "acme"))[0] for _ in range(3)] == [200, 200, 429]


async def test_a_tenant_override_raises_or_lowers_its_own_budget_only() -> None:
    mw, app, _ = await _middleware({"globex": 5, "tiny": 1})
    assert [(await _status(mw, app, "globex"))[0] for _ in range(6)] == [200] * 5 + [429]
    assert [(await _status(mw, app, "tiny"))[0] for _ in range(2)] == [200, 429]
    assert [(await _status(mw, app, "acme"))[0] for _ in range(3)] == [200, 200, 429]
    assert (await _status(mw, app, "globex"))[1] == "5", "the response names the effective limit"


async def test_a_zero_override_disables_the_limit_for_that_tenant() -> None:
    mw, app, _ = await _middleware({"unlimited": 0})
    assert [(await _status(mw, app, "unlimited"))[0] for _ in range(10)] == [200] * 10


async def test_a_tenant_quota_applies_even_when_the_service_default_is_off() -> None:
    mw, app, _ = await _middleware({"tiny": 1}, per_minute=0)
    assert [(await _status(mw, app, "tiny"))[0] for _ in range(2)] == [200, 429]
    assert [(await _status(mw, app, "free"))[0] for _ in range(5)] == [200] * 5


async def test_a_caller_that_sends_no_tenant_header_is_metered_by_the_tenant_its_key_names() -> (
    None
):
    mw, app, registry = await _middleware({"tiny": 1})
    key_id, token, _ = mint_token()
    registry.observe_key(key_id, "tiny")
    assert [(await _status(mw, app, None, key=token))[0] for _ in range(2)] == [200, 429]
    assert (await _status(mw, app, None, key=f"Bearer {token}"))[0] == 429, (
        "one credential, one bucket"
    )
    _, other_token, _ = mint_token()  # a key the registry has not seen: the default
    assert [(await _status(mw, app, None, key=other_token))[0] for _ in range(3)] == [200, 200, 429]
    registry.forget_key(key_id)
    assert (await _status(mw, app, None, key=token))[0] == 200, (
        "forgotten keys fall back to default"
    )
