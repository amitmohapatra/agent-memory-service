"""The verifier: hash comparison, expiry, revocation, suspension, and the cache's part in each.

No database: a unit of work fake holds the rows, a dict-backed cache stands in for Dragonfly.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.errors import AuthenticationFailed, AuthorizationFailed
from memory_service.domain.tenancy import KEY_ID_LENGTH, ApiKey, KeyRole, Tenant, mint_token
from memory_service.modules.auth.keys import MISSING, TOMBSTONE, ApiKeyVerifier
from memory_service.ports.cache import CacheUnavailable

pytestmark = pytest.mark.unit


class _Keys:
    def __init__(self, keys: dict[str, ApiKey]) -> None:
        self.rows = keys
        self.touched: list[str] = []
        self.reads = 0

    async def get(self, key_id: str) -> ApiKey | None:
        self.reads += 1
        return self.rows.get(key_id)

    async def list(self, tenant_id: str) -> list[ApiKey]:
        return [k for k in self.rows.values() if k.tenant_id == tenant_id]

    async def touch(self, key_id: str, *, at: datetime) -> None:
        self.touched.append(key_id)
        self.rows[key_id] = self.rows[key_id].model_copy(update={"last_used_at": at})


class _Tenants:
    def __init__(self, tenants: dict[str, Tenant]) -> None:
        self.rows = tenants

    async def get(self, tenant_id: str) -> Tenant | None:
        return self.rows.get(tenant_id)


class _Uow:
    def __init__(self, keys: _Keys, tenants: _Tenants) -> None:
        self.api_keys, self.tenants, self.commits = keys, tenants, 0

    async def commit(self) -> None:
        self.commits += 1


class _Cache:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.reads = 0

    async def get(self, key: str) -> bytes | None:
        self.reads += 1
        return self.data.get(key)

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        self.data[key] = value

    async def set_if_absent(
        self, key: str, value: bytes, *, ttl_seconds: int | None = None
    ) -> bool:
        if key in self.data:
            return False
        self.data[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        return sum(1 for k in keys if self.data.pop(k, None) is not None)


class _DownCache(_Cache):
    async def get(self, key: str) -> bytes | None:
        raise CacheUnavailable("cache away")

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        raise CacheUnavailable("cache away")

    async def set_if_absent(
        self, key: str, value: bytes, *, ttl_seconds: int | None = None
    ) -> bool:
        raise CacheUnavailable("cache away")


class _Clock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _world(
    *,
    status: str = "active",
    expires: datetime | None = None,
    revoked: bool = False,
    touch_every: int = 60,
    clock: _Clock | None = None,
):
    key_id, token, secret_hash = mint_token()
    key = ApiKey(
        key_id=key_id,
        tenant_id="acme",
        role=KeyRole.SERVICE,
        name="svc",
        secret_hash=secret_hash,
        created_by="platform",
        expires_at=expires,
        revoked_at=datetime.now(UTC) if revoked else None,
    )
    keys = _Keys({key_id: key})
    uow = _Uow(keys, _Tenants({"acme": Tenant(tenant_id="acme", name="Acme", status=status)}))  # type: ignore[arg-type]

    @asynccontextmanager
    async def factory():
        yield uow

    cache = _Cache()
    verifier = ApiKeyVerifier(
        factory,  # type: ignore[arg-type]
        cache,  # type: ignore[arg-type]
        touch_every_seconds=touch_every,
        clock=clock or _Clock(),
    )
    return verifier, token, key_id, uow, cache


def _wrong(token: str) -> str:
    return token[:-1] + ("x" if token[-1] != "x" else "y")


async def test_a_valid_key_verifies_and_the_second_use_is_served_from_cache() -> None:
    verifier, token, key_id, uow, cache = _world()
    first = await verifier.verify(token)
    assert (first.tenant_id, first.role, first.key_id) == ("acme", KeyRole.SERVICE, key_id)
    assert uow.api_keys.touched == [key_id] and uow.commits == 1
    await verifier.verify(token)
    assert uow.api_keys.touched == [key_id], "the second use did not touch the store"
    assert uow.api_keys.reads == 1 and cache.reads == 2


async def test_a_wrong_secret_against_a_real_id_is_not_a_use() -> None:
    verifier, token, key_id, uow, _ = _world()
    with pytest.raises(AuthenticationFailed):
        await verifier.verify(_wrong(token))
    assert uow.api_keys.touched == [] and uow.commits == 0, "last_used_at records uses only"
    assert uow.api_keys.rows[key_id].last_used_at is None


async def test_touches_are_bounded_per_window_and_resume_after_it() -> None:
    clock = _Clock()
    verifier, token, key_id, uow, _ = _world(touch_every=60, clock=clock)
    await verifier.verify(token)
    clock.advance(30)
    await verifier.verify(token)
    assert uow.api_keys.touched == [key_id], "inside the window: no second write"
    clock.advance(31)
    await verifier.verify(token)
    assert uow.api_keys.touched == [key_id, key_id], "past the window: touched again"


async def test_a_key_that_expires_while_cached_is_refused_without_a_store_read() -> None:
    clock = _Clock()
    verifier, token, _, uow, _ = _world(expires=clock.now + timedelta(seconds=10), clock=clock)
    await verifier.verify(token)
    clock.advance(11)
    with pytest.raises(AuthenticationFailed):
        await verifier.verify(token)
    assert uow.api_keys.reads == 1, "the cached record already says when it expires"


async def test_the_touch_memo_is_bounded() -> None:
    from memory_service.modules.auth import keys as keys_module

    verifier, token, key_id, _, _ = _world()
    verifier._touched = {f"k{i}": datetime.now(UTC) for i in range(keys_module.MAX_TOUCH_MEMO)}  # noqa: SLF001
    await verifier.verify(token)
    assert list(verifier._touched) == [key_id], "a full memo is cleared, not grown"  # noqa: SLF001


async def test_wrong_secret_unknown_id_expired_and_revoked_all_say_the_same_thing() -> None:
    verifier, token, _, _, _ = _world()
    with pytest.raises(AuthenticationFailed, match="Missing or invalid API key"):
        await verifier.verify(_wrong(token))
    with pytest.raises(AuthenticationFailed, match="Missing or invalid API key"):
        await verifier.verify("mk_" + "z" * KEY_ID_LENGTH + ".secret")
    expired, token, *_ = _world(expires=datetime.now(UTC) - timedelta(seconds=1))
    with pytest.raises(AuthenticationFailed, match="Missing or invalid API key"):
        await expired.verify(token)
    revoked, token, *_ = _world(revoked=True)
    with pytest.raises(AuthenticationFailed, match="Missing or invalid API key"):
        await revoked.verify(token)


async def test_an_unknown_id_is_cached_as_missing_for_a_moment() -> None:
    verifier, _, _, uow, cache = _world()
    ghost = "mk_" + "z" * KEY_ID_LENGTH + ".secret"
    for _ in range(3):
        with pytest.raises(AuthenticationFailed):
            await verifier.verify(ghost)
    assert uow.api_keys.reads == 1, "a flood of garbage tokens costs the store one read"
    assert cache.data[verifier.cache_key("z" * KEY_ID_LENGTH)] == MISSING
    await verifier.forget_missing("z" * KEY_ID_LENGTH)  # what issuing that id does
    with pytest.raises(AuthenticationFailed):
        await verifier.verify(ghost)
    assert uow.api_keys.reads == 2, "the marker is gone; the store is asked again"


async def test_a_suspended_tenant_is_a_policy_refusal_not_a_bad_key() -> None:
    verifier, token, *_ = _world(status="suspended")
    with pytest.raises(AuthorizationFailed, match="suspended"):
        await verifier.verify(token)


async def test_a_key_whose_tenant_row_is_gone_is_refused_as_suspended() -> None:
    verifier, token, _, uow, _ = _world()
    uow.tenants.rows.clear()
    with pytest.raises(AuthorizationFailed, match="suspended"):
        await verifier.verify(token)


async def test_suspension_reaches_a_cached_key_through_invalidate_tenant() -> None:
    verifier, token, key_id, uow, _ = _world()
    await verifier.verify(token)  # cached as active
    uow.tenants.rows["acme"] = uow.tenants.rows["acme"].model_copy(update={"status": "suspended"})
    assert (await verifier.verify(token)).key_id == key_id, "stale cache still serves"
    await verifier.invalidate_tenant("acme")
    with pytest.raises(AuthorizationFailed):
        await verifier.verify(token)


async def test_revocation_reaches_a_cached_key_through_invalidate() -> None:
    verifier, token, key_id, uow, _ = _world()
    await verifier.verify(token)
    uow.api_keys.rows[key_id] = uow.api_keys.rows[key_id].model_copy(
        update={"revoked_at": datetime.now(UTC)}
    )
    await verifier.invalidate(key_id)
    with pytest.raises(AuthenticationFailed):
        await verifier.verify(token)


async def test_a_reader_that_loaded_before_the_revoke_cannot_resurrect_the_key() -> None:
    """The cache-aside race: R reads the row (active), the admin revokes and invalidates,
    then R's write lands. With delete-on-invalidate R would re-cache the active record for a
    whole TTL; with a tombstone and set-if-absent R's write is refused."""
    verifier, token, key_id, uow, cache = _world()
    stale = await verifier._load(key_id)  # noqa: SLF001 - the race is inside the verifier
    assert stale is not None
    cache.data.clear()  # as if the entry had expired before the admin acted
    uow.api_keys.rows[key_id] = uow.api_keys.rows[key_id].model_copy(
        update={"revoked_at": datetime.now(UTC)}
    )
    await verifier.invalidate(key_id)  # tombstone
    await verifier._put(key_id, stale.model_dump_json().encode(), 60)  # noqa: SLF001 - R's late write
    assert cache.data[verifier.cache_key(key_id)] == TOMBSTONE, "the stale write was refused"
    with pytest.raises(AuthenticationFailed):
        await verifier.verify(token)
    kept = cache.data[verifier.cache_key(key_id)]
    assert kept not in (TOMBSTONE, MISSING) and b"revoked_at" in kept, (
        "behind the tombstone the store's terminal answer is cached, never the stale one"
    )


async def test_a_cache_outage_falls_back_to_the_store_and_never_raises() -> None:
    verifier, token, key_id, uow, _ = _world()
    verifier.cache = _DownCache()  # type: ignore[assignment]
    assert (await verifier.verify(token)).key_id == key_id
    assert (await verifier.verify(token)).key_id == key_id
    assert uow.api_keys.reads == 2, "every use reads the store while the cache is away"
    await verifier.invalidate(key_id)  # must not raise
    await verifier.invalidate_tenant("acme")


async def test_no_cache_configured_still_verifies() -> None:
    verifier, token, key_id, *_ = _world()
    verifier.cache = None
    assert (await verifier.verify(token)).key_id == key_id


async def test_malformed_tokens_never_reach_the_store() -> None:
    verifier, _, _, uow, cache = _world()
    for bad in (
        "",
        "Bearer x",
        "mk_short.s",
        "sk_" + "a" * KEY_ID_LENGTH + ".s",
        "mk_" + "A" * KEY_ID_LENGTH + ".s",
    ):
        with pytest.raises(AuthenticationFailed):
            await verifier.verify(bad)
    assert cache.reads == 0 and uow.commits == 0 and uow.api_keys.reads == 0


async def test_a_flood_of_distinct_unknown_ids_is_bounded_per_minute() -> None:
    from memory_service.modules.auth import keys as keys_module

    clock = _Clock()
    verifier, token, key_id, uow, cache = _world(clock=clock)
    verifier.known = lambda k: k == key_id  # what the tenant registry answers
    budget = keys_module.UNKNOWN_IDS_PER_MINUTE
    for i in range(budget + 50):
        with pytest.raises(AuthenticationFailed):
            await verifier.verify(f"mk_{i:016d}.secret")
    assert uow.api_keys.reads == budget, "past the budget unknown ids cost the store nothing"
    cache.data.clear()  # a real key whose entry lapsed during the flood still verifies
    assert (await verifier.verify(token)).key_id == key_id
    assert uow.api_keys.reads == budget + 1
    clock.advance(61)
    with pytest.raises(AuthenticationFailed):
        await verifier.verify("mk_zzzzzzzzzzzzzzzz.secret")
    assert uow.api_keys.reads == budget + 2, "a new minute, a new budget"


async def test_a_repeated_unknown_id_costs_neither_store_nor_cache() -> None:
    verifier, _, _, uow, cache = _world()
    ghost = "mk_" + "q" * KEY_ID_LENGTH + ".secret"
    for _ in range(5):
        with pytest.raises(AuthenticationFailed):
            await verifier.verify(ghost)
    assert uow.api_keys.reads == 1 and cache.reads == 1, "the local copy of MISSING answers"


async def test_a_terminal_answer_replaces_the_tombstone() -> None:
    verifier, token, key_id, uow, cache = _world()
    await verifier.verify(token)
    uow.api_keys.rows[key_id] = uow.api_keys.rows[key_id].model_copy(
        update={"revoked_at": datetime.now(UTC)}
    )
    await verifier.invalidate(key_id)
    for _ in range(3):
        with pytest.raises(AuthenticationFailed):
            await verifier.verify(token)
    assert uow.api_keys.reads == 2, "revoked is cached: one store read behind the tombstone"
    assert cache.data[verifier.cache_key(key_id)] not in (TOMBSTONE, MISSING)


async def test_without_a_registry_the_budget_refuses_nothing() -> None:
    from memory_service.modules.auth import keys as keys_module

    verifier, _, _, uow, _ = _world()
    assert verifier.known is None
    n = keys_module.UNKNOWN_IDS_PER_MINUTE + 5
    for i in range(n):
        with pytest.raises(AuthenticationFailed):
            await verifier.verify(f"mk_{i:016d}.secret")
    assert uow.api_keys.reads == n, "only negative caching, no refusal, without a registry"


async def test_a_suspension_behind_a_tombstone_is_not_cached_because_it_can_be_lifted() -> None:
    verifier, token, key_id, uow, cache = _world()
    await verifier.verify(token)
    uow.tenants.rows["acme"] = uow.tenants.rows["acme"].model_copy(update={"status": "suspended"})
    await verifier.invalidate(key_id)
    with pytest.raises(AuthorizationFailed):
        await verifier.verify(token)
    assert cache.data[verifier.cache_key(key_id)] == TOMBSTONE, "suspended is reversible"
    uow.tenants.rows["acme"] = uow.tenants.rows["acme"].model_copy(update={"status": "active"})
    assert (await verifier.verify(token)).key_id == key_id, "resumed: served at once"


async def test_an_entry_this_build_cannot_read_is_a_miss_not_a_crash() -> None:
    verifier, token, key_id, uow, cache = _world()
    cache.data[verifier.cache_key(key_id)] = b'{"not": "a cached key"}'
    assert (await verifier.verify(token)).key_id == key_id
    assert uow.api_keys.reads == 1
