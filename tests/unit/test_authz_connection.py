"""Resolving a caller's authorization scope (the cache, then the relationship store over the
network) holds no database connection: only the revision read does."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.authz.service import AuthorizationService
from memory_service.ports.authorization import AuthorizedScope

pytestmark = pytest.mark.unit


class _Revisions:
    async def get_many(self, tenant_id, keys):
        return {f"{kind.value}:{ident}": 3 for kind, ident in keys}


class _Factory:
    """A unit of work factory that knows whether one is open."""

    def __init__(self) -> None:
        self.open = 0

    @asynccontextmanager
    async def __call__(self):
        self.open += 1
        try:
            yield type("UoW", (), {"revisions": _Revisions()})()
        finally:
            self.open -= 1


class _Cache:
    def __init__(self, factory: _Factory) -> None:
        self.factory = factory
        self.open_during: list[int] = []
        self.kept: dict[str, bytes] = {}

    async def get(self, key):
        self.open_during.append(self.factory.open)
        return self.kept.get(key)

    async def set(self, key, value, ttl_seconds=None):
        self.open_during.append(self.factory.open)
        self.kept[key] = value


async def test_the_scope_is_resolved_and_cached_with_no_unit_of_work_open() -> None:
    factory = _Factory()
    cache = _Cache(factory)
    service = AuthorizationService(provider=None, cache=cache)  # type: ignore[arg-type]
    resolved: list[int] = []

    async def resolve(ctx):
        resolved.append(factory.open)
        return AuthorizedScope(tenant_id=ctx.tenant_id, principal=ctx.principal_id)

    service.resolver.resolve = resolve  # type: ignore[method-assign]
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1")
    first = await service.visibility(ctx, uow_factory=factory)  # type: ignore[arg-type]
    again = await service.visibility(ctx, uow_factory=factory)  # type: ignore[arg-type]
    assert resolved == [0], "resolved once, and with no connection held"
    assert cache.open_during and set(cache.open_during) == {0}
    assert first == again
