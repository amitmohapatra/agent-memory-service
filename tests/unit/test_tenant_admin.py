"""MEM-05b: who is the tenant's administrator, decided in one place
(``AuthorizationService.is_tenant_admin``).

The docs name two: "anything sent with the tenant's admin key (or the platform key), or by a
tenant admin user" (docs/api/feedback.md). Only the second was read, so the admin key - the
credential the tenant is handed at onboarding - could not forget a memory it did not own.
"""

from __future__ import annotations

import pytest

from memory_service.adapters.authz.memory_provider import MemoryAuthorizationProvider
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.authz.service import AuthorizationService
from memory_service.ports.authorization import AccessCheck, RelationTuple

pytestmark = pytest.mark.unit

USER = MemoryExecutionContext(tenant_id="acme", user_id="u1")
ADMIN_KEY = USER.model_copy(update={"administers_tenant": True})


class _Unreachable(MemoryAuthorizationProvider):
    async def batch_check(self, checks: list[AccessCheck]) -> list[bool]:
        raise ConnectionError("authorization store unreachable")


async def _authz(*tuples: RelationTuple) -> AuthorizationService:
    provider = MemoryAuthorizationProvider()
    await provider.write(list(tuples))
    return AuthorizationService(provider, None)


async def test_the_admin_credential_is_the_tenants_admin_whoever_it_acts_as() -> None:
    authz = await _authz()
    assert await authz.is_tenant_admin(ADMIN_KEY)
    assert await authz.is_tenant_admin(
        ADMIN_KEY.model_copy(update={"agent_id": "concierge", "agent_run_id": "run_1"})
    )
    assert not await authz.is_tenant_admin(USER)


async def test_a_user_the_store_holds_as_admin_still_is_one() -> None:
    authz = await _authz(RelationTuple(user="user:u1", relation="admin", object="tenant:acme"))
    assert await authz.is_tenant_admin(USER)
    assert not await authz.is_tenant_admin(USER.model_copy(update={"tenant_id": "globex"}))


async def test_an_unreachable_store_refuses_rather_than_allows() -> None:
    authz = AuthorizationService(_Unreachable(), None)
    with pytest.raises(ConnectionError):
        await authz.is_tenant_admin(USER)


def test_a_stored_context_does_not_carry_the_requests_authority() -> None:
    """A context saved for later work (a profile's source context) is read back without it."""
    stored = ADMIN_KEY.model_dump(mode="json")
    assert "administers_tenant" not in stored
    assert not MemoryExecutionContext.model_validate(stored).administers_tenant
