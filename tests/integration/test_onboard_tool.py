"""Onboarding grants tenant membership and tenant admin, and nothing wider.

A tenant needs no registration - identifiers are tenant-prefixed everywhere, so the first
write creates it and nothing is required to read your own memories back. What the tool exists
for is the authority that cannot be asserted in a header: tenant admin, which gates forgetting
another principal's memory and widening a tool policy.

``grant_membership`` had no caller anywhere before this - no endpoint, no command - so admin
could never be granted and both branches reading it were unreachable in a running service.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.authz.service import AuthorizationService

pytestmark = pytest.mark.integration


async def test_admin_is_grantable_at_all(container, uow_factory) -> None:
    authz: AuthorizationService = container.services["authz"]
    root = MemoryExecutionContext(tenant_id="arhaus", user_id="root")
    assert not await authz.is_tenant_admin(root)

    async with uow_factory() as uow:
        await authz.grant_membership("arhaus", "root", admin=True, revisions=uow.revisions)
        await uow.commit()
    assert await authz.is_tenant_admin(root)


async def test_admin_in_one_tenant_is_not_admin_in_another(container, uow_factory) -> None:
    """Membership objects are tenant-prefixed, so the same user id is a different principal."""
    authz: AuthorizationService = container.services["authz"]
    async with uow_factory() as uow:
        await authz.grant_membership("arhaus", "carol", admin=True, revisions=uow.revisions)
        await uow.commit()

    assert await authz.is_tenant_admin(MemoryExecutionContext(tenant_id="arhaus", user_id="carol"))
    assert not await authz.is_tenant_admin(
        MemoryExecutionContext(tenant_id="globex", user_id="carol")
    )


async def test_plain_membership_confers_no_admin(container, uow_factory) -> None:
    authz: AuthorizationService = container.services["authz"]
    async with uow_factory() as uow:
        await authz.grant_membership("arhaus", "dave", revisions=uow.revisions)
        await uow.commit()
    assert not await authz.is_tenant_admin(
        MemoryExecutionContext(tenant_id="arhaus", user_id="dave")
    )
