"""Onboarding grants the membership that makes team-visible memory readable.

A tenant needs no registration - the first write creates it - but a WORKSPACE-visible
memory resolves the reader's workspaces from the authorization store, so until somebody is
granted the workspace it is readable only by whoever wrote it. Measured against the running
service before this tool existed: alice writes a WORKSPACE decision in ws_eng, bob asserts
the same workspace header, and reads nothing.

``grant_membership`` always wrote the right tuples and never had a caller - no endpoint, no
command. These tests pin both halves: before the grant the teammate is excluded, after it
they are not.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.authz.service import AuthorizationService

pytestmark = pytest.mark.integration


def _ctx(user: str, workspace: str | None = None) -> MemoryExecutionContext:
    return MemoryExecutionContext(tenant_id="arhaus", user_id=user, workspace_id=workspace)


async def test_a_teammate_reads_the_workspace_only_after_being_granted_it(
    container, uow_factory
) -> None:
    authz: AuthorizationService = container.services["authz"]
    bob = _ctx("bob", "ws_eng")

    async with uow_factory() as uow:
        before = await authz.scope(bob, revisions=uow.revisions)
        assert before.workspace_ids == [], "asserting the header is not being in the team"

        await authz.grant_membership(
            "arhaus", "bob", workspaces=["ws_eng"], revisions=uow.revisions
        )
        after = await authz.scope(bob, revisions=uow.revisions)
        assert after.workspace_ids == ["ws_eng"]
        await uow.commit()


async def test_admin_is_grantable_at_all(container, uow_factory) -> None:
    """``is_tenant_admin`` gates forgetting anyone's memory and widening tool policy.

    Nothing could grant it before, so both branches were unreachable in a running service.
    """
    authz: AuthorizationService = container.services["authz"]
    root = _ctx("root")
    assert not await authz.is_tenant_admin(root)

    async with uow_factory() as uow:
        await authz.grant_membership("arhaus", "root", admin=True, revisions=uow.revisions)
        await uow.commit()
    assert await authz.is_tenant_admin(root)


async def test_a_grant_does_not_leak_across_tenants(container, uow_factory) -> None:
    """Membership objects are tenant-prefixed, so the same workspace name is a different team."""
    authz: AuthorizationService = container.services["authz"]
    async with uow_factory() as uow:
        await authz.grant_membership(
            "arhaus", "carol", workspaces=["ws_eng"], revisions=uow.revisions
        )
        await uow.commit()

    other = MemoryExecutionContext(tenant_id="globex", user_id="carol", workspace_id="ws_eng")
    async with uow_factory() as uow:
        assert (await authz.scope(other, revisions=uow.revisions)).workspace_ids == []
