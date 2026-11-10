"""A person's "Always allow" for a tool, end to end through the SDK: the harness writes it with
its service key, reads it back for the user's later runs, and the user (or anyone acting for
them) revokes it. An agent's or a user's own credential cannot grant it, and no tenant reads
or revokes another's."""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk
from trellis.memory import AuthorizationError, NotFoundError

pytestmark = pytest.mark.e2e


async def _tenant(app, tenant_id: str):
    """A tenant, its administrator and the service key a harness holds."""
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    return admin, sdk(app, service.token)


@pytest.mark.covers(
    "tools.list_tool_permissions", "tools.allow_tool", "tools.revoke_tool_permission"
)
async def test_the_harness_allows_a_tool_for_a_user_and_the_user_revokes_it(app, running) -> None:
    _, harness = await _tenant(app, "acme")
    alice = harness.bind(user_id="alice").advanced.tool_permissions
    assert await alice.list() == []

    first = await alice.allow("crm-update_quote", allowed_by="alice")
    again = await alice.allow("crm-update_quote", allowed_by="alice")
    assert again.permission_id == first.permission_id, "allowing twice keeps one permission"
    for_finance = await alice.allow("crm-update_quote", assignee="finance")
    assert for_finance.permission_id != first.permission_id
    assert for_finance.allowed_by == "alice", "allowed by the user unless named"

    listed = await alice.list()
    assert [(p.tool, p.assignee) for p in listed] == [
        ("crm-update_quote", None),
        ("crm-update_quote", "finance"),
    ]
    assert await harness.bind(user_id="bob").advanced.tool_permissions.list() == []

    await alice.revoke(first.permission_id, idempotency_key="revoke-1")
    await alice.revoke(first.permission_id, idempotency_key="revoke-1")  # the 204 replayed
    assert [p.permission_id for p in await alice.list()] == [for_finance.permission_id]
    with pytest.raises(NotFoundError):
        await alice.revoke(first.permission_id)


@pytest.mark.covers_error("tools.allow_tool")
async def test_only_the_service_credential_grants_a_permission(app, running) -> None:
    admin, harness = await _tenant(app, "acme")
    # an agent's key that may act for alice, and alice's own: neither may act for anyone
    agent_key = await admin.tenant.keys.issue(
        "service", "writer", may_act_as=["agent:writer", "user:alice"]
    )
    user_key = await admin.tenant.keys.issue("service", "alice", may_act_as=["user:alice"])
    for token in (agent_key.token, user_key.token):
        own = sdk(app, token).bind(user_id="alice", agent_id="writer")
        with pytest.raises(AuthorizationError):
            await own.advanced.tool_permissions.allow("crm-update_quote")
    with pytest.raises(AuthorizationError):  # an administrator administers; it does not grant
        await admin.bind(user_id="alice").advanced.tool_permissions.allow("crm-update_quote")
    assert await harness.bind(user_id="alice").advanced.tool_permissions.list() == []


@pytest.mark.covers_error("tools.list_tool_permissions", "tools.revoke_tool_permission")
async def test_a_tenant_never_reads_or_revokes_another_tenants_permission(app, running) -> None:
    _, acme = await _tenant(app, "acme")
    _, globex = await _tenant(app, "globex")
    granted = await acme.bind(user_id="alice").advanced.tool_permissions.allow("crm-update_quote")

    theirs = globex.bind(user_id="alice").advanced.tool_permissions
    assert await theirs.list() == []
    with pytest.raises(NotFoundError):
        await theirs.revoke(granted.permission_id)
    with pytest.raises(AuthorizationError):  # a key names its tenant; another is refused
        await globex.bind(tenant_id="acme", user_id="alice").advanced.tool_permissions.list()
    remaining = await acme.bind(user_id="alice").advanced.tool_permissions.list()
    assert [p.permission_id for p in remaining] == [granted.permission_id]


async def test_a_key_for_one_user_reads_and_revokes_only_that_users_permissions(
    app, running
) -> None:
    admin, harness = await _tenant(app, "acme")
    bobs = await harness.bind(user_id="bob").advanced.tool_permissions.allow("crm-update_quote")
    alice_key = await admin.tenant.keys.issue("service", "alice", may_act_as=["user:alice"])
    alice = sdk(app, alice_key.token)
    with pytest.raises(AuthorizationError):
        await alice.bind(user_id="bob").advanced.tool_permissions.list()
    with pytest.raises(NotFoundError):  # not alice's: as if it were not there
        await alice.bind(user_id="alice").advanced.tool_permissions.revoke(bobs.permission_id)
    assert await alice.bind(user_id="alice").advanced.tool_permissions.list() == []
