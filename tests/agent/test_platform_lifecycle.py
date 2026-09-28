"""Onboard a tenant, give a team a key, share inside the team, and take it all back again -
through the SDK alone, the way a harness would.

    a workspace memory is read by the team and by nobody else
    another tenant's key cannot name this tenant
    a removed member is denied on its next request
    a revoked key is refused on its next request
    the read audit shows who read
"""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk
from universal_memory import MemoryError

pytestmark = pytest.mark.e2e

FACT = "The Q3 forecast review happens every Tuesday at 10:00 in the finance workspace."
QUERY = "when is the Q3 forecast review"


def _mentions(items, needle: str = "Q3 forecast") -> bool:
    return any(needle in (getattr(i, "text", "") or "") for i in items)


async def test_onboard_share_revoke(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)

    # -- onboarding: the platform key creates tenants and nothing else -------------------
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    assert acme.admin_key.token.startswith("mk_") and acme.tenant.status == "active"
    with pytest.raises(MemoryError) as denied:
        await platform.bind(tenant_id="acme", user_id="u1").recall(QUERY)
    assert denied.value.status == 403

    # -- the tenant admin builds a team and hands its harness a service key -------------
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    await admin.tenant.workspaces.set_member("finance", "user:u1")
    await admin.tenant.workspaces.set_member("finance", "user:u2")
    service = await admin.tenant.keys.issue("service", "finance-harness")
    assert {m.principal for m in await admin.tenant.workspaces.members("finance")} == {
        "user:u1",
        "user:u2",
    }

    # -- the harness: no tenant_id anywhere, the key names it ----------------------------
    harness = sdk(app, service.token)
    u1 = harness.bind(user_id="u1", workspace_id="finance")
    await u1.remember(FACT, visibility="WORKSPACE")

    u2 = harness.bind(user_id="u2", workspace_id="finance")
    assert _mentions(await u2.recall(QUERY)), "a team member reads what the team stored"
    u3 = harness.bind(user_id="u3", workspace_id="finance")
    assert not _mentions(await u3.recall(QUERY)), "a non-member reads nothing of it"
    outside = harness.bind(user_id="u2")  # a member, asking outside any workspace
    assert _mentions(await outside.recall(QUERY)), "no workspace named: every team I am in"

    # -- another tenant's key cannot reach this tenant, whatever it claims ---------------
    globex = await platform.admin.create_tenant("Globex", tenant_id="globex")
    gkey = await sdk(app, globex.admin_key.token).tenant.keys.issue("service", "h")
    with pytest.raises(MemoryError) as cross:
        await sdk(app, gkey.token).bind(tenant_id="acme", user_id="u1").recall(QUERY)
    assert cross.value.status == 403
    assert not _mentions(await sdk(app, gkey.token).bind(user_id="u1").recall(QUERY))

    # -- revocation is immediate --------------------------------------------------------
    await admin.tenant.workspaces.remove_member("finance", "user:u2")
    assert not _mentions(await u2.recall(QUERY)), "removed on this request, not at a TTL"
    assert _mentions(await u1.recall(QUERY)), "the author keeps reading its own memory"

    await admin.tenant.keys.revoke(service.key_id)
    with pytest.raises(MemoryError) as revoked:
        await u1.recall(QUERY)
    assert revoked.value.status == 401

    # -- the audit knows who read -------------------------------------------------------
    reads = await admin.tenant.reads()
    principals = {r.principal for r in reads if r.kind == "recall"}
    assert {"user:u1", "user:u2", "user:u3"} <= principals
    assert all(r.query_hash and not r.query_hash.startswith("when") for r in reads)


async def test_groups_admit_users_at_once_and_a_bound_key_is_pinned(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Legal", workspace_id="legal")
    await admin.tenant.groups.create("Counsel", group_id="counsel")
    await admin.tenant.groups.add_user("counsel", "lawyer1")
    await admin.tenant.groups.add_user("counsel", "lawyer2")
    await admin.tenant.workspaces.set_member("legal", "group:counsel")

    pinned = await admin.tenant.keys.issue("service", "legal-bot", workspace_id="legal")
    bot = sdk(app, pinned.token)
    await bot.bind(user_id="lawyer1").remember(
        "Outside counsel invoices are approved by the general counsel.", visibility="WORKSPACE"
    )
    ask = "who approves invoices"
    assert _mentions(await bot.bind(user_id="lawyer2").recall(ask), "general counsel")
    with pytest.raises(MemoryError) as elsewhere:
        await bot.bind(user_id="lawyer2", workspace_id="finance").recall(ask)
    assert elsewhere.value.status == 403

    # leaving the group is leaving the workspace; the author alone keeps its own memory
    await admin.tenant.groups.remove_user("counsel", "lawyer2")
    assert not _mentions(await bot.bind(user_id="lawyer2").recall(ask), "general counsel")
    assert _mentions(await bot.bind(user_id="lawyer1").recall(ask), "general counsel")
