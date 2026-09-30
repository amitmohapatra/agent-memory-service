"""The control plane, operation by operation: onboarding, teams, groups, keys, the read audit
and the three levels a model key resolves through.

``test_platform_lifecycle.py`` and ``test_platform_edge_cases.py`` already prove the *behaviour*
these routes exist for (isolation, revocation on the next request, quotas). This file proves the
*contract* of each one: the status it answers, the shape it returns, what a replay does, and
that a listing pages. Twenty-six operations.
"""

from __future__ import annotations

import base64

import pytest

from memory_service.api.app import create_app
from tests.agent.conftest import BOOTSTRAP, sdk
from tests.conftest import PG_AVAILABLE, _test_overrides
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e

ENVELOPE = base64.urlsafe_b64encode(b"p9b-tenancy-envelope-key-32bytes").decode()


@pytest.fixture
def app(make_settings):
    """The agent app with an envelope key, so the model-key levels can be exercised and not
    only refused."""
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings = make_settings(
        authentication={"mode": "api_key", "bootstrap_admin_key": BOOTSTRAP},
        agent_credentials={"active_key_id": "p9b", "encryption_keys": {"p9b": ENVELOPE}},
    )
    return create_app(settings, overrides=_test_overrides(tasks="inline"))


@pytest.mark.covers(
    "admin.create_tenant", "admin.list_tenants", "admin.get_tenant", "admin.update_tenant"
)
async def test_a_platform_operator_onboards_lists_and_suspends_a_tenant(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)

    created = await platform.admin.create_tenant("Acme", tenant_id="acme")
    assert created.tenant.tenant_id == "acme" and created.tenant.status == "active"
    assert created.admin_key.token.startswith("mk_") and created.admin_key.role == "admin"
    await platform.admin.create_tenant("Globex", tenant_id="globex")

    listed = await platform.admin.tenants()
    assert {t.tenant_id for t in listed} >= {"acme", "globex"}

    page = await platform.admin.tenants_page(limit=1)
    assert len(page.items) == 1 and page.next_cursor, "a listing pages with a cursor"
    rest = await platform.admin.tenants_page(limit=1, cursor=page.next_cursor)
    assert rest.items and rest.items[0].tenant_id != page.items[0].tenant_id

    one = await platform.admin.get_tenant("acme")
    assert one.tenant_id == "acme" and one.name == "Acme"

    suspended = await platform.admin.update_tenant("acme", status="suspended")
    assert suspended.status == "suspended"
    assert (await platform.admin.get_tenant("acme")).status == "suspended"
    assert (await platform.admin.update_tenant("acme", status="active")).status == "active"


@pytest.mark.covers(
    "tenancy.create_workspace",
    "tenancy.list_workspaces",
    "tenancy.get_workspace",
    "tenancy.set_member",
    "tenancy.list_members",
    "tenancy.remove_member",
    "tenancy.delete_workspace",
)
async def test_a_tenant_admin_runs_a_team_from_creation_to_deletion(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)

    finance = await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    assert finance.workspace_id == "finance" and finance.name == "Finance"
    await admin.tenant.workspaces.create("Legal", workspace_id="legal")

    listed = await admin.tenant.workspaces.list()
    assert {w.workspace_id for w in listed} == {"finance", "legal"}
    page = await admin.tenant.workspaces.page(limit=1)
    assert len(page.items) == 1 and page.next_cursor

    assert (await admin.tenant.workspaces.get("finance")).name == "Finance"

    member = await admin.tenant.workspaces.set_member("finance", "user:u1")
    assert member.principal == "user:u1" and member.role == "member"
    promoted = await admin.tenant.workspaces.set_member("finance", "user:u1", role="admin")
    assert promoted.role == "admin", "setting a member twice is an update, not a conflict"
    await admin.tenant.workspaces.set_member("finance", "user:u2", role="viewer")
    assert {m.principal: m.role for m in await admin.tenant.workspaces.members("finance")} == {
        "user:u1": "admin",
        "user:u2": "viewer",
    }

    await admin.tenant.workspaces.remove_member("finance", "user:u2")
    assert {m.principal for m in await admin.tenant.workspaces.members("finance")} == {"user:u1"}
    # Removing a principal that is not a member converges rather than failing: an operator
    # script must be safe to re-run.
    await admin.tenant.workspaces.remove_member("finance", "user:u2")

    await admin.tenant.workspaces.delete("finance")
    with pytest.raises(MemoryError) as gone:
        await admin.tenant.workspaces.get("finance")
    assert gone.value.status == 404
    assert {w.workspace_id for w in await admin.tenant.workspaces.list()} == {"legal"}


@pytest.mark.covers(
    "tenancy.create_group",
    "tenancy.list_groups",
    "tenancy.add_group_user",
    "tenancy.list_group_members",
    "tenancy.remove_group_user",
    "tenancy.delete_group",
)
async def test_a_tenant_admin_admits_users_by_group(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)

    counsel = await admin.tenant.groups.create("Counsel", group_id="counsel")
    assert counsel.group_id == "counsel" and counsel.name == "Counsel"
    await admin.tenant.groups.create("Interns", group_id="interns")

    assert {g.group_id for g in await admin.tenant.groups.list()} == {"counsel", "interns"}
    page = await admin.tenant.groups.page(limit=1)
    assert len(page.items) == 1 and page.next_cursor

    added = await admin.tenant.groups.add_user("counsel", "lawyer1")
    assert added.user_id == "lawyer1"
    # Adding the same user twice is the same membership: 200, not 409.
    assert (await admin.tenant.groups.add_user("counsel", "lawyer1")).user_id == "lawyer1"
    await admin.tenant.groups.add_user("counsel", "lawyer2")
    assert {m.user_id for m in await admin.tenant.groups.members("counsel")} == {
        "lawyer1",
        "lawyer2",
    }

    await admin.tenant.groups.remove_user("counsel", "lawyer2")
    assert {m.user_id for m in await admin.tenant.groups.members("counsel")} == {"lawyer1"}
    await admin.tenant.groups.remove_user("counsel", "lawyer2")  # converges

    await admin.tenant.groups.delete("counsel")
    assert {g.group_id for g in await admin.tenant.groups.list()} == {"interns"}


@pytest.mark.covers("tenancy.issue_key", "tenancy.list_keys", "tenancy.revoke_key")
async def test_a_tenant_admin_issues_lists_and_revokes_keys(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)

    issued = await admin.tenant.keys.issue("service", "harness", idempotency_key="issue-1")
    assert issued.token.startswith("mk_") and issued.key_id and issued.role == "service"

    # A replayed issuance is the same record and never a second token (the token is shown once).
    replay = await admin.tenant.keys.issue("service", "harness", idempotency_key="issue-1")
    assert replay.key_id == issued.key_id and replay.token is None

    listed = await admin.tenant.keys.list()
    assert issued.key_id in {k.key_id for k in listed}
    assert all(not getattr(k, "token", None) for k in listed), "a listing never carries tokens"
    assert (await admin.tenant.keys.page(limit=1)).items

    await admin.tenant.keys.revoke(issued.key_id)
    with pytest.raises(MemoryError) as dead:
        await sdk(app, issued.token).bind(user_id="u1").search("anything")
    assert dead.value.status == 401


@pytest.mark.covers("tenancy.list_reads")
async def test_the_read_audit_names_who_read_and_never_the_query(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    service = await admin.tenant.keys.issue("service", "harness")
    harness = sdk(app, service.token)
    await harness.bind(user_id="u1").remember("The budget review is on Monday.", visibility="USER")
    await harness.bind(user_id="u1").search("when is the budget review")

    reads = await admin.tenant.reads()
    assert reads, "a recall is an audited read"
    recall = next(r for r in reads if r.kind == "recall")
    assert recall.principal == "user:u1" and recall.query_hash
    assert "budget" not in recall.query_hash, "the audit keeps a hash, not the question"
    assert (await admin.tenant.reads_page(limit=1)).items


@pytest.mark.covers(
    "tenancy.set_tenant_key",
    "tenancy.tenant_key_status",
    "tenancy.revoke_tenant_key",
    "tenancy.set_workspace_key",
    "tenancy.workspace_key_status",
    "tenancy.revoke_workspace_key",
)
async def test_a_model_key_is_set_read_and_revoked_at_the_tenant_and_the_team(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")

    assert (await admin.tenant.model_key_status()).registered is False
    registered = await admin.tenant.set_model_key("vk-tenant-first")
    assert registered.registered is True and registered.revision == 1
    assert (await admin.tenant.set_model_key("vk-tenant-second")).revision == 2
    status = await admin.tenant.model_key_status()
    assert status.revision == 2 and "vk-tenant" not in str(status.model_dump())

    team = await admin.tenant.workspaces.set_model_key("finance", "vk-finance")
    assert team.registered is True and team.revision == 1
    assert (await admin.tenant.workspaces.model_key_status("finance")).revision == 1
    # The two levels are separate records: setting the team's key leaves the tenant's alone.
    assert (await admin.tenant.model_key_status()).revision == 2

    revoked_team = await admin.tenant.workspaces.revoke_model_key("finance")
    assert revoked_team.revoked is True and revoked_team.revision == 2
    assert (await admin.tenant.workspaces.model_key_status("finance")).revoked is True

    revoked_tenant = await admin.tenant.revoke_model_key()
    assert revoked_tenant.revoked is True and revoked_tenant.revision == 3
    assert (await admin.tenant.model_key_status()).revoked is True


@pytest.mark.covers(
    "tenancy.tenant_policy",
    "tenancy.set_tenant_policy",
    "tenancy.workspace_policy",
    "tenancy.set_workspace_policy",
    "tenancy.tenant_usage",
)
async def test_a_tenant_admin_sets_the_model_policy_and_reads_the_usage(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")

    default = await admin.tenant.model_policy()
    assert default.stored is False and default.read_assist is True and "summaries" in default.uses
    narrowed = await admin.tenant.set_model_policy(["summaries"], read_assist=False)
    assert narrowed.stored and narrowed.uses == ["summaries"] and narrowed.revision == 1
    assert (await admin.tenant.model_policy()).read_assist is False

    # the team's own policy is a separate record; the tenant's is untouched by it
    assert (await admin.tenant.workspaces.model_policy("finance")).stored is False
    team = await admin.tenant.workspaces.set_model_policy(
        "finance", ["reflection", "summaries"], read_assist=True
    )
    assert team.uses == ["reflection", "summaries"]
    assert (await admin.tenant.model_policy()).uses == ["summaries"]

    with pytest.raises(MemoryError) as unknown_use:
        await admin.tenant.set_model_policy(["mind_reading"], read_assist=True)
    assert unknown_use.value.status == 422

    usage = await admin.tenant.model_usage()
    assert usage.days == [] and (usage.until - usage.since).days == 29


@pytest.mark.covers_error(
    "admin.create_tenant",
    "admin.list_tenants",
    "admin.get_tenant",
    "admin.update_tenant",
    "tenancy.create_workspace",
    "tenancy.list_workspaces",
    "tenancy.get_workspace",
    "tenancy.set_member",
    "tenancy.list_members",
    "tenancy.remove_member",
    "tenancy.delete_workspace",
    "tenancy.create_group",
    "tenancy.list_groups",
    "tenancy.add_group_user",
    "tenancy.list_group_members",
    "tenancy.remove_group_user",
    "tenancy.delete_group",
    "tenancy.issue_key",
    "tenancy.list_keys",
    "tenancy.revoke_key",
    "tenancy.list_reads",
    "tenancy.set_tenant_key",
    "tenancy.tenant_key_status",
    "tenancy.revoke_tenant_key",
    "tenancy.set_workspace_key",
    "tenancy.workspace_key_status",
    "tenancy.revoke_workspace_key",
)
async def test_a_service_key_administers_nothing_and_a_tenant_admin_is_not_the_platform(
    app, running
) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    await admin.tenant.groups.create("Counsel", group_id="counsel")
    key = await admin.tenant.keys.issue("service", "harness")
    service = sdk(app, key.token)

    # A data-plane key administers nothing, on any of these routes.
    for call in (
        service.tenant.workspaces.create("Sneaky", workspace_id="sneaky"),
        service.tenant.workspaces.list(),
        service.tenant.workspaces.get("finance"),
        service.tenant.workspaces.set_member("finance", "user:u9"),
        service.tenant.workspaces.members("finance"),
        service.tenant.workspaces.remove_member("finance", "user:u9"),
        service.tenant.workspaces.delete("finance"),
        service.tenant.groups.create("Sneaky", group_id="sneaky"),
        service.tenant.groups.list(),
        service.tenant.groups.add_user("counsel", "u9"),
        service.tenant.groups.members("counsel"),
        service.tenant.groups.remove_user("counsel", "u9"),
        service.tenant.groups.delete("counsel"),
        service.tenant.keys.issue("admin", "escalation"),
        service.tenant.keys.list(),
        service.tenant.keys.revoke(key.key_id),
        service.tenant.reads(),
        service.tenant.set_model_key("vk-sneaky"),
        service.tenant.model_key_status(),
        service.tenant.revoke_model_key(),
        service.tenant.workspaces.set_model_key("finance", "vk-sneaky"),
        service.tenant.workspaces.model_key_status("finance"),
        service.tenant.workspaces.revoke_model_key("finance"),
    ):
        with pytest.raises(MemoryError) as refused:
            await call
        assert refused.value.status == 403, refused.value

    # A tenant admin is not the platform: it cannot onboard, list or reach another tenant.
    for call in (
        admin.administer("globex").workspaces.list(),
        sdk(app, acme.admin_key.token).admin.create_tenant("Globex", tenant_id="globex"),
        sdk(app, acme.admin_key.token).admin.tenants(),
        sdk(app, acme.admin_key.token).admin.get_tenant("acme"),
        sdk(app, acme.admin_key.token).admin.update_tenant("acme", status="suspended"),
    ):
        with pytest.raises(MemoryError) as denied:
            await call
        assert denied.value.status == 403, denied.value

    # The platform key names a tenant nobody onboarded: not found, not a 500.
    with pytest.raises(MemoryError) as missing:
        await platform.admin.get_tenant("nobody")
    assert missing.value.status == 404
