"""Onboarding and team administration (``modules.tenancy.service``) against PostgreSQL.

Every rule the HTTP routes rely on is the service's: a tenant id is checked before it reaches
a query, a name that is only control characters is no name, an admin key is tenant-wide, a
suspended tenant gets no new keys, a workspace id is never reused, and every membership
change lands in the authorization store with a bumped membership revision.

The tenancy tables are not truncated between tests, so each test onboards tenants of its own.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.conversation import Thread
from memory_service.domain.errors import Conflict, NotFound, ValidationFailed
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.tenancy import KeyRole, parse_token, secret_matches
from memory_service.modules.authz.scope import object_id
from memory_service.modules.tenancy import service as tenancy_module
from memory_service.modules.tenancy.service import ACTOR_MAX, TenancyService

pytestmark = pytest.mark.integration


def _fresh(prefix: str = "t") -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


@pytest.fixture
def tenancy(container) -> TenancyService:
    return container.services["tenancy"]


async def _onboard(uow_factory, tenancy: TenancyService, tenant_id: str | None = None) -> str:
    async with uow_factory() as uow:
        tenant, _ = await tenancy.create_tenant(
            uow, name="Acme", created_by="platform", tenant_id=tenant_id or _fresh()
        )
        await uow.commit()
    return tenant.tenant_id


async def _team(uow_factory, tenancy: TenancyService, tenant_id: str, ws: str = "fin") -> str:
    async with uow_factory() as uow:
        workspace = await tenancy.create_workspace(uow, tenant_id, name="Finance", workspace_id=ws)
        await uow.commit()
    return workspace.workspace_id


async def _members_in_store(container, tenant_id: str, ws: str, principal: str) -> set[str]:
    """The workspace relations the authorization store resolves for ``principal`` (direct
    or implied: admin implies member, member implies viewer)."""
    provider = container.services["authz"].provider
    obj = f"workspace:{object_id(tenant_id, ws)}"
    return {
        role
        for role in ("admin", "member", "viewer")
        if obj in await provider.list_objects(principal, role, "workspace")
    }


async def _membership_revision(uow_factory, tenant_id: str, ident: str) -> int:
    async with uow_factory() as uow:
        found = await uow.revisions.get_many(tenant_id, [(RevisionKind.MEMBERSHIP, ident)])
    return found.get(f"membership:{ident}", 0)


# --------------------------------------------------------------------------- tenants


async def test_onboarding_creates_the_tenant_and_a_working_first_admin_key(
    uow_factory, tenancy
) -> None:
    tenant_id = _fresh("acme")
    async with uow_factory() as uow:
        tenant, admin = await tenancy.create_tenant(
            uow,
            name="  Acme\x00 Corp ",
            created_by="platform",
            tenant_id=tenant_id,
            retention_days=30,
            rate_limit_per_minute=60,
        )
        await uow.commit()
    assert tenant.name == "Acme Corp"
    assert (tenant.retention_days, tenant.rate_limit_per_minute) == (30, 60)
    assert admin.key.role is KeyRole.ADMIN and admin.key.name == "initial admin"
    parsed = parse_token(admin.token)
    assert parsed is not None and parsed[0] == admin.key.key_id
    assert secret_matches(parsed[1], admin.key.secret_hash)
    async with uow_factory() as uow:
        stored = await tenancy.get_tenant(uow, tenant_id)
        [key] = await tenancy.list_keys(uow, tenant_id)
    assert stored.tenant_id == tenant_id and stored.status == "active"
    assert key.key_id == admin.key.key_id and key.created_by == "platform"


async def test_a_tenant_without_an_id_is_given_a_generated_one(uow_factory, tenancy) -> None:
    async with uow_factory() as uow:
        tenant, _ = await tenancy.create_tenant(uow, name="Unnamed", created_by="platform")
        await uow.commit()
    assert tenant.tenant_id
    async with uow_factory() as uow:
        assert (await tenancy.get_tenant(uow, tenant.tenant_id)).name == "Unnamed"


async def test_onboarding_one_id_twice_is_a_conflict(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        with pytest.raises(Conflict, match="already exists"):
            await tenancy.create_tenant(uow, name="Again", created_by="p", tenant_id=tenant_id)


@pytest.mark.parametrize("tenant_id", ["platform", "acme:evil", "a/b", "bad\x00id"])
async def test_a_reserved_or_malformed_tenant_id_is_refused(
    uow_factory, tenancy, tenant_id
) -> None:
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="invalid tenant_id"):
            await tenancy.create_tenant(uow, name="X", created_by="p", tenant_id=tenant_id)


async def test_a_name_of_only_control_characters_is_refused(uow_factory, tenancy) -> None:
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="name is empty"):
            await tenancy.create_tenant(uow, name="\x00\x01 ", created_by="p", tenant_id=_fresh())


async def test_an_unknown_tenant_is_not_found(uow_factory, tenancy) -> None:
    async with uow_factory() as uow:
        with pytest.raises(NotFound, match="Tenant not found"):
            await tenancy.get_tenant(uow, _fresh("ghost"))


async def test_a_malformed_tenant_id_never_reaches_a_lookup(uow_factory, tenancy) -> None:
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="invalid tenant_id"):
            await tenancy.get_tenant(uow, "../etc")


async def test_tenants_are_listed_in_id_order_after_a_cursor(uow_factory, tenancy) -> None:
    stem = _fresh("page")
    ids = [await _onboard(uow_factory, tenancy, f"{stem}-{n}") for n in ("a", "b", "c")]
    async with uow_factory() as uow:
        after_first = await tenancy.list_tenants(uow, after=ids[0], limit=2)
        from_start = await tenancy.list_tenants(uow, limit=100_000)
    assert [t.tenant_id for t in after_first] == ids[1:]
    listed = [t.tenant_id for t in from_start]
    assert listed == sorted(listed) and set(ids) <= set(listed)


async def test_a_malformed_tenant_cursor_is_refused(uow_factory, tenancy) -> None:
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed):
            await tenancy.list_tenants(uow, after="bad\x00cursor")


async def test_updating_a_tenant_reports_whether_its_status_changed(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        renamed, changed = await tenancy.update_tenant(
            uow, tenant_id, name="Acme Two", retention_days=7, rate_limit_per_minute=5
        )
        await uow.commit()
    assert not changed
    assert (renamed.name, renamed.retention_days, renamed.rate_limit_per_minute) == (
        "Acme Two",
        7,
        5,
    )
    async with uow_factory() as uow:
        suspended, changed = await tenancy.update_tenant(uow, tenant_id, status="suspended")
        await uow.commit()
    assert changed and suspended.status == "suspended"
    async with uow_factory() as uow:
        _, changed_again = await tenancy.update_tenant(uow, tenant_id, status="suspended")
        stored = await tenancy.get_tenant(uow, tenant_id)
    assert not changed_again
    assert (stored.name, stored.status, stored.retention_days) == ("Acme Two", "suspended", 7)


async def test_retention_and_rate_limit_can_be_cleared_back_to_the_default(
    uow_factory, tenancy
) -> None:
    async with uow_factory() as uow:
        tenant, _ = await tenancy.create_tenant(
            uow,
            name="Acme",
            created_by="p",
            tenant_id=_fresh(),
            retention_days=30,
            rate_limit_per_minute=60,
        )
        await uow.commit()
    async with uow_factory() as uow:
        cleared, _ = await tenancy.update_tenant(
            uow,
            tenant.tenant_id,
            retention_days=99,  # clearing wins over a value sent beside it
            clear_retention=True,
            clear_rate_limit=True,
        )
        await uow.commit()
    assert cleared.retention_days is None and cleared.rate_limit_per_minute is None
    async with uow_factory() as uow:
        stored = await tenancy.get_tenant(uow, tenant.tenant_id)
    assert stored.retention_days is None and stored.rate_limit_per_minute is None
    assert stored.name == "Acme"  # what was not named is left alone


async def test_updating_an_unknown_tenant_is_not_found(uow_factory, tenancy) -> None:
    async with uow_factory() as uow:
        with pytest.raises(NotFound):
            await tenancy.update_tenant(uow, _fresh("ghost"), name="x")


# --------------------------------------------------------------------------- keys


async def test_a_service_key_is_issued_with_its_binding_expiry_and_principals(
    uow_factory, tenancy
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id)
    before = datetime.now(UTC)
    async with uow_factory() as uow:
        issued = await tenancy.issue_key(
            uow,
            tenant_id,
            role=KeyRole.SERVICE,
            name="harness",
            created_by="x" * (ACTOR_MAX + 40),
            workspace_id=ws,
            expires_in_days=10,
            may_act_as=["user:alice", "agent:bot", "user:alice"],
        )
        await uow.commit()
    key = issued.key
    assert key.role is KeyRole.SERVICE and key.workspace_id == ws
    assert key.may_act_as == ["user:alice", "agent:bot"]
    assert len(key.created_by) == ACTOR_MAX
    assert key.expires_at is not None
    assert timedelta(days=10) <= key.expires_at - before <= timedelta(days=10, minutes=5)
    async with uow_factory() as uow:
        keys = {k.key_id: k for k in await tenancy.list_keys(uow, tenant_id)}
    assert keys[key.key_id].may_act_as == ["user:alice", "agent:bot"]


async def test_a_key_without_an_expiry_never_expires(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        issued = await tenancy.issue_key(
            uow, tenant_id, role=KeyRole.SERVICE, name="svc", created_by="admin"
        )
    assert issued.key.expires_at is None and issued.key.may_act_as == ["*"]


async def test_the_platform_role_is_never_issued(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="never issued"):
            await tenancy.issue_key(
                uow, tenant_id, role=KeyRole.PLATFORM, name="root", created_by="admin"
            )


async def test_an_admin_key_cannot_be_bound_to_a_workspace(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id)
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="tenant-wide"):
            await tenancy.issue_key(
                uow, tenant_id, role=KeyRole.ADMIN, name="a", created_by="x", workspace_id=ws
            )


async def test_a_suspended_tenant_is_issued_no_keys(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        await tenancy.update_tenant(uow, tenant_id, status="suspended")
        await uow.commit()
    async with uow_factory() as uow:
        with pytest.raises(Conflict, match="is suspended"):
            await tenancy.issue_key(uow, tenant_id, role=KeyRole.SERVICE, name="s", created_by="x")


async def test_a_key_bound_to_an_unknown_workspace_is_refused(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        with pytest.raises(NotFound, match="Workspace not found"):
            await tenancy.issue_key(
                uow, tenant_id, role=KeyRole.SERVICE, name="s", created_by="x", workspace_id="nope"
            )


async def test_a_key_for_an_unknown_tenant_is_refused(uow_factory, tenancy) -> None:
    async with uow_factory() as uow:
        with pytest.raises(NotFound):
            await tenancy.issue_key(
                uow, _fresh("ghost"), role=KeyRole.SERVICE, name="s", created_by="x"
            )


@pytest.mark.parametrize(
    "principals", [["group:eng"], ["user:"], [f"user:u{i}" for i in range(101)]]
)
async def test_a_key_may_act_only_for_well_formed_principals(
    uow_factory, tenancy, principals
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed):
            await tenancy.issue_key(
                uow,
                tenant_id,
                role=KeyRole.SERVICE,
                name="s",
                created_by="x",
                may_act_as=principals,
            )


async def test_a_tenant_at_its_key_cap_must_revoke_before_issuing(
    uow_factory, tenancy, monkeypatch
) -> None:
    monkeypatch.setattr(tenancy_module, "MAX_KEYS_PER_TENANT", 2)
    tenant_id = await _onboard(uow_factory, tenancy)  # the initial admin is the first
    async with uow_factory() as uow:
        second = await tenancy.issue_key(
            uow, tenant_id, role=KeyRole.SERVICE, name="s", created_by="x"
        )
        await uow.commit()
    async with uow_factory() as uow:
        with pytest.raises(Conflict, match="holds 2 live keys"):
            await tenancy.issue_key(uow, tenant_id, role=KeyRole.SERVICE, name="t", created_by="x")
    async with uow_factory() as uow:
        assert await tenancy.revoke_key(uow, tenant_id, second.key.key_id)
        await uow.commit()
    async with uow_factory() as uow:
        third = await tenancy.issue_key(
            uow, tenant_id, role=KeyRole.SERVICE, name="t", created_by="x"
        )
    assert third.key.key_id != second.key.key_id


async def test_whom_a_key_acts_for_can_be_replaced(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        issued = await tenancy.issue_key(
            uow, tenant_id, role=KeyRole.SERVICE, name="s", created_by="x"
        )
        await uow.commit()
    async with uow_factory() as uow:
        updated = await tenancy.set_may_act_as(
            uow, tenant_id, issued.key.key_id, ["agent:bot", "agent:bot"]
        )
        await uow.commit()
    assert updated.may_act_as == ["agent:bot"]
    async with uow_factory() as uow:
        [stored] = [
            k for k in await tenancy.list_keys(uow, tenant_id) if k.key_id == issued.key.key_id
        ]
    assert stored.may_act_as == ["agent:bot"]


async def test_replacing_principals_of_another_tenants_key_is_not_found(
    uow_factory, tenancy
) -> None:
    mine = await _onboard(uow_factory, tenancy)
    theirs = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        issued = await tenancy.issue_key(
            uow, theirs, role=KeyRole.SERVICE, name="s", created_by="x"
        )
        await uow.commit()
    async with uow_factory() as uow:
        with pytest.raises(NotFound, match="not found"):
            await tenancy.set_may_act_as(uow, mine, issued.key.key_id, ["*"])
        with pytest.raises(ValidationFailed):
            await tenancy.set_may_act_as(uow, theirs, issued.key.key_id, ["robot:r2"])


async def test_keys_page_oldest_first_from_a_cursor(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        for name in ("one", "two"):
            await tenancy.issue_key(uow, tenant_id, role=KeyRole.SERVICE, name=name, created_by="x")
        await uow.commit()
    async with uow_factory() as uow:
        everything = await tenancy.list_keys(uow, tenant_id)
        first = everything[0]
        rest = await tenancy.list_keys(uow, tenant_id, after=(first.created_at, first.key_id))
        one = await tenancy.list_keys(uow, tenant_id, limit=1)
    assert len(everything) == 3
    assert [k.key_id for k in rest] == [k.key_id for k in everything[1:]]
    assert [k.key_id for k in one] == [first.key_id]


async def test_revoking_is_idempotent_and_scoped_to_the_tenant(uow_factory, tenancy) -> None:
    mine = await _onboard(uow_factory, tenancy)
    theirs = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        issued = await tenancy.issue_key(
            uow, theirs, role=KeyRole.SERVICE, name="s", created_by="x"
        )
        await uow.commit()
    key_id = issued.key.key_id
    async with uow_factory() as uow:
        assert not await tenancy.revoke_key(uow, mine, key_id), "another tenant's key"
        assert await tenancy.revoke_key(uow, theirs, key_id)
        await uow.commit()
    async with uow_factory() as uow:
        assert not await tenancy.revoke_key(uow, theirs, key_id), "already revoked"
        [stored] = [k for k in await tenancy.list_keys(uow, theirs) if k.key_id == key_id]
    assert stored.revoked_at is not None


async def test_a_malformed_key_id_is_refused_before_revocation(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="invalid key_id"):
            await tenancy.revoke_key(uow, tenant_id, "bad\x00key")


# --------------------------------------------------------------------------- workspaces


async def test_a_workspace_is_created_and_granted_to_its_tenant(
    container, uow_factory, tenancy
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        workspace = await tenancy.create_workspace(uow, tenant_id, name=" Finance\x07 ")
        await uow.commit()
    assert workspace.name == "Finance" and workspace.workspace_id
    provider = container.services["authz"].provider
    obj = f"workspace:{object_id(tenant_id, workspace.workspace_id)}"
    assert obj in await provider.list_objects(f"tenant:{tenant_id}", "tenant", "workspace")
    async with uow_factory() as uow:
        stored = await tenancy.get_workspace(uow, tenant_id, workspace.workspace_id)
        listed = await tenancy.list_workspaces(uow, tenant_id)
    assert stored.name == "Finance"
    assert [w.workspace_id for w in listed] == [workspace.workspace_id]


async def test_workspaces_are_listed_in_id_order_after_a_cursor(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    for ws in ("ops", "eng", "fin"):
        await _team(uow_factory, tenancy, tenant_id, ws)
    async with uow_factory() as uow:
        assert [w.workspace_id for w in await tenancy.list_workspaces(uow, tenant_id)] == [
            "eng",
            "fin",
            "ops",
        ]
        after = await tenancy.list_workspaces(uow, tenant_id, after="eng", limit=1)
        with pytest.raises(ValidationFailed, match="invalid workspace_id"):
            await tenancy.list_workspaces(uow, tenant_id, after="a\x00b")
    assert [w.workspace_id for w in after] == ["fin"]


async def test_a_workspace_id_is_never_reused(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id)
    async with uow_factory() as uow:
        with pytest.raises(Conflict, match="exists or was used before"):
            await tenancy.create_workspace(uow, tenant_id, name="Again", workspace_id=ws)
    async with uow_factory() as uow:
        await tenancy.delete_workspace(uow, tenant_id, ws)
        await uow.commit()
    async with uow_factory() as uow:
        with pytest.raises(Conflict, match="exists or was used before"):
            await tenancy.create_workspace(uow, tenant_id, name="Reborn", workspace_id=ws)


async def test_an_id_already_used_as_a_thread_anchor_cannot_become_a_team(
    uow_factory, tenancy
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        await uow.threads.add(Thread(tenant_id=tenant_id, workspace_id="sales", title="t"))
        await uow.commit()
    async with uow_factory() as uow:
        with pytest.raises(Conflict, match="in use as an anchor"):
            await tenancy.create_workspace(uow, tenant_id, name="Sales", workspace_id="sales")


async def test_a_malformed_workspace_id_is_refused(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="invalid workspace_id"):
            await tenancy.create_workspace(uow, tenant_id, name="x", workspace_id="a\x00b")
        with pytest.raises(ValidationFailed, match="invalid workspace_id"):
            await tenancy.get_workspace(uow, tenant_id, "a\x00b")


async def test_deleting_a_workspace_revokes_its_members_and_bound_keys_only(
    container, uow_factory, tenancy
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id, "fin")
    other = await _team(uow_factory, tenancy, tenant_id, "ops")
    async with uow_factory() as uow:
        await tenancy.set_member(uow, tenant_id, ws, "user:alice", role="admin", added_by="root")
        await tenancy.set_member(uow, tenant_id, ws, "agent:bot", role="member", added_by="root")
        bound = await tenancy.issue_key(
            uow, tenant_id, role=KeyRole.SERVICE, name="b", created_by="x", workspace_id=ws
        )
        dead = await tenancy.issue_key(
            uow, tenant_id, role=KeyRole.SERVICE, name="d", created_by="x", workspace_id=ws
        )
        elsewhere = await tenancy.issue_key(
            uow, tenant_id, role=KeyRole.SERVICE, name="e", created_by="x", workspace_id=other
        )
        await uow.commit()
    async with uow_factory() as uow:
        assert await tenancy.revoke_key(uow, tenant_id, dead.key.key_id)
        await uow.commit()
    alice_before = await _membership_revision(uow_factory, tenant_id, "alice")

    async with uow_factory() as uow:
        revoked = await tenancy.delete_workspace(uow, tenant_id, ws)
        await uow.commit()

    # the already-revoked key is not revoked again, and another team's key is untouched
    assert revoked == [bound.key.key_id]
    assert await _members_in_store(container, tenant_id, ws, "user:alice") == set()
    assert await _members_in_store(container, tenant_id, ws, "agent:bot") == set()
    assert await _membership_revision(uow_factory, tenant_id, "alice") > alice_before
    async with uow_factory() as uow:
        keys = {k.key_id: k for k in await tenancy.list_keys(uow, tenant_id)}
        listed = [w.workspace_id for w in await tenancy.list_workspaces(uow, tenant_id)]
        with pytest.raises(NotFound):
            await tenancy.get_workspace(uow, tenant_id, ws)
    assert keys[bound.key.key_id].revoked_at is not None
    assert keys[elsewhere.key.key_id].revoked_at is None
    assert listed == ["ops"]


async def test_deleting_a_deleted_workspace_converges_and_an_unknown_one_is_not_found(
    uow_factory, tenancy
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id)
    async with uow_factory() as uow:
        assert await tenancy.delete_workspace(uow, tenant_id, ws) == []
        await uow.commit()
    async with uow_factory() as uow:
        assert await tenancy.delete_workspace(uow, tenant_id, ws) == []
        with pytest.raises(NotFound, match="Workspace not found"):
            await tenancy.delete_workspace(uow, tenant_id, "never")
        with pytest.raises(ValidationFailed):
            await tenancy.delete_workspace(uow, tenant_id, "a\x00b")


# --------------------------------------------------------------------------- members


async def test_a_member_holds_exactly_one_role_in_the_store(
    container, uow_factory, tenancy
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id)
    start = await _membership_revision(uow_factory, tenant_id, "alice")
    async with uow_factory() as uow:
        member = await tenancy.set_member(
            uow, tenant_id, ws, "user:alice", role="viewer", added_by="y" * (ACTOR_MAX + 1)
        )
        await uow.commit()
    assert member.role == "viewer" and len(member.added_by) == ACTOR_MAX
    assert await _members_in_store(container, tenant_id, ws, "user:alice") == {"viewer"}
    after_add = await _membership_revision(uow_factory, tenant_id, "alice")
    assert after_add > start

    async with uow_factory() as uow:
        await tenancy.set_member(uow, tenant_id, ws, "user:alice", role="admin", added_by="root")
        await uow.commit()
    # admin implies member and viewer in the model
    assert await _members_in_store(container, tenant_id, ws, "user:alice") == {
        "admin",
        "member",
        "viewer",
    }
    assert await _membership_revision(uow_factory, tenant_id, "alice") > after_add
    async with uow_factory() as uow:
        [stored] = await tenancy.members(uow, tenant_id, ws)
    assert (stored.principal, stored.role, stored.added_by) == ("user:alice", "admin", "root")

    # the admin tuple is deleted with the demotion: no union of the two roles is left behind
    async with uow_factory() as uow:
        await tenancy.set_member(uow, tenant_id, ws, "user:alice", role="viewer", added_by="root")
        await uow.commit()
    assert await _members_in_store(container, tenant_id, ws, "user:alice") == {"viewer"}


async def test_only_a_user_may_administer_a_workspace(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id)
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="only a user"):
            await tenancy.set_member(uow, tenant_id, ws, "agent:bot", role="admin", added_by="r")


@pytest.mark.parametrize("principal", ["alice", "group:eng", "user:", "user:a\x00b"])
async def test_a_malformed_principal_is_refused_on_add_and_remove(
    uow_factory, tenancy, principal
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id)
    async with uow_factory() as uow:
        with pytest.raises(ValidationFailed, match="invalid principal"):
            await tenancy.set_member(uow, tenant_id, ws, principal, role="member", added_by="r")
        with pytest.raises(ValidationFailed, match="invalid principal"):
            await tenancy.remove_member(uow, tenant_id, ws, principal)


async def test_membership_of_an_unknown_workspace_is_not_found(uow_factory, tenancy) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    async with uow_factory() as uow:
        with pytest.raises(NotFound):
            await tenancy.set_member(uow, tenant_id, "nope", "user:a", role="member", added_by="r")
        with pytest.raises(NotFound):
            await tenancy.remove_member(uow, tenant_id, "nope", "user:a")
        with pytest.raises(NotFound):
            await tenancy.members(uow, tenant_id, "nope")


async def test_removing_a_member_drops_every_role_and_bumps_their_revision(
    container, uow_factory, tenancy
) -> None:
    tenant_id = await _onboard(uow_factory, tenancy)
    ws = await _team(uow_factory, tenancy, tenant_id)
    async with uow_factory() as uow:
        await tenancy.set_member(uow, tenant_id, ws, "user:alice", role="member", added_by="r")
        await tenancy.set_member(uow, tenant_id, ws, "user:bob", role="viewer", added_by="r")
        await uow.commit()
    before = await _membership_revision(uow_factory, tenant_id, "alice")
    async with uow_factory() as uow:
        await tenancy.remove_member(uow, tenant_id, ws, "user:alice")
        await uow.commit()
    assert await _members_in_store(container, tenant_id, ws, "user:alice") == set()
    assert await _members_in_store(container, tenant_id, ws, "user:bob") == {"viewer"}
    assert await _membership_revision(uow_factory, tenant_id, "alice") > before
    async with uow_factory() as uow:
        assert [m.principal for m in await tenancy.members(uow, tenant_id, ws)] == ["user:bob"]
