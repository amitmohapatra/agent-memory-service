"""The edges of the platform layer, through the SDK and raw HTTP, against a real database.

a retried key issuance returns the same key without repeating the token
a service key cannot administer; an admin key cannot name another tenant
the platform key onboards and nothing else; no bootstrap secret, no platform
a suspended tenant's keys stop on the next request and resume when the tenant does
identifiers are never reused; deleting a workspace revokes the keys bound to it
deleting a group removes it from every workspace it was admitted to
revocation is idempotent; the audit is paginated and newest first
"""

from __future__ import annotations

import pytest

from memory_service.api.app import create_app
from tests.agent.conftest import BOOTSTRAP, sdk
from tests.conftest import PG_AVAILABLE, _test_overrides
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e
PLATFORM = {"X-API-Key": BOOTSTRAP}


def _mentions(items, needle: str) -> bool:
    return any(needle in (getattr(i, "text", "") or "") for i in items)


def _admin_headers(token: str, **extra: str) -> dict[str, str]:
    return {"X-API-Key": token, **extra}


async def test_a_retried_issuance_replays_the_record_and_never_the_token(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    headers = _admin_headers(acme.admin_key.token, **{"Idempotency-Key": "issue-1"})
    body = {"role": "service", "name": "harness"}
    first = running.post("/v1/keys", headers=headers, json=body)
    assert first.status_code == 201, first.text
    assert first.json()["token"].startswith("mk_")
    again = running.post("/v1/keys", headers=headers, json=body)
    assert again.status_code == 201 and again.headers.get("Idempotent-Replayed") == "true"
    assert again.json()["key_id"] == first.json()["key_id"]
    assert again.json()["token"] is None, "the secret is shown exactly once"
    # a different body under the same key is a conflict, not a second key
    other = running.post("/v1/keys", headers=headers, json={**body, "name": "other"})
    assert other.status_code == 409 and "Idempotency-Key" in other.text, other.text
    keys = running.get("/v1/keys", headers=_admin_headers(acme.admin_key.token)).json()
    assert sorted(k["name"] for k in keys) == ["harness", "initial admin"], "one key, not two"
    # without the header, two keys under one name are two keys, each with its own secret
    admin = sdk(app, acme.admin_key.token)
    one = await admin.tenant.keys.issue("service", "worker")
    two = await admin.tenant.keys.issue("service", "worker")
    assert one.key_id != two.key_id and one.token and two.token and one.token != two.token
    replay = await admin.tenant.keys.issue("service", "worker", idempotency_key="w-1")
    again = await admin.tenant.keys.issue("service", "worker", idempotency_key="w-1")
    assert again.key_id == replay.key_id and again.token is None and replay.token


async def test_roles_and_tenants_are_enforced_on_administration(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    globex = await platform.admin.create_tenant("Globex", tenant_id="globex")
    admin = sdk(app, acme.admin_key.token)
    service = await admin.tenant.keys.issue("service", "harness")

    # a service key reads and writes memory; it does not administer
    with pytest.raises(MemoryError) as denied:
        await sdk(app, service.token).tenant.workspaces.create("Finance")
    assert denied.value.status == 403
    # an admin key naming another tenant is refused, not silently redirected
    r = running.post(
        "/v1/workspaces",
        headers=_admin_headers(acme.admin_key.token, **{"X-Trellis-Tenant": "globex"}),
        json={"name": "Finance"},
    )
    assert r.status_code == 403, r.text
    # the platform key administers any tenant it names, and none it does not
    assert (
        running.post(
            "/v1/workspaces",
            headers={**PLATFORM, "X-Trellis-Tenant": "globex"},
            json={"name": "Ops"},
        ).status_code
        == 201
    )
    assert running.post("/v1/workspaces", headers=PLATFORM, json={"name": "x"}).status_code == 422
    # ...and it never acts as a memory client
    with pytest.raises(MemoryError) as no_memory:
        await platform.bind(tenant_id="globex", user_id="u1").recall("anything")
    assert no_memory.value.status == 403
    # tenants are separate even for the platform's own listing of keys
    listed = running.get("/v1/keys", headers={**PLATFORM, "X-Trellis-Tenant": "acme"}).json()
    assert {k["tenant_id"] for k in listed} == {"acme"} and globex.tenant.tenant_id == "globex"


async def test_suspension_stops_every_key_on_its_next_request(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    service = await sdk(app, acme.admin_key.token).tenant.keys.issue("service", "h")
    harness = sdk(app, service.token).bind(user_id="u1")
    await harness.remember("The budget review is on Friday.")
    assert _mentions(await harness.recall("budget review"), "budget review")  # warms the cache
    await platform.admin.update_tenant("acme", status="suspended")
    with pytest.raises(MemoryError) as stopped:
        await harness.recall("budget review")
    assert stopped.value.status == 403
    with pytest.raises(MemoryError) as no_new_keys:
        await sdk(app, acme.admin_key.token).tenant.keys.issue("service", "again")
    assert no_new_keys.value.status == 403, "the admin key is suspended too"
    await platform.admin.update_tenant("acme", status="active")
    assert _mentions(await harness.recall("budget review"), "budget review")


async def test_identifiers_are_never_reused_and_bound_keys_die_with_their_workspace(
    app, running
) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    with pytest.raises(MemoryError) as dup:
        await admin.tenant.workspaces.create("Finance again", workspace_id="finance")
    assert dup.value.status == 409
    bound = await admin.tenant.keys.issue("service", "finance-bot", workspace_id="finance")
    await sdk(app, bound.token).bind(user_id="u1").remember("Q3 close is on the 5th.")
    await admin.tenant.workspaces.delete("finance")
    with pytest.raises(MemoryError) as gone:
        await sdk(app, bound.token).bind(user_id="u1").recall("Q3 close")
    assert gone.value.status == 401, "a key bound to a deleted workspace is revoked"
    with pytest.raises(MemoryError) as reused:
        await admin.tenant.workspaces.create("Finance v2", workspace_id="finance")
    assert reused.value.status == 409, "a deleted id is not available again"
    assert [w.workspace_id for w in await admin.tenant.workspaces.list()] == []
    with pytest.raises(MemoryError) as missing:
        await admin.tenant.keys.issue("service", "x", workspace_id="finance")
    assert missing.value.status == 404


async def test_deleting_a_group_removes_it_from_every_workspace(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    for ws in ("legal", "finance"):
        await admin.tenant.workspaces.create(ws.title(), workspace_id=ws)
    await admin.tenant.groups.create("Counsel", group_id="counsel")
    for user in ("lawyer1", "lawyer2"):
        await admin.tenant.groups.add_user("counsel", user)
    await admin.tenant.workspaces.set_member("legal", "group:counsel")
    await admin.tenant.workspaces.set_member("finance", "group:counsel", role="viewer")
    service = await admin.tenant.keys.issue("service", "h")
    reader = sdk(app, service.token).bind(user_id="lawyer1", workspace_id="legal")
    await reader.remember("Retainer letters are renewed in March.", visibility="WORKSPACE")
    colleague = sdk(app, service.token).bind(user_id="lawyer2", workspace_id="legal")
    assert _mentions(await colleague.recall("retainer letters"), "Retainer"), "through the group"
    await admin.tenant.groups.delete("counsel")
    assert not _mentions(await colleague.recall("retainer letters"), "Retainer"), "gone with it"
    for ws in ("legal", "finance"):
        members = await admin.tenant.workspaces.members(ws)
        assert all(m.principal != "group:counsel" for m in members), ws
    with pytest.raises(MemoryError) as unknown_group:
        await admin.tenant.workspaces.set_member("legal", "group:counsel")
    assert unknown_group.value.status == 404
    with pytest.raises(MemoryError) as bad_principal:
        await admin.tenant.workspaces.set_member("legal", "robot:x")
    assert bad_principal.value.status == 422


async def test_revocation_is_idempotent_and_the_audit_pages_newest_first(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    service = await admin.tenant.keys.issue("service", "h")
    ctx = sdk(app, service.token).bind(user_id="u1")
    for i in range(3):
        await ctx.recall(f"question {i}")
    reads = await admin.tenant.reads(limit=2)
    assert len(reads) == 2 and reads[0].at >= reads[1].at
    assert {r.credential for r in reads} == {f"key:{service.key_id}"}, "who: the key"
    assert {r.principal for r in reads} == {"user:u1"}, "for whom: the asserted user"
    older = await admin.tenant.reads(before=reads[-1].at)
    assert len(older) == 1 and older[0].at < reads[-1].at, "the next page is the older entry"
    assert await admin.tenant.reads(before=older[-1].at) == [], "and then there is nothing"
    newer = await admin.tenant.reads(after=reads[1].at)
    assert [r.at for r in newer] == [reads[0].at], "after= is a since-filter"
    await admin.tenant.keys.revoke(service.key_id)
    await admin.tenant.keys.revoke(service.key_id)  # already revoked: still 204
    await admin.tenant.keys.revoke("nonexistent-key-id")  # unknown: still 204, leaks nothing
    with pytest.raises(MemoryError) as refused:
        await ctx.recall("question")
    assert refused.value.status == 401


async def test_two_onboardings_of_one_id_at_once_yield_one_tenant_and_one_409(app, running) -> None:
    """Both pass the existence check; the primary key decides and the loser gets the same
    answer a sequential duplicate gets, not a 500."""
    import asyncio

    platform = sdk(app, BOOTSTRAP)
    results = await asyncio.gather(
        *(platform.admin.create_tenant("Acme", tenant_id="acme") for _ in range(3)),
        return_exceptions=True,
    )
    created = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, MemoryError)]
    assert len(created) == 1 and len(refused) == 2, results
    assert {r.status for r in refused} == {409}
    assert len(await platform.admin.tenants()) == 1
    keys = running.get("/v1/keys", headers={**PLATFORM, "X-Trellis-Tenant": "acme"}).json()
    assert len(keys) == 1, "the losers issued no admin key"


async def test_a_key_only_caller_is_metered_by_its_tenant_s_quota(app, running) -> None:
    """The limiter runs before authentication and the SDK sends no tenant header, so the
    tenant's quota has to reach the request through the key it presents. The limit the
    response names is the proof; the 429 itself is the unit suite's (a burst of 200 would
    make this test cross a minute window)."""
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme", rate_limit_per_minute=2)
    globex = await platform.admin.create_tenant("Globex", tenant_id="globex")
    acme_key = await sdk(app, acme.admin_key.token).tenant.keys.issue("service", "h")
    globex_key = await sdk(app, globex.admin_key.token).tenant.keys.issue("service", "h")
    body = {"scope": {"user_id": "u1"}, "query": "anything"}
    default = str(app.state.settings.service.rate_limit_per_minute)

    def limit_for(token: str) -> str:
        r = running.post("/v1/recall", headers={"X-API-Key": token}, json=body)
        assert r.status_code == 200, r.text
        return r.headers["X-RateLimit-Limit"]

    assert limit_for(acme_key.token) == "2", "the tenant's quota, found through the key"
    assert limit_for(globex_key.token) == default, "no override: the service default"
    await platform.admin.update_tenant("acme", clear_rate_limit=True)
    assert limit_for(acme_key.token) == default, "cleared on this instance at once"
    await platform.admin.update_tenant("acme", rate_limit_per_minute=7)
    assert limit_for(acme_key.token) == "7", "existing keys of a newly overriding tenant, at once"


async def test_without_a_bootstrap_secret_nobody_is_the_platform(make_settings) -> None:
    from fastapi.testclient import TestClient

    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings = make_settings(authentication={"mode": "api_key"})
    app = create_app(settings, overrides=_test_overrides(tasks="inline"))
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/admin/tenants", headers={"X-API-Key": BOOTSTRAP}, json={"name": "x"})
        assert r.status_code == 401


async def test_a_tenant_admin_is_not_the_platform_and_cannot_mint_one(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    await platform.admin.create_tenant("Globex", tenant_id="globex")
    h = _admin_headers(acme.admin_key.token)
    assert running.post("/v1/admin/tenants", headers=h, json={"name": "Evil"}).status_code == 403
    assert running.get("/v1/admin/tenants", headers=h).status_code == 403
    r = running.patch("/v1/admin/tenants/globex", headers=h, json={"status": "suspended"})
    assert r.status_code == 403 and (await platform.admin.get_tenant("globex")).status == "active"
    r = running.post("/v1/keys", headers=h, json={"role": "platform", "name": "escalate"})
    assert r.status_code == 422, r.text
    assert all(k["role"] != "platform" for k in running.get("/v1/keys", headers=h).json())
    # administration is tenant-wide, so an admin key cannot be "bound" to a workspace
    await sdk(app, acme.admin_key.token).tenant.workspaces.create("Finance", workspace_id="finance")
    r = running.post(
        "/v1/keys", headers=h, json={"role": "admin", "name": "bound", "workspace_id": "finance"}
    )
    assert r.status_code == 422 and "tenant-wide" in r.text


async def test_a_retried_onboarding_replays_the_tenant_and_never_the_admin_token(
    app, running
) -> None:
    headers = {**PLATFORM, "Idempotency-Key": "onboard-1"}
    body = {"name": "Acme", "tenant_id": "acme"}
    first = running.post("/v1/admin/tenants", headers=headers, json=body)
    assert first.status_code == 201 and first.json()["admin_key"]["token"].startswith("mk_")
    again = running.post("/v1/admin/tenants", headers=headers, json=body)
    assert again.status_code == 201 and again.headers.get("Idempotent-Replayed") == "true"
    assert again.json()["admin_key"]["key_id"] == first.json()["admin_key"]["key_id"]
    assert again.json()["admin_key"]["token"] is None, "the secret is shown exactly once"
    listed = running.get("/v1/keys", headers={**PLATFORM, "X-Trellis-Tenant": "acme"}).json()
    assert len(listed) == 1, "one tenant, one admin key"
    token = first.json()["admin_key"]["token"]
    assert running.get("/v1/keys", headers={"X-API-Key": token}).status_code == 200
    reserved = running.post(
        "/v1/admin/tenants", headers=PLATFORM, json={"name": "x", "tenant_id": "platform"}
    )
    assert reserved.status_code == 422, "the platform's own scope is not a tenant id"


async def test_the_platform_cannot_issue_a_key_for_a_suspended_tenant(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    await platform.admin.create_tenant("Acme", tenant_id="acme")
    await platform.admin.update_tenant("acme", status="suspended")
    r = running.post(
        "/v1/keys",
        headers={**PLATFORM, "X-Trellis-Tenant": "acme"},
        json={"role": "service", "name": "h"},
    )
    assert r.status_code == 409, r.text


async def test_deleting_twice_converges_and_a_workspace_admin_must_be_a_user(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    await admin.tenant.groups.create("Analysts", group_id="analysts")
    with pytest.raises(MemoryError) as not_a_user:
        await admin.tenant.workspaces.set_member("finance", "agent:bot", role="admin")
    assert not_a_user.value.status == 422
    await admin.tenant.workspaces.set_member("finance", "agent:bot", role="member")
    for _ in range(2):  # a retry after a lost 204 is not an error
        await admin.tenant.workspaces.delete("finance")
        await admin.tenant.groups.delete("analysts")
    with pytest.raises(MemoryError) as never:
        await admin.tenant.workspaces.delete("never-existed")
    assert never.value.status == 404


async def test_only_members_write_into_a_team(app, running) -> None:
    """A workspace id is a caller-supplied anchor. Once it names a team, publishing into it
    takes membership: WORKSPACE memory, documents and threads alike. A bare anchor with no
    team behind it keeps its old meaning."""
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    await admin.tenant.workspaces.set_member("finance", "user:u1")
    await admin.tenant.workspaces.set_member("finance", "user:v1", role="viewer")
    harness = sdk(app, (await admin.tenant.keys.issue("service", "h")).token)
    with pytest.raises(MemoryError) as outsider:
        await harness.bind(user_id="u3", workspace_id="finance").remember(
            "Planted.", visibility="WORKSPACE"
        )
    assert outsider.value.status == 403
    with pytest.raises(MemoryError) as viewer:
        await harness.bind(user_id="v1", workspace_id="finance").remember(
            "Planted.", visibility="WORKSPACE"
        )
    assert viewer.value.status == 403, "viewers read; they do not write"
    with pytest.raises(MemoryError) as no_team:
        await harness.bind(user_id="u1", workspace_id="ghost").remember("x", visibility="WORKSPACE")
    assert no_team.value.status == 404
    member = harness.bind(user_id="u1", workspace_id="finance")
    await member.remember("Q3 close is on the 5th.", visibility="WORKSPACE")
    reader = harness.bind(user_id="v1", workspace_id="finance")
    assert not _mentions(await reader.recall("planted"), "Planted")
    assert _mentions(await reader.recall("Q3 close"), "Q3 close")
    # threads: a non-member may not open one inside the team; anyone may under a bare anchor
    with pytest.raises(MemoryError) as thread:
        await harness.bind(user_id="u3", workspace_id="finance").chat.create(title="plan")
    assert thread.value.status == 403
    anchored = harness.bind(user_id="u3", workspace_id="anchor-only")
    assert (await anchored.chat.create(title="ok")).thread_id
    assert (await member.chat.create(title="ours")).thread_id


async def test_retention_forgets_only_live_rows_of_active_tenants(app, running) -> None:
    from sqlalchemy import text

    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme", retention_days=1)
    key = await sdk(app, acme.admin_key.token).tenant.keys.issue("service", "h")
    await sdk(app, key.token).bind(user_id="u1").remember("An old fact from another quarter.")
    container = app.state.container

    async def backdate() -> None:  # the one reach into the store: making the row due
        async with container.database.engine.begin() as conn:
            await conn.execute(text("UPDATE memories SET created_at = now() - interval '3 days'"))

    running.portal.call(backdate)
    retention = container.services["retention"]
    await platform.admin.update_tenant("acme", status="suspended")
    assert await retention.sweep() == 0, "suspended tenants are skipped"
    await platform.admin.update_tenant("acme", status="active")
    assert await retention.sweep() == 1
    assert await retention.sweep() == 0, "forgotten rows are not swept again"
    reader = sdk(app, key.token).bind(user_id="u1")
    assert not _mentions(await reader.recall("old fact"), "old fact")


async def test_every_path_that_mints_a_workspace_audience_is_gated(app, running) -> None:
    """Observations were gated first; a message hint and a tool record mint the same
    audience and must answer the same 403."""
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    await admin.tenant.workspaces.set_member("finance", "user:u1")
    token = (await admin.tenant.keys.issue("service", "h")).token
    harness = sdk(app, token)
    thread = await harness.bind(user_id="u3").chat.create(title="mine")  # own thread, no team
    planted = running.post(
        "/v1/messages",
        headers={"X-API-Key": token, "X-Trellis-User": "u3", "X-Trellis-Workspace": "finance"},
        json={
            "scope": {"thread_id": thread.thread_id, "session_id": "ses_1", "turn_id": "trn_1"},
            "role": "USER",
            "content": "Planted through a message.",
            "hints": {"visibility": "WORKSPACE"},
        },
    )
    assert planted.status_code == 403, planted.text
    tool = running.post(
        "/v1/tools/invocations",
        headers={"X-API-Key": token, "X-Trellis-User": "u3", "X-Trellis-Workspace": "finance"},
        json={
            "scope": {"agent_id": "bot", "agent_run_id": "run_1"},
            "tool": "search",
            "args": {"q": "planted"},
            "output_summary": "Planted through a tool record.",
            "visibility": "WORKSPACE",
        },
    )
    assert tool.status_code == 403, tool.text
    member = harness.bind(user_id="u1", workspace_id="finance")
    ok = running.post(
        "/v1/messages",
        headers={"X-API-Key": token, "X-Trellis-User": "u1", "X-Trellis-Workspace": "finance"},
        json={
            "scope": {
                "thread_id": (await member.chat.create(title="ours")).thread_id,
                "session_id": "s",
                "turn_id": "t",
            },
            "role": "USER",
            "content": "Budget review moved to Thursday.",
            "hints": {"visibility": "WORKSPACE"},
        },
    )
    assert ok.status_code in (200, 201, 202), ok.text
    assert not _mentions(await member.recall("planted"), "Planted")


async def test_only_members_ingest_documents_into_a_team(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    await admin.tenant.workspaces.set_member("finance", "user:u1")
    await admin.tenant.workspaces.set_member("finance", "user:v1", role="viewer")
    harness = sdk(app, (await admin.tenant.keys.issue("service", "h")).token)
    doc = ("plan.md", b"# Planted\n", "text/markdown")
    for user in ("u3", "v1"):
        with pytest.raises(MemoryError) as refused:
            await harness.bind(user_id=user, workspace_id="finance").documents.add(doc)
        assert refused.value.status == 403, user
    assert (await harness.bind(user_id="u1", workspace_id="finance").documents.add(doc)).document_id
    assert (
        await harness.bind(user_id="u3", workspace_id="anchor-only").documents.add(doc)
    ).document_id


async def test_an_admin_cannot_revoke_another_tenant_s_key(app, running) -> None:
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    globex = await platform.admin.create_tenant("Globex", tenant_id="globex")
    gkey = await sdk(app, globex.admin_key.token).tenant.keys.issue("service", "h")
    reader = sdk(app, gkey.token).bind(user_id="u1")
    await reader.recall("warm the verifier cache")
    await sdk(app, acme.admin_key.token).tenant.keys.revoke(gkey.key_id)  # 204, and a no-op
    await reader.recall("still served")
    listed = await sdk(app, globex.admin_key.token).tenant.keys.list()
    assert next(k for k in listed if k.key_id == gkey.key_id).revoked_at is None


async def test_the_platform_cannot_administer_a_tenant_nobody_onboarded(app, running) -> None:
    r = running.post(
        "/v1/workspaces", headers={**PLATFORM, "X-Trellis-Tenant": "ghost"}, json={"name": "x"}
    )
    assert r.status_code == 404, r.text
    assert (
        running.get("/v1/keys", headers={**PLATFORM, "X-Trellis-Tenant": "ghost"}).status_code
        == 404
    )


async def test_the_registry_reloads_suspensions_and_quotas_from_the_store(app, running) -> None:
    """What a restarted or sibling instance sees: not what this one observed, what is stored."""
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme", rate_limit_per_minute=3)
    key = await sdk(app, acme.admin_key.token).tenant.keys.issue("service", "h")
    await platform.admin.update_tenant("acme", status="suspended")
    registry = app.state.container.services["tenant_registry"]
    registry._limits, registry._key_tenants, registry._suspended = {}, {}, set()  # noqa: SLF001 - a cold instance
    assert await registry.refresh() is True
    assert registry.is_suspended("acme")
    assert registry.quota_for(None, key.token) == ("acme", 3)
    r = running.post(
        "/v1/recall",
        headers={"X-API-Key": "dev-key", "X-Trellis-Tenant": "acme"},
        json={"scope": {"user_id": "u1"}, "query": "anything"},
    )
    assert r.status_code in (401, 403), "a suspended tenant is refused whatever the credential"


async def test_the_read_audit_is_purged_past_its_retention(app, running) -> None:
    from sqlalchemy import text

    from memory_service.config.constants import TASKS

    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    key = await admin.tenant.keys.issue("service", "h")
    ctx = sdk(app, key.token).bind(user_id="u1")
    for i in range(3):
        await ctx.recall(f"question {i}")
    assert len(await admin.tenant.reads()) == 3
    container = app.state.container

    async def backdate() -> None:
        async with container.database.engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE memory_reads SET at = now() - interval '500 days' WHERE id IN "
                    "(SELECT id FROM memory_reads ORDER BY id LIMIT 2)"
                )
            )

    running.portal.call(backdate)
    audit = container.services["read_audit"]
    assert await audit.purge(older_than_days=TASKS.read_audit_retention_days, limit=1) == 2
    assert len(await admin.tenant.reads()) == 1
    assert await audit.purge(older_than_days=TASKS.read_audit_retention_days) == 0


async def test_a_workspace_upload_is_gated_not_500(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    await admin.tenant.workspaces.set_member("finance", "user:u1")
    token = (await admin.tenant.keys.issue("service", "h")).token

    def upload(user: str, workspace: str) -> int:
        # distinct bytes each time: identical content by the same user is deduplicated
        # before any audience is minted, and that early return is not what is under test
        body = f"# plan for {user} in {workspace}\n".encode()
        r = running.post(
            "/v1/documents",
            headers={"X-API-Key": token, "X-Trellis-User": user, "X-Trellis-Workspace": workspace},
            files={"file": ("plan.md", body, "text/markdown")},
            data={"visibility": "WORKSPACE"},
        )
        return r.status_code

    assert upload("u3", "finance") == 403
    assert upload("u1", "finance") in (200, 201, 202)
    assert upload("u1", "ghost") == 404
    r = running.post(
        "/v1/documents",
        headers={"X-API-Key": token, "X-Trellis-User": "u1"},
        files={"file": ("plan.md", b"# x\n", "text/markdown")},
        data={"visibility": "WORKSPACE"},
    )
    assert r.status_code == 422, "no anchor at all is a request error, not a crash"


async def test_a_workspace_tool_record_without_an_anchor_is_422_not_500(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    token = (await sdk(app, acme.admin_key.token).tenant.keys.issue("service", "h")).token
    r = running.post(
        "/v1/tools/invocations",
        headers={"X-API-Key": token, "X-Trellis-User": "u1"},
        json={
            "scope": {"agent_id": "bot", "agent_run_id": "run_1"},
            "tool": "search",
            "args": {"q": "x"},
            "output_summary": "x",
            "visibility": "WORKSPACE",
        },
    )
    assert r.status_code == 422, r.text


async def test_odd_identifiers_and_control_characters_are_request_errors(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    h = _admin_headers(acme.admin_key.token)
    assert running.get("/v1/workspaces/not%20valid", headers=h).status_code == 422
    assert running.delete("/v1/keys/a%00b", headers=h).status_code == 422
    assert running.delete("/v1/groups/x%20y", headers=h).status_code == 422
    created = running.post(
        "/v1/workspaces", headers=h, json={"name": "Fin\u0000ance", "workspace_id": "fin"}
    )
    assert created.status_code == 201 and created.json()["name"] == "Finance", "sanitised, stored"
    assert (
        running.get("/v1/admin/tenants", headers={**PLATFORM}, params={"after": "a b"}).status_code
        == 422
    )


async def test_a_document_keeps_the_audience_its_uploader_chose_inside_a_team(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    for user in ("u1", "u2"):
        await admin.tenant.workspaces.set_member("finance", f"user:{user}")
    token = (await admin.tenant.keys.issue("service", "h")).token

    def upload(user: str, visibility: str) -> str:
        r = running.post(
            "/v1/documents",
            headers={"X-API-Key": token, "X-Trellis-User": user, "X-Trellis-Workspace": "finance"},
            files={
                "file": ("notes.md", f"# {visibility} notes of {user}\n".encode(), "text/markdown")
            },
            data={"visibility": visibility},
        )
        assert r.status_code in (200, 201, 202), r.text
        return r.json()["document_id"]

    private = upload("u1", "PRIVATE")
    shared = upload("u1", "WORKSPACE")
    peer = {"X-API-Key": token, "X-Trellis-User": "u2", "X-Trellis-Workspace": "finance"}
    assert running.get(f"/v1/documents/{shared}", headers=peer).status_code == 200
    assert running.get(f"/v1/documents/{private}", headers=peer).status_code in (403, 404), (
        "a private upload inside the team is not the team's"
    )


async def test_an_id_already_used_as_an_anchor_cannot_become_a_team(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    harness = sdk(app, (await admin.tenant.keys.issue("service", "h")).token)
    await harness.bind(user_id="u1", workspace_id="proj-x").chat.create(title="before teams")
    with pytest.raises(MemoryError) as taken:
        await admin.tenant.workspaces.create("Project X", workspace_id="proj-x")
    assert taken.value.status == 409 and "anchor" in str(taken.value)
    assert (
        await admin.tenant.workspaces.create("Project X", workspace_id="proj-x-team")
    ).workspace_id


async def test_listing_memories_inside_a_team_includes_the_team_s(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    await admin.tenant.workspaces.create("Finance", workspace_id="finance")
    for user in ("u1", "u2"):
        await admin.tenant.workspaces.set_member("finance", f"user:{user}")
    harness = sdk(app, (await admin.tenant.keys.issue("service", "h")).token)
    await harness.bind(user_id="u1", workspace_id="finance").remember(
        "The close calendar is published on the first Monday.", visibility="WORKSPACE"
    )
    listed = await harness.bind(user_id="u2", workspace_id="finance").memories()
    assert any("close calendar" in m.content for m in listed), "a member lists the team's memory"
    outside = await harness.bind(user_id="u3", workspace_id="finance").memories()
    assert not any("close calendar" in m.content for m in outside)


async def test_a_name_that_is_only_control_characters_is_refused(app, running) -> None:
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")
    h = _admin_headers(acme.admin_key.token)
    assert running.post("/v1/workspaces", headers=h, json={"name": "\u007f"}).status_code == 422
    assert (
        running.post("/v1/keys", headers=h, json={"role": "service", "name": "\u0000"}).status_code
        == 422
    )
    r = running.patch("/v1/admin/tenants/acme", headers=PLATFORM, json={"name": "\u0001\u0002"})
    assert (
        r.status_code == 422 and (await sdk(app, BOOTSTRAP).admin.get_tenant("acme")).name == "Acme"
    )


async def test_a_tenant_holds_a_bounded_number_of_live_keys(app, running, monkeypatch) -> None:
    from memory_service.modules.tenancy import service as tenancy_service

    monkeypatch.setattr(tenancy_service, "MAX_KEYS_PER_TENANT", 3)
    acme = await sdk(app, BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")  # 1 live key
    admin = sdk(app, acme.admin_key.token)
    a = await admin.tenant.keys.issue("service", "a")
    await admin.tenant.keys.issue("service", "b")
    with pytest.raises(MemoryError) as full:
        await admin.tenant.keys.issue("service", "c")
    assert full.value.status == 409 and "live keys" in str(full.value)
    await admin.tenant.keys.revoke(a.key_id)
    assert (await admin.tenant.keys.issue("service", "c")).token, "room again after a revocation"
