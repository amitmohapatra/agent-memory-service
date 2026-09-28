"""SDK 0.2.1: feedback, webhooks, team model keys and cursor paging map to the wire."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from trellis.memory import MemoryClient, Page
from trellis.memory.models import Feedback, MemoryResult

BASE = "http://memory.test"


@pytest.fixture
def client() -> MemoryClient:
    return MemoryClient(BASE, api_key="k", max_retries=0)


def _feedback(**fields) -> dict:
    return {
        "feedback_id": "fb_1",
        "tenant_id": "acme",
        "workspace_id": "fin",
        "user_id": "u1",
        "agent_id": None,
        "agent_run_id": None,
        "trace_id": None,
        "target_kind": "memory",
        "target_id": "mem_1",
        "verdict": "correct",
        "correction": "March, not May",
        "score": None,
        "comment": None,
        "reviewer": "u1",
        "source": "human",
        "evidence_refs": [],
        "metadata": {},
        "created_at": "2026-09-28T07:00:00Z",
        "projection": None,
        **fields,
    }


@respx.mock
async def test_feedback_submit_sends_the_contracts_record_bound_to_the_context(
    client: MemoryClient,
) -> None:
    route = respx.post(f"{BASE}/v1/feedback").respond(201, json=_feedback())
    async with client.bind(
        tenant_id="acme", user_id="u1", workspace_id="fin", agent_id="ref"
    ) as ctx:
        record = await ctx.feedback.submit(
            "memory",
            "mem_1",
            "correct",
            correction="March, not May",
            reviewer="u1",
            feedback_id="fb_1",
        )
    assert isinstance(record, Feedback) and record.verdict == "correct"
    sent = json.loads(route.calls.last.request.content)
    assert sent["target_kind"] == "memory" and sent["feedback_id"] == "fb_1"
    assert sent["agent_id"] == "ref" and sent["tenant_id"] == "acme" and "score" not in sent
    assert set(sent) <= {
        "feedback_id",
        "tenant_id",
        "workspace_id",
        "user_id",
        "agent_id",
        "agent_run_id",
        "target_kind",
        "target_id",
        "verdict",
        "correction",
        "score",
        "comment",
        "reviewer",
        "source",
        "evidence_refs",
        "metadata",
    }  # nothing else of the scope (thread, session, custom_metadata) travels in the body
    assert route.calls.last.request.headers["X-Trellis-Tenant"] == "acme"
    assert route.calls.last.request.headers["X-Trellis-User"] == "u1"


@respx.mock
async def test_feedback_pages_follow_next_cursor(client: MemoryClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["target_kind"] == "run" and request.url.params["limit"] == "1"
        if request.url.params.get("cursor") is None:
            record = _feedback(
                feedback_id="fb_a",
                target_kind="run",
                target_id="r",
                verdict="confirm",
                correction=None,
            )
            return httpx.Response(200, json={"feedback": [record], "next_cursor": "c1"})
        assert request.url.params["cursor"] == "c1"
        return httpx.Response(200, json={"feedback": [], "next_cursor": None})

    listing = respx.get(f"{BASE}/v1/feedback").mock(side_effect=handler)
    async with client.bind(tenant_id="acme", user_id="u1") as ctx:
        page = await ctx.feedback.page_for("run", "r", limit=1)
        assert isinstance(page, Page) and page.has_more and page.next_cursor == "c1"
        assert [f.feedback_id for f in page.items] == ["fb_a"]
        last = await ctx.feedback.page_for("run", "r", limit=1, cursor="c1")
        assert not last.has_more and last.items == []
        assert [f.feedback_id for f in await ctx.feedback.list_for("run", "r", limit=1)] == ["fb_a"]
        assert listing.call_count == 3
        one = respx.get(f"{BASE}/v1/feedback/fb_a").respond(200, json=_feedback(feedback_id="fb_a"))
        assert (await ctx.feedback.get("fb_a")).feedback_id == "fb_a" and one.called


@respx.mock
async def test_memories_iterate_every_page(client: MemoryClient) -> None:
    def memory(i: int) -> dict:
        return {
            "memory_id": f"mem_{i}",
            "content": f"m{i}",
            "memory_type": "SEMANTIC",
            "lifetime": "LONG_TERM",
            "visibility": "USER",
        }

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(
                200, json={"memories": [memory(1), memory(2)], "next_cursor": "p2"}
            )
        assert cursor == "p2"
        return httpx.Response(200, json={"memories": [memory(3)], "next_cursor": None})

    respx.get(f"{BASE}/v1/memories").mock(side_effect=handler)
    async with client.bind(tenant_id="acme", user_id="u1") as ctx:
        page = await ctx.memories_page(limit=2)
        assert [m.memory_id for m in page.items] == ["mem_1", "mem_2"] and page.next_cursor == "p2"
        assert [m.memory_id async for m in ctx.iter_memories(page_size=2)] == [
            "mem_1",
            "mem_2",
            "mem_3",
        ]
        assert all(isinstance(m, MemoryResult) for m in page.items)
        assert [m.memory_id for m in await ctx.memories(limit=2)] == ["mem_1", "mem_2"]


@respx.mock
async def test_bare_list_routes_page_through_the_link_header(client: MemoryClient) -> None:
    key = {
        "key_id": "k1",
        "tenant_id": "acme",
        "role": "service",
        "name": "n",
        "workspace_id": None,
        "created_by": "svc",
        "created_at": "2026-09-28T07:00:00Z",
        "expires_at": None,
        "revoked_at": None,
        "last_used_at": None,
    }
    respx.get(f"{BASE}/v1/keys").respond(
        200, json=[key], headers={"Link": f'<{BASE}/v1/keys?limit=1&cursor=next1>; rel="next"'}
    )
    admin = client.administer("acme")
    page = await admin.keys.page(limit=1)
    assert [k.key_id for k in page.items] == ["k1"] and page.next_cursor == "next1"
    assert (await admin.keys.list(limit=1))[0].key_id == "k1"
    respx.get(f"{BASE}/v1/workspaces").respond(
        200,
        json=[
            {
                "workspace_id": "fin",
                "tenant_id": "acme",
                "name": "Finance",
                "created_at": "2026-09-28T07:00:00Z",
            }
        ],
    )
    workspaces = await admin.workspaces.page()
    assert workspaces.items[0].workspace_id == "fin" and workspaces.next_cursor is None
    respx.get(f"{BASE}/v1/groups").respond(
        200,
        json=[
            {
                "group_id": "g1",
                "tenant_id": "acme",
                "name": "Analysts",
                "created_at": "2026-09-28T07:00:00Z",
            }
        ],
        headers={"Link": f'<{BASE}/v1/groups?cursor=g-next>; rel="next"'},
    )
    groups = await admin.groups.page()
    assert groups.items[0].group_id == "g1" and groups.next_cursor == "g-next"
    briefs = respx.get(f"{BASE}/v1/briefs").respond(
        200,
        json=[{"brief_id": "brf_1", "spec": {"title": "Changes", "question": "What changed?"}}],
        headers={"Link": f'<{BASE}/v1/briefs?limit=1&cursor=b-next>; rel="next"'},
    )
    async with client.bind(tenant_id="acme", user_id="u1") as ctx:
        page = await ctx.briefs.page(limit=1, cursor="b-prev")
        assert page.items[0].brief_id == "brf_1" and page.next_cursor == "b-next"
        assert briefs.calls.last.request.url.params["cursor"] == "b-prev"
    tenants = respx.get(f"{BASE}/v1/admin/tenants").respond(
        200,
        json=[
            {
                "tenant_id": "acme",
                "name": "Acme",
                "status": "active",
                "created_at": "2026-09-28T07:00:00Z",
                "updated_at": "2026-09-28T07:00:00Z",
            }
        ],
    )
    listed = await client.admin.tenants_page(cursor="t-prev")
    assert listed.items[0].tenant_id == "acme" and listed.next_cursor is None
    assert tenants.calls.last.request.url.params["cursor"] == "t-prev"
    reads = respx.get(f"{BASE}/v1/reads").respond(200, json=[])
    assert (await admin.reads_page(cursor="r-prev")).items == []
    assert reads.calls.last.request.url.params["cursor"] == "r-prev"


@respx.mock
async def test_webhooks_and_team_model_keys_are_tenant_administration(client: MemoryClient) -> None:
    created = {
        "subscription_id": "whk_1",
        "tenant_id": "acme",
        "workspace_id": None,
        "url": "https://hooks.example/x",
        "events": ["memory.created"],
        "description": None,
        "enabled": True,
        "failures": 0,
        "created_by": "svc",
        "created_at": "2026-09-28T07:00:00Z",
        "updated_at": "2026-09-28T07:00:00Z",
        "secret": "s" * 64,
    }
    create = respx.post(f"{BASE}/v1/webhooks").respond(201, json=created)
    listing = respx.get(f"{BASE}/v1/webhooks").respond(
        200, json={"webhooks": [{**created, "secret": None}], "next_cursor": None}
    )
    patch = respx.patch(f"{BASE}/v1/webhooks/whk_1").respond(
        200, json={**created, "enabled": False, "secret": None}
    )
    deliveries = respx.get(f"{BASE}/v1/webhooks/whk_1/deliveries").respond(
        200,
        json={
            "deliveries": [
                {
                    "delivery_id": "wdl_1",
                    "subscription_id": "whk_1",
                    "event_id": "evt_1",
                    "event_type": "memory.created",
                    "status": "DELIVERED",
                    "attempts": 1,
                    "status_code": 200,
                    "last_error": None,
                    "created_at": "2026-09-28T07:00:00Z",
                    "delivered_at": "2026-09-28T07:00:01Z",
                }
            ],
            "next_cursor": None,
        },
    )
    test = respx.post(f"{BASE}/v1/webhooks/whk_1/test").respond(
        202,
        json={**json.loads(deliveries.return_value.content)["deliveries"][0], "status": "PENDING"},
    )
    delete = respx.delete(f"{BASE}/v1/webhooks/whk_1").respond(204)
    admin = client.administer("acme")
    hook = await admin.webhooks.create(
        "https://hooks.example/x", ["memory.created"], idempotency_key="idem-1"
    )
    assert hook.secret == "s" * 64 and hook.subscription_id == "whk_1"
    sent = json.loads(create.calls.last.request.content)
    assert sent == {"url": "https://hooks.example/x", "events": ["memory.created"]}
    assert create.calls.last.request.headers["Idempotency-Key"] == "idem-1"
    assert create.calls.last.request.headers["X-Trellis-Tenant"] == "acme"
    assert (await admin.webhooks.list())[0].enabled is True
    assert (await admin.webhooks.page()).next_cursor is None
    paused = await admin.webhooks.update("whk_1", enabled=False)
    assert paused.enabled is False and json.loads(patch.calls.last.request.content) == {
        "enabled": False
    }
    assert (await admin.webhooks.deliveries("whk_1"))[0].status == "DELIVERED"
    assert (await admin.webhooks.deliveries_page("whk_1")).items[0].delivery_id == "wdl_1"
    assert (await admin.webhooks.test("whk_1")).status == "PENDING"
    await admin.webhooks.delete("whk_1")
    assert listing.called and test.called and delete.called

    status = {"registered": True, "revoked": False, "revision": 1, "updated_at": None}
    ws_put = respx.put(f"{BASE}/v1/workspaces/fin/model-key").respond(200, json=status)
    tenant_put = respx.put(f"{BASE}/v1/model-key").respond(200, json=status)
    tenant_get = respx.get(f"{BASE}/v1/model-key").respond(200, json=status)
    ws_delete = respx.delete(f"{BASE}/v1/workspaces/fin/model-key").respond(
        200, json={**status, "revoked": True}
    )
    assert (await admin.workspaces.set_model_key("fin", "vk-team")).registered
    assert json.loads(ws_put.calls.last.request.content) == {"virtual_key": "vk-team"}
    assert (await admin.set_model_key("vk-tenant")).revision == 1
    assert (await admin.model_key_status()).registered and tenant_get.called and tenant_put.called
    assert (await admin.workspaces.revoke_model_key("fin")).revoked and ws_delete.called
