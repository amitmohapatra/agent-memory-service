"""The list routes page one way and the polled reads validate (ADR 0030): ``cursor`` +
``limit`` with ``Link: rel="next"`` on every list, and ``ETag`` / ``If-None-Match`` on the
agent tool set and the tool catalog."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}


def _walk(client, path: str, key: str, headers: dict, **params) -> list:  # type: ignore[no-untyped-def]
    """Every item of a paged list, following ``Link: rel="next"``; asserts the envelope's
    ``next_cursor`` agrees with the header when the body is an envelope."""
    items: list = []
    r = client.get(path, headers=headers, params=params)
    for _ in range(50):
        assert r.status_code == 200, r.text
        body = r.json()
        items += body if isinstance(body, list) else body[key]
        nxt = r.links.get("next")
        if not isinstance(body, list):
            assert (body["next_cursor"] is None) == (nxt is None)
        if nxt is None:
            return items
        r = client.get(nxt["url"], headers=headers)
    raise AssertionError("the walk did not end")


def _admin(client) -> dict[str, str]:  # type: ignore[no-untyped-def]
    client.post("/v1/admin/tenants", headers=H, json={"tenant_id": "pager", "name": "Pager"})
    return {"X-API-Key": "test-key", "X-Trellis-Tenant": "pager"}


def test_workspace_members_page_by_principal(client) -> None:
    admin = _admin(client)
    client.post("/v1/workspaces", headers=admin, json={"workspace_id": "ops", "name": "Ops"})
    for user in ("u3", "u1", "u2"):
        client.put(f"/v1/workspaces/ops/members/user:{user}", headers=admin, json={})
    walked = _walk(client, "/v1/workspaces/ops/members", "", admin, limit=2)
    assert [m["principal"] for m in walked] == ["user:u1", "user:u2", "user:u3"]


def test_the_admin_tenant_list_takes_a_cursor_only(client) -> None:
    _admin(client)
    walked = _walk(client, "/v1/admin/tenants", "", H, limit=1)
    assert "pager" in [t["tenant_id"] for t in walked]
    assert client.get("/v1/admin/tenants", headers=H, params={"cursor": "%%"}).status_code == 422


def test_the_read_audit_filters_with_since(client) -> None:
    admin = _admin(client)
    assert client.get(
        "/v1/reads", headers=admin, params={"since": "2026-01-01T00:00:00"}
    ).is_success
    future = client.get("/v1/reads", headers=admin, params={"since": "2999-01-01T00:00:00Z"})
    assert future.status_code == 200 and future.json() == []


def test_the_catalog_pages_by_name_and_answers_304_when_unchanged(client) -> None:
    tools = [{"name": f"erp-{n}", "side_effects": "read"} for n in ("c", "a", "b")]
    assert client.put("/v1/tools/catalog", headers=H, json={"tools": tools}).status_code == 200
    walked = _walk(client, "/v1/tools", "tools", H, limit=2)
    assert [t["name"] for t in walked] == ["erp-a", "erp-b", "erp-c"]

    first = client.get("/v1/tools", headers=H)
    etag = first.headers["ETag"]
    assert etag.startswith('W/"') and first.headers["Cache-Control"] == "private, no-cache"
    unchanged = client.get("/v1/tools", headers={**H, "If-None-Match": etag})
    assert unchanged.status_code == 304 and unchanged.content == b""
    assert unchanged.headers["ETag"] == etag
    # an administrator raising a tool's tier changes the answer, and so its tag
    raised = [{"name": "erp-a", "side_effects": "irreversible"}]
    client.put("/v1/tools/catalog", headers=H, json={"tools": raised})
    changed = client.get("/v1/tools", headers={**H, "If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["ETag"] != etag
    assert changed.json()["tools"][0]["risk"] == "irreversible"


def test_the_agent_tool_set_is_cached_and_validated(client) -> None:
    first = client.get("/v1/agent-tools", headers=H)
    assert first.status_code == 200 and first.headers["Cache-Control"] == "private, max-age=300"
    again = client.get("/v1/agent-tools", headers={**H, "If-None-Match": first.headers["ETag"]})
    assert again.status_code == 304 and again.content == b""
    other = client.get("/v1/agent-tools", headers={**H, "If-None-Match": 'W/"stale"'})
    assert other.status_code == 200 and other.json() == first.json()


def test_approval_suggestions_page_most_supported_first(client, container) -> None:
    async def decide() -> None:
        async with container.services["uow_factory"]() as uow:
            for shape, times in (("amount:num:1e2", 9), ("amount:num:1e3", 7), ("id:str", 5)):
                for _ in range(times):
                    await uow.tools.count_approval("acme", "bot", "erp-pay", shape, "approvals")
            await uow.commit()

    client.portal.call(decide)
    scoped = {"agent_id": "bot"}
    walked = _walk(client, "/v1/tools/approval-suggestions", "suggestions", H, limit=1, **scoped)
    assert [s["arg_shape"] for s in walked] == ["amount:num:1e2", "amount:num:1e3", "id:str"]


def test_graph_entities_page_through_the_ranking(client, container) -> None:
    from memory_service.ports.intelligence import Entity

    async def seed() -> None:
        graph = container.services["graph"]
        visibility = ["tenant:acme"]  # every principal of the tenant reads it
        await graph.store.upsert_entities(
            [
                Entity(
                    entity_id=f"ent_{n}",
                    tenant_id="acme",
                    name=f"Acme {n}",
                    canonical_name=f"acme {n}",
                    entity_type="ORG",
                    visibility_keys=visibility,
                )
                for n in ("one", "two", "three")
            ]
        )

    client.portal.call(seed)
    walked = _walk(client, "/v1/graph/entities", "entities", H, q="acme", limit=2)
    assert sorted(e["entity_id"] for e in walked) == ["ent_one", "ent_three", "ent_two"]


def test_the_review_queue_is_a_filter_of_the_feedback_list(client) -> None:
    """The queue's content is the agent suite's (tests/agent/test_api_knowledge.py: a service
    key's votes wait, the admin key reviews); here, the route and its deprecated alias."""
    admin = _admin(client)
    queue = client.get("/v1/feedback", headers=admin, params={"review": "pending"})
    assert queue.status_code == 200 and queue.json() == {"feedback": [], "next_cursor": None}
    alias = client.get("/v1/feedback/pending", headers=admin)
    assert alias.json() == queue.json()
    assert alias.headers["Deprecation"] == "true"
    assert 'rel="successor-version"' in alias.headers["Link"]
    assert "review=pending" in alias.headers["Link"]
    both = {"review": "pending", "target_kind": "memory", "target_id": "mem_1"}
    assert client.get("/v1/feedback", headers=admin, params=both).status_code == 422
    assert client.get("/v1/feedback", headers=H).status_code == 422
