"""Every write honours the Idempotency-Key the OpenAPI document advertises on it, and every
create says where its result lives (ADR 0030).

A retry is the first response again - status, body and ``Location`` - with
``Idempotent-Replayed: true``: a forgotten memory or a deleted thread is not a 404 the
second time, an edited profile block not a 409, a revoked key not a second revocation.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}
REPLAYED = "Idempotent-Replayed"


def _keyed(key: str) -> dict[str, str]:
    return {**H, "Idempotency-Key": key}


def _twice(call, key: str):  # type: ignore[no-untyped-def]
    first, second = call(_keyed(key)), call(_keyed(key))
    assert first.status_code < 300, first.text
    assert (second.status_code, second.content) == (first.status_code, first.content)
    assert REPLAYED not in first.headers and second.headers[REPLAYED] == "true"
    return first, second


def test_a_memory_is_created_with_its_location_and_forgotten_once(client) -> None:
    created, replay = _twice(
        lambda h: client.post(
            "/v1/memories", headers=h, json={"content": "Prefers metric units.", "scope": {}}
        ),
        "remember-1",
    )
    memory_id = created.json()["memory_id"]
    assert created.status_code == 201
    assert created.headers["Location"] == replay.headers["Location"] == f"/v1/memories/{memory_id}"
    assert client.get(created.headers["Location"], headers=H).status_code == 200

    gone, again = _twice(
        lambda h: client.delete(f"/v1/memories/{memory_id}", headers=h), "forget-1"
    )
    assert gone.status_code == again.status_code == 204 and again.content == b""


def test_a_thread_is_patched_and_deleted_once(client) -> None:
    _twice(
        lambda h: client.patch(
            "/v1/threads/thr_idem", headers=h, json={"scope": {}, "title": "Quarterly close"}
        ),
        "thread-patch-1",
    )
    deleted, _ = _twice(lambda h: client.delete("/v1/threads/thr_idem", headers=h), "thread-del")
    assert deleted.status_code == 204


def test_an_accepted_message_points_at_its_job(client) -> None:
    accepted = client.post(
        "/v1/messages",
        headers=H,
        json={
            "scope": {"thread_id": "thr_jobs"},
            "messages": [{"role": "USER", "content": "We moved the close to Friday."}],
        },
    )
    assert accepted.status_code == 202
    job = accepted.json()["messages"][0]["job_ids"][0]
    assert accepted.headers["Location"] == f"/v1/jobs/{job}"
    assert client.get(accepted.headers["Location"], headers=H).status_code == 200


def test_a_profile_edit_retried_is_not_a_conflict(client) -> None:
    client.patch("/v1/profile/user", headers=H, json={"new": "address: Hauptstr. 1"})
    edit = {"old": "Hauptstr. 1", "new": "Ringstr. 9"}
    first, _ = _twice(
        lambda h: client.patch("/v1/profile/user", headers=h, json=edit), "profile-edit-1"
    )
    assert first.json()["text"] == "address: Ringstr. 9"
    # the same edit without the key: its old text is gone, which is the documented 409
    assert client.patch("/v1/profile/user", headers=H, json=edit).status_code == 409


def test_an_agent_tool_write_retried_is_one_memory(client) -> None:
    body = {"scope": {"agent_id": "bot"}, "args": {"content": "Ships on Tuesdays."}}
    first, _ = _twice(
        lambda h: client.post("/v1/agent-tools/memory_remember", headers=h, json=body),
        "agent-remember-1",
    )
    listed = client.get("/v1/memories", headers=H, params={"agent_id": "bot"}).json()
    assert [m["content"] for m in listed["memories"]].count("Ships on Tuesdays.") == 1
    assert first.json()["result"]


def test_feedback_and_the_catalog_honour_the_key(client) -> None:
    memory = client.post("/v1/memories", headers=H, json={"content": "Renewal in May."}).json()
    verdict = {"target_kind": "memory", "target_id": memory["memory_id"], "verdict": "reject"}
    created, _ = _twice(lambda h: client.post("/v1/feedback", headers=h, json=verdict), "fb-1")
    assert created.status_code == 201
    assert created.headers["Location"] == f"/v1/feedback/{created.json()['feedback_id']}"

    catalog = {"tools": [{"name": "erp-get_stock", "side_effects": "read"}]}
    _twice(lambda h: client.put("/v1/tools/catalog", headers=h, json=catalog), "catalog-1")
    record = {
        "tool": "erp-get_stock",
        "args": {"sku": "A4"},
        "scope": {"agent_id": "bot", "agent_run_id": "run_1"},
    }
    recorded, _ = _twice(
        lambda h: client.post("/v1/tools/invocations", headers=h, json=record), "record-1"
    )
    assert recorded.status_code == 202 and "Location" not in recorded.headers


def test_tenancy_writes_are_located_and_replayed(client) -> None:
    onboarded = client.post(
        "/v1/admin/tenants", headers=H, json={"tenant_id": "idem-co", "name": "Idem Co"}
    )
    assert onboarded.status_code == 201
    assert onboarded.headers["Location"] == "/v1/admin/tenants/idem-co"
    admin = {"X-API-Key": "test-key", "X-Trellis-Tenant": "idem-co"}
    _twice(
        lambda h: client.patch(
            "/v1/admin/tenants/idem-co", headers={**h, **admin}, json={"name": "Idem Company"}
        ),
        "tenant-patch-1",
    )

    workspace = client.post(
        "/v1/workspaces", headers=admin, json={"workspace_id": "finance", "name": "Finance"}
    )
    assert workspace.headers["Location"] == "/v1/workspaces/finance"
    member = "/v1/workspaces/finance/members/user:u9"
    _twice(lambda h: client.put(member, headers={**admin, **_key(h)}, json={}), "member-put")
    removed, _ = _twice(lambda h: client.delete(member, headers={**admin, **_key(h)}), "member-del")
    assert removed.status_code == 204

    issued = client.post("/v1/keys", headers=admin, json={"role": "service", "name": "harness"})
    key_id = issued.json()["key_id"]
    assert issued.status_code == 201 and issued.headers["Location"] == f"/v1/keys/{key_id}"
    _twice(
        lambda h: client.patch(
            f"/v1/keys/{key_id}", headers={**admin, **_key(h)}, json={"may_act_as": ["user:u1"]}
        ),
        "key-patch",
    )
    revoked, _ = _twice(
        lambda h: client.delete(f"/v1/keys/{key_id}", headers={**admin, **_key(h)}), "key-del"
    )
    assert revoked.status_code == 204
    gone, _ = _twice(
        lambda h: client.delete("/v1/workspaces/finance", headers={**admin, **_key(h)}), "ws-del"
    )
    assert gone.status_code == 204


def _key(headers: dict[str, str]) -> dict[str, str]:
    return {"Idempotency-Key": headers["Idempotency-Key"]}


def test_a_read_only_post_ignores_the_key(client) -> None:
    r = client.post("/v1/recall", headers=_keyed("recall-1"), json={"query": "units"})
    again = client.post("/v1/recall", headers=_keyed("recall-1"), json={"query": "other"})
    assert r.status_code == again.status_code == 200
    assert REPLAYED not in again.headers, "a read is answered afresh, never replayed"
