"""The M3-lite routes against the hermetic app: feedback, webhooks, team model keys."""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from memory_service.config.constants import WebhookTuning
from memory_service.domain.webhooks import DeliveryStatus
from tests.unit.test_agent_credential_cipher import TEST_KEY
from trellis.memory.webhooks import verify_signature

#: The unit database is shared across runs, so every run administers tenants of its own.
TENANT = f"acme-{secrets.token_hex(3)}"
OTHER = f"globex-{secrets.token_hex(3)}"
ADMIN = {"X-API-Key": "test-key", "X-Trellis-Tenant": TENANT}
USER = {**ADMIN, "X-Trellis-User": "alice", "X-Trellis-Workspace": "fin"}
BOB = {**ADMIN, "X-Trellis-User": "bob"}


@pytest.fixture
def app_client(make_settings, overrides) -> Iterator[TestClient]:
    settings = make_settings(
        agent_credentials={"active_key_id": "test", "encryption_keys": {"test": TEST_KEY}},
        webhooks={"allow_local_targets": True},
    )
    app = create_app(settings, overrides=overrides)
    with TestClient(app) as client:
        # one attempt per delivery and a short fuse, so the dead-endpoint path is one request
        client.app.state.container.services["webhooks"].tuning = WebhookTuning(
            max_attempts=1, disable_after_failures=2
        )
        _onboard(client, TENANT, OTHER)
        yield client


def _onboard(client: TestClient, *tenants: str) -> None:
    """Administrative routes act on tenants somebody onboarded; the dev key is the platform."""
    for tenant_id in tenants:
        created = client.post(
            "/v1/admin/tenants",
            headers={"X-API-Key": "test-key"},
            json={"tenant_id": tenant_id, "name": tenant_id.title()},
        )
        assert created.status_code in (201, 409), created.text


def _problem(response: httpx.Response, status: int) -> dict:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/problem+json")
    return response.json()


# --------------------------------------------------------------------------- feedback


def test_feedback_is_stored_once_per_id_and_bound_to_the_headers(app_client: TestClient) -> None:
    body = {
        "feedback_id": "fb_route_1",
        "target_kind": "run",
        "target_id": "run_1",
        "verdict": "confirm",
        "score": 0.9,
        "reviewer": "alice",
        "metadata": {"channel": "ui"},
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "created_at": "2020-01-01T00:00:00Z",
    }
    first = app_client.post("/v1/feedback", headers=USER, json=body)
    assert first.status_code == 201, first.text
    record = first.json()
    assert record["feedback_id"] == "fb_route_1" and record["tenant_id"] == TENANT
    # provenance is the request's, never the body's
    assert record["trace_id"] == first.headers["X-Trace-ID"] != body["trace_id"]
    assert record["created_at"].startswith("2026")
    assert record["workspace_id"] == "fin" and record["user_id"] == "alice"
    assert record["projection"] is None  # the 201 body precedes the projector
    again = app_client.post("/v1/feedback", headers=USER, json=body)
    assert again.status_code == 200 and again.json()["feedback_id"] == "fb_route_1"
    replay = app_client.post(
        "/v1/feedback", headers={**USER, "Idempotency-Key": "fb-k1"}, json=body
    )
    assert replay.status_code == 200  # the stored record: feedback_id wins before the key
    listed = app_client.get(
        "/v1/feedback", headers=USER, params={"target_kind": "run", "target_id": "run_1"}
    )
    assert listed.status_code == 200
    assert [f["feedback_id"] for f in listed.json()["feedback"]] == ["fb_route_1"]
    assert listed.json()["next_cursor"] is None and "link" not in listed.headers
    one = app_client.get("/v1/feedback/fb_route_1", headers=USER)
    assert one.status_code == 200 and one.json()["verdict"] == "confirm"
    # after the inline projector ran, a run target is recorded, not projected
    assert one.json()["projection"]["action"] == "none"


def test_feedback_claiming_another_identity_is_refused(app_client: TestClient) -> None:
    base = {"target_kind": "run", "target_id": "run_1", "verdict": "confirm"}
    for field, value in (("tenant_id", OTHER), ("user_id", "mallory"), ("workspace_id", "ops")):
        problem = _problem(
            app_client.post("/v1/feedback", headers=USER, json={**base, field: value}), 422
        )
        assert problem["code"] == "VALIDATION" and field in problem["detail"]
    problem = _problem(
        app_client.post("/v1/feedback", headers=USER, json={**base, "verdict": "correct"}), 422
    )
    assert problem["code"] == "VALIDATION"
    unknown_memory = app_client.post(
        "/v1/feedback", headers=USER, json={**base, "target_kind": "memory", "target_id": "mem_x"}
    )
    assert unknown_memory.status_code == 404


def test_feedback_on_other_targets_is_visible_to_its_author_and_workspace(
    app_client: TestClient,
) -> None:
    body = {
        "feedback_id": "fb_vis_1",
        "target_kind": "answer",
        "target_id": "art_1",
        "verdict": "reject",
    }
    assert app_client.post("/v1/feedback", headers=USER, json=body).status_code == 201
    assert app_client.get("/v1/feedback/fb_vis_1", headers=USER).status_code == 200
    teammate = {**ADMIN, "X-Trellis-User": "carol", "X-Trellis-Workspace": "fin"}
    assert app_client.get("/v1/feedback/fb_vis_1", headers=teammate).status_code == 200
    assert app_client.get("/v1/feedback/fb_vis_1", headers=BOB).status_code == 404
    listed = app_client.get(
        "/v1/feedback", headers=BOB, params={"target_kind": "answer", "target_id": "art_1"}
    )
    assert listed.status_code == 200 and listed.json()["feedback"] == []


def test_a_malformed_cursor_is_a_problem_not_a_500(app_client: TestClient) -> None:
    for path, params in (
        ("/v1/feedback", {"target_kind": "run", "target_id": "r", "cursor": "!!"}),
        ("/v1/memories", {"cursor": "!!"}),
        ("/v1/webhooks", {"cursor": "!!"}),
        ("/v1/keys", {"cursor": "!!"}),
    ):
        problem = _problem(app_client.get(path, headers=USER, params=params), 422)
        assert problem["code"] == "VALIDATION" and "cursor" in problem["detail"]


# --------------------------------------------------------------------------- webhooks


def _deliveries(app_client: TestClient, subscription_id: str) -> list[dict]:
    response = app_client.get(f"/v1/webhooks/{subscription_id}/deliveries", headers=ADMIN)
    assert response.status_code == 200, response.text
    return response.json()["deliveries"]


def test_webhook_lifecycle_paging_and_a_signed_test_delivery(app_client: TestClient) -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200)

    container = app_client.app.state.container
    container.services["webhooks"]._client_factory = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    created = app_client.post(
        "/v1/webhooks",
        headers={**ADMIN, "Idempotency-Key": "whk-create-1"},
        json={"url": "https://127.0.0.1:9/hook", "events": ["memory.created", "webhook.test"]},
    )
    assert created.status_code == 201, created.text
    subscription = created.json()
    secret = subscription["secret"]
    assert len(secret) == 64 and subscription["enabled"] and subscription["failures"] == 0
    assert subscription["events"] == ["memory.created", "webhook.test"]
    replay = app_client.post(
        "/v1/webhooks",
        headers={**ADMIN, "Idempotency-Key": "whk-create-1"},
        json={"url": "https://127.0.0.1:9/hook", "events": ["memory.created", "webhook.test"]},
    )
    assert replay.status_code == 201 and replay.json()["secret"] is None
    assert replay.json()["subscription_id"] == subscription["subscription_id"]
    second = app_client.post(
        "/v1/webhooks",
        headers=ADMIN,
        json={
            "url": "http://localhost:9/other",
            "events": ["feedback.received"],
            "workspace_id": "fin",
        },
    )
    assert second.status_code == 201
    page1 = app_client.get("/v1/webhooks", headers=ADMIN, params={"limit": 1})
    assert len(page1.json()["webhooks"]) == 1 and page1.json()["next_cursor"]
    assert 'rel="next"' in page1.headers["link"]
    page2 = app_client.get(
        "/v1/webhooks", headers=ADMIN, params={"limit": 1, "cursor": page1.json()["next_cursor"]}
    )
    assert len(page2.json()["webhooks"]) == 1 and page2.json()["next_cursor"] is None
    ids = {
        page1.json()["webhooks"][0]["subscription_id"],
        page2.json()["webhooks"][0]["subscription_id"],
    }
    assert ids == {subscription["subscription_id"], second.json()["subscription_id"]}

    sid = subscription["subscription_id"]
    got = app_client.get(f"/v1/webhooks/{sid}", headers=ADMIN)
    assert got.status_code == 200 and "secret" not in got.json()
    patched = app_client.patch(
        f"/v1/webhooks/{sid}",
        headers=ADMIN,
        json={"description": "inbox", "events": ["webhook.test"]},
    )
    assert patched.status_code == 200 and patched.json()["events"] == ["webhook.test"]
    assert app_client.patch(f"/v1/webhooks/{sid}", headers=ADMIN, json={}).status_code == 422

    queued = app_client.post(f"/v1/webhooks/{sid}/test", headers=ADMIN)
    assert queued.status_code == 202, queued.text
    delivery = queued.json()
    assert delivery["event_type"] == "webhook.test"
    # the inline queue delivered it during the request
    assert len(received) == 1
    request = received[0]
    assert str(request.url) == "https://127.0.0.1:9/hook"
    assert request.headers["X-Trellis-Event"] == "webhook.test"
    assert request.headers["X-Trellis-Delivery"] == delivery["delivery_id"]
    assert request.headers["X-Request-ID"].startswith("req_")
    assert request.headers["traceparent"].startswith("00-")
    assert verify_signature(secret, request.headers["X-Trellis-Signature"], request.content)
    assert not verify_signature("wrong", request.headers["X-Trellis-Signature"], request.content)
    event = json.loads(request.content)
    assert event["type"] == "webhook.test" and event["tenant_id"] == TENANT
    assert event["data"] == {"subscription_id": sid}
    rows = _deliveries(app_client, sid)
    assert rows[0]["status"] == DeliveryStatus.DELIVERED.value and rows[0]["attempts"] == 1
    assert rows[0]["status_code"] == 200

    gone = app_client.delete(f"/v1/webhooks/{sid}", headers=ADMIN)
    assert gone.status_code == 204
    assert app_client.get(f"/v1/webhooks/{sid}", headers=ADMIN).status_code == 404
    assert app_client.post(f"/v1/webhooks/{sid}/test", headers=ADMIN).status_code == 404


def test_a_failing_endpoint_goes_dead_is_counted_and_finally_disabled(
    app_client: TestClient,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    container = app_client.app.state.container
    container.services["webhooks"]._client_factory = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    created = app_client.post(
        "/v1/webhooks",
        headers=ADMIN,
        json={"url": "https://127.0.0.1:9/h", "events": ["webhook.test"]},
    )
    sid = created.json()["subscription_id"]
    assert app_client.post(f"/v1/webhooks/{sid}/test", headers=ADMIN).status_code == 202
    assert calls == 1  # max_attempts=1 for this app: the first attempt is the last
    rows = _deliveries(app_client, sid)
    assert rows[0]["status"] == "DEAD" and rows[0]["attempts"] == 1
    assert rows[0]["status_code"] == 500 and rows[0]["last_error"] == "HTTP 500"
    assert app_client.get(f"/v1/webhooks/{sid}", headers=ADMIN).json()["failures"] == 1
    # the second consecutive dead delivery reaches disable_after_failures=2
    assert app_client.post(f"/v1/webhooks/{sid}/test", headers=ADMIN).status_code == 202
    after = app_client.get(f"/v1/webhooks/{sid}", headers=ADMIN).json()
    assert calls == 2 and after["failures"] == 2 and after["enabled"] is False
    # a disabled subscription gets nothing; re-enabling forgives the history
    assert app_client.post(f"/v1/webhooks/{sid}/test", headers=ADMIN).status_code == 202
    assert calls == 2
    assert _deliveries(app_client, sid)[0]["last_error"] == "subscription gone or disabled"
    enabled = app_client.patch(f"/v1/webhooks/{sid}", headers=ADMIN, json={"enabled": True})
    assert enabled.json()["enabled"] and enabled.json()["failures"] == 0


def test_private_targets_are_refused_when_the_policy_is_strict(make_settings, overrides) -> None:
    settings = make_settings(
        agent_credentials={"active_key_id": "test", "encryption_keys": {"test": TEST_KEY}}
    )
    with TestClient(create_app(settings, overrides=overrides)) as client:
        _onboard(client, TENANT)
        for url in ("http://hooks.example.com/x", "https://127.0.0.1/x", "https://localhost/x"):
            problem = _problem(
                client.post(
                    "/v1/webhooks", headers=ADMIN, json={"url": url, "events": ["webhook.test"]}
                ),
                422,
            )
            assert problem["code"] == "VALIDATION"


def test_webhooks_are_tenant_administration(app_client: TestClient) -> None:
    other = {"X-API-Key": "test-key", "X-Trellis-Tenant": OTHER}
    created = app_client.post(
        "/v1/webhooks",
        headers=ADMIN,
        json={"url": "https://127.0.0.1:9/h", "events": ["webhook.test"]},
    )
    sid = created.json()["subscription_id"]
    assert app_client.get(f"/v1/webhooks/{sid}", headers=other).status_code == 404
    assert app_client.get("/v1/webhooks", headers=other).json()["webhooks"] == []


# --------------------------------------------------------------------------- team model keys


def test_workspace_and_tenant_model_keys_are_administered_without_showing_secrets(
    app_client: TestClient,
) -> None:
    assert (
        app_client.post(
            "/v1/workspaces", headers=ADMIN, json={"name": "Finance", "workspace_id": "fin"}
        ).status_code
        == 201
    )
    status = app_client.get("/v1/workspaces/fin/model-key", headers=ADMIN)
    assert status.status_code == 200 and status.json() == {
        "registered": False,
        "revoked": False,
        "revision": 0,
        "updated_at": None,
    }
    put = app_client.put(
        "/v1/workspaces/fin/model-key", headers=ADMIN, json={"virtual_key": "vk-team-test"}
    )
    assert put.status_code == 200, put.text
    assert put.json()["registered"] and not put.json()["revoked"] and put.json()["revision"] == 1
    assert "vk-team-test" not in put.text
    rotated = app_client.put(
        "/v1/workspaces/fin/model-key", headers=ADMIN, json={"virtual_key": "vk-team-2"}
    )
    assert rotated.json()["revision"] == 2
    revoked = app_client.delete("/v1/workspaces/fin/model-key", headers=ADMIN)
    assert revoked.status_code == 200 and revoked.json()["revoked"]
    assert app_client.get("/v1/workspaces/nope/model-key", headers=ADMIN).status_code == 404
    assert (
        app_client.put(
            "/v1/workspaces/fin/model-key", headers=ADMIN, json={"virtual_key": "has space"}
        ).status_code
        == 422
    )

    tenant = app_client.put("/v1/model-key", headers=ADMIN, json={"virtual_key": "vk-tenant-test"})
    assert tenant.status_code == 200 and tenant.json()["registered"]
    assert app_client.get("/v1/model-key", headers=ADMIN).json()["revision"] == 1
    assert app_client.delete("/v1/model-key", headers=ADMIN).json()["revoked"]
    # the agent's own route is unchanged
    own = app_client.get(
        "/v1/agents/model-key",
        headers={**USER, "X-Trellis-Workspace": "fin"},
        params={"agent_id": "ref"},
    )
    assert own.status_code == 200 and not own.json()["registered"]


async def test_the_sdk_body_is_what_the_service_accepts(app_client: TestClient) -> None:
    """The SDK is exercised against the real request model: the contracts record shape,
    evidence references without ``observed_at``, a comment, and nothing else of the scope."""
    from trellis.memory import EvidenceRef, MemoryClient

    transport = httpx.ASGITransport(app=app_client.app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as http,
        MemoryClient("http://testserver", api_key="test-key", http_client=http) as client,
        client.bind(
            tenant_id=TENANT, user_id="alice", workspace_id="fin", agent_id="ref", thread_id="thr_1"
        ) as ctx,
    ):
        record = await ctx.feedback.submit(
            "run",
            "run_sdk",
            "confirm",
            score=0.7,
            comment="looked right",
            evidence_refs=[EvidenceRef(source_type="message", source_id="msg_1")],
            feedback_id=f"fb_sdk_{secrets.token_hex(3)}",
        )
        assert record.comment == "looked right" and record.agent_id == "ref"
        assert record.evidence_refs[0].source_id == "msg_1"
        assert [f.feedback_id for f in await ctx.feedback.list_for("run", "run_sdk")]
        page = await ctx.feedback.page_for("run", "run_sdk", limit=1)
        assert page.items[0].feedback_id == record.feedback_id


def test_every_public_route_refuses_a_missing_credential(app_client: TestClient) -> None:
    """Authentication is a dependency each route declares; a route that forgets it is open.
    Dependencies are solved before the body is validated, so no body is needed here."""
    from fastapi.routing import APIRoute

    unguarded = []
    for route in app_client.app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/v1"):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            path = route.path.replace("{", "x").replace("}", "")
            response = app_client.request(method, path, headers={"X-Trellis-Tenant": TENANT})
            if response.status_code != 401:
                unguarded.append(f"{method} {route.path} -> {response.status_code}")
    assert not unguarded, unguarded


def test_a_forged_cursor_value_is_a_validation_problem(app_client: TestClient) -> None:
    from memory_service.api.pagination import encode_cursor

    forged = {
        "/v1/feedback": (
            {"target_kind": "run", "target_id": "r"},
            {"created_at": "garbage", "feedback_id": "fb_1"},
        ),
        "/v1/keys": ({}, {"created_at": "2026-09-28T07:00:00", "key_id": "k"}),  # naive
        "/v1/reads": ({}, {"before": 12}),
        "/v1/memories": ({}, {"created_at": "x", "memory_id": "m"}),
        "/v1/threads/thr_x/messages": ({}, {"before_sequence": "1"}),
        "/v1/webhooks": ({}, {"subscription_id": 1}),
    }
    for path, (params, position) in forged.items():
        problem = _problem(
            app_client.get(
                path, headers=USER, params={**params, "cursor": encode_cursor(position)}
            ),
            422,
        )
        assert problem["code"] == "VALIDATION" and "cursor" in problem["detail"], path


def test_admin_lists_page_one_row_at_a_time(app_client: TestClient) -> None:
    for n in range(2):
        assert (
            app_client.post(
                "/v1/workspaces", headers=ADMIN, json={"name": f"W{n}", "workspace_id": f"w{n}"}
            ).status_code
            == 201
        )
        assert (
            app_client.post(
                "/v1/groups", headers=ADMIN, json={"name": f"G{n}", "group_id": f"g{n}"}
            ).status_code
            == 201
        )
        assert (
            app_client.post(
                "/v1/keys", headers=ADMIN, json={"role": "service", "name": f"K{n}"}
            ).status_code
            == 201
        )
    for path, key in (
        ("/v1/workspaces", "workspace_id"),
        ("/v1/groups", "group_id"),
        ("/v1/keys", "key_id"),
    ):
        first = app_client.get(path, headers=ADMIN, params={"limit": 1})
        assert first.status_code == 200 and len(first.json()) == 1, path
        assert 'rel="next"' in first.headers["link"], path
        cursor = first.headers["link"].split("cursor=")[1].split(">")[0]
        second = app_client.get(path, headers=ADMIN, params={"limit": 1, "cursor": cursor})
        assert second.status_code == 200 and len(second.json()) == 1
        assert second.json()[0][key] != first.json()[0][key]
        seen = {first.json()[0][key], second.json()[0][key]}
        rest = app_client.get(path, headers=ADMIN, params={"limit": 100})
        assert seen <= {row[key] for row in rest.json()} and "link" not in rest.headers
    tenants = app_client.get(
        "/v1/admin/tenants", headers={"X-API-Key": "test-key"}, params={"limit": 1}
    )
    assert (
        tenants.status_code == 200
        and len(tenants.json()) == 1
        and 'rel="next"' in tenants.headers["link"]
    )


def test_messages_page_backwards_through_the_cursor(app_client: TestClient) -> None:
    # no workspace header: writes into a workspace need membership, and another test in
    # this module creates the workspace the shared headers name
    writer = {**ADMIN, "X-Trellis-User": "alice"}
    thread = f"thr_page_{secrets.token_hex(3)}"
    scope = {"thread_id": thread, "session_id": f"ses_{thread}", "turn_id": f"trn_{thread}"}
    for text in ("one", "two", "three"):
        r = app_client.post(
            "/v1/messages", headers=writer, json={"scope": scope, "role": "USER", "content": text}
        )
        assert r.status_code == 202, r.text
    newest = app_client.get(
        f"/v1/threads/{scope['thread_id']}/messages", headers=writer, params={"limit": 2}
    )
    assert newest.status_code == 200, newest.text
    page = newest.json()
    assert [m["content"] for m in page["messages"]] == ["two", "three"]
    assert page["next_cursor"] and 'rel="next"' in newest.headers["link"]
    older = app_client.get(
        f"/v1/threads/{thread}/messages",
        headers=writer,
        params={"limit": 2, "cursor": page["next_cursor"]},
    )
    assert [m["content"] for m in older.json()["messages"]] == ["one"]
    assert older.json()["next_cursor"] is None and "link" not in older.headers
    exact = app_client.get(
        f"/v1/threads/{scope['thread_id']}/messages", headers=USER, params={"limit": 3}
    )
    assert len(exact.json()["messages"]) == 3 and "link" not in exact.headers


def test_feedback_pages_skip_rows_the_caller_may_not_see(app_client: TestClient) -> None:
    target = f"run_hidden_{secrets.token_hex(3)}"
    order = []
    for who, n in (("alice", 0), ("bob", 1), ("alice", 2), ("bob", 3), ("bob", 4)):
        headers = USER if who == "alice" else BOB
        r = app_client.post(
            "/v1/feedback",
            headers=headers,
            json={
                "feedback_id": f"fb_{target}_{n}",
                "target_kind": "run",
                "target_id": target,
                "verdict": "confirm",
            },
        )
        assert r.status_code == 201, r.text
        order.append((who, f"fb_{target}_{n}"))
    seen, cursor = [], None
    while True:
        params = {"target_kind": "run", "target_id": target, "limit": 1}
        if cursor:
            params["cursor"] = cursor
        page = app_client.get("/v1/feedback", headers=BOB, params=params).json()
        seen += [f["feedback_id"] for f in page["feedback"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == [fid for who, fid in reversed(order) if who == "bob"]


def test_a_delivery_is_pinned_to_the_vetted_address_and_a_failed_one_progresses_to_dead(
    app_client: TestClient,
) -> None:
    seen: list[httpx.Request] = []
    status = {"code": 500}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status["code"])

    container = app_client.app.state.container
    service = container.services["webhooks"]
    service._client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service.tuning = WebhookTuning(max_attempts=2, disable_after_failures=1)
    created = app_client.post(
        "/v1/webhooks",
        headers=ADMIN,
        json={"url": "https://127.0.0.1:9/pin?x=1", "events": ["webhook.test"]},
    )
    sid = created.json()["subscription_id"]
    queued = app_client.post(f"/v1/webhooks/{sid}/test", headers=ADMIN)
    assert queued.status_code == 202
    # attempt 1 (inline queue): FAILED, counted on the row, not against the subscription
    assert str(seen[0].url) == "https://127.0.0.1:9/pin?x=1"
    assert seen[0].headers["host"] == "127.0.0.1:9"
    assert seen[0].extensions.get("sni_hostname") == "127.0.0.1"
    rows = _deliveries(app_client, sid)
    assert rows[0]["status"] == "FAILED" and rows[0]["attempts"] == 1
    assert app_client.get(f"/v1/webhooks/{sid}", headers=ADMIN).json()["failures"] == 0
    # attempt 2 (the queue's retry): the last permitted one, so DEAD and disabling
    import asyncio

    asyncio.run(service.deliver({"tenant_id": TENANT, "delivery_id": rows[0]["delivery_id"]}))
    rows = _deliveries(app_client, sid)
    assert rows[0]["status"] == "DEAD" and rows[0]["attempts"] == 2
    after = app_client.get(f"/v1/webhooks/{sid}", headers=ADMIN).json()
    assert after["failures"] == 1 and after["enabled"] is False
    # a settled row is left alone by a duplicate run
    asyncio.run(service.deliver({"tenant_id": TENANT, "delivery_id": rows[0]["delivery_id"]}))
    assert len(seen) == 2 and _deliveries(app_client, sid)[0]["attempts"] == 2
