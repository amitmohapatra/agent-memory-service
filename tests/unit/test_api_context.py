"""build_context: trusted headers vs body lineage; security fields can't be spoofed."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context

HEADERS = {
    "X-API-Key": "test-key",
    "X-Memory-Tenant": "acme",
    "X-Memory-User": "u1",
    "X-Memory-Groups": "legal, finance",
}


def _app(settings):
    app = create_app(settings)
    router = APIRouter()

    @router.post("/echo-context")
    async def echo(
        request: Request,
        container: ContainerDep,
        _: ServicePrincipalDep,
        scope: ScopeBody | None = None,
    ):
        ctx = build_context(request, container, scope)
        return ctx.model_dump(mode="json", exclude={"created_at"})

    app.include_router(router)
    return app


def test_headers_and_body_merge(settings) -> None:
    with TestClient(_app(settings), raise_server_exceptions=False) as c:
        r = c.post(
            "/echo-context",
            headers=HEADERS,
            json={
                "thread_id": "thr_1",
                "session_id": "ses_1",
                "turn_id": "trn_1",
                "group_ids": ["ops"],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tenant_id"] == "acme" and body["user_id"] == "u1"
        assert body["group_ids"] == ["finance", "legal", "ops"]
        assert body["thread_id"] == "thr_1" and body["turn_id"] == "trn_1"
        assert body["request_id"] == r.headers["X-Request-ID"]


def test_body_cannot_override_trusted_headers(settings) -> None:
    with TestClient(_app(settings), raise_server_exceptions=False) as c:
        r = c.post("/echo-context", headers=HEADERS, json={"tenant_id": "globex"})
        assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION"
        r = c.post("/echo-context", headers=HEADERS, json={"user_id": "someone-else"})
        assert r.status_code == 422
        r = c.post(
            "/echo-context", headers=HEADERS, json={"custom_metadata": {"tenant_id": "globex"}}
        )
        assert r.status_code == 422
        r = c.post(
            "/echo-context", headers=HEADERS, json={"session_id": "ses_1"}
        )  # session without thread
        assert r.status_code == 422


def test_missing_tenant_and_auth(settings) -> None:
    with TestClient(_app(settings), raise_server_exceptions=False) as c:
        r = c.post("/echo-context", headers={"X-API-Key": "test-key"}, json={})
        assert r.status_code == 422 and "tenant_id" in r.json()["error"]["message"]
        r = c.post("/echo-context", headers={"X-Memory-Tenant": "acme"}, json={})
        assert r.status_code == 401 and r.json()["error"]["code"] == "AUTHENTICATION"
