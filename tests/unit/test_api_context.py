"""build_context: trusted headers vs body lineage; security fields can't be spoofed."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context

HEADERS = {
    "X-API-Key": "test-key",
    "X-Trellis-Tenant": "acme",
    "X-Trellis-User": "u1",
}


def _app(settings, overrides):
    app = create_app(settings, overrides=overrides)
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


def test_headers_and_body_merge(settings, overrides) -> None:
    """Lineage comes from the body; the groups stay exactly what the header said."""
    with TestClient(_app(settings, overrides), raise_server_exceptions=False) as c:
        r = c.post(
            "/echo-context",
            headers=HEADERS,
            json={"thread_id": "thr_1", "session_id": "ses_1", "turn_id": "trn_1"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tenant_id"] == "acme" and body["user_id"] == "u1"
        assert body["thread_id"] == "thr_1" and body["turn_id"] == "trn_1"
        assert body["request_id"] == r.headers["X-Request-ID"]


def test_a_body_cannot_smuggle_a_field_the_scope_does_not_declare(settings, overrides) -> None:
    """``ScopeBody`` is ``extra="forbid"``, so an unknown field is refused rather than merged.

    This used to be about ``group_ids`` specifically: the body's groups were UNIONED with the
    header's, so a body could add ``group:acme/ops`` to its own read keys and see every row
    addressed to that group with no membership checked. The GROUP audience is gone now, but
    the property that caught it - a body may not introduce scope the header did not grant -
    is what keeps the next such field from doing the same.
    """
    with TestClient(_app(settings, overrides), raise_server_exceptions=False) as c:
        r = c.post("/echo-context", headers=HEADERS, json={"group_ids": ["ops"]})
        assert r.status_code == 422, r.text


def test_body_cannot_override_trusted_headers(settings, overrides) -> None:
    with TestClient(_app(settings, overrides), raise_server_exceptions=False) as c:
        r = c.post("/echo-context", headers=HEADERS, json={"tenant_id": "globex"})
        assert r.status_code == 422 and r.json()["code"] == "VALIDATION"
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


def test_missing_tenant_and_auth(settings, overrides) -> None:
    with TestClient(_app(settings, overrides), raise_server_exceptions=False) as c:
        r = c.post("/echo-context", headers={"X-API-Key": "test-key"}, json={})
        assert r.status_code == 422 and "tenant_id" in r.json()["detail"]
        r = c.post("/echo-context", headers={"X-Trellis-Tenant": "acme"}, json={})
        assert r.status_code == 401 and r.json()["code"] == "AUTHENTICATION"


def test_the_removed_header_spellings_name_no_tenant(settings, overrides) -> None:
    """``X-Memory-*`` was removed in 0.3.0: a request that names its tenant only that way
    names none."""
    old = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
    with TestClient(_app(settings, overrides), raise_server_exceptions=False) as c:
        r = c.post("/echo-context", headers=old, json={"thread_id": "thr_1"})
        assert r.status_code == 422 and "tenant_id" in r.json()["detail"]


def test_a_correlation_id_named_in_the_body_is_the_one_echoed(settings, overrides) -> None:
    """The body names it (a hand-rolled client's turn id, say); the response and the log
    context carry that id, not the header's or a generated one."""
    with TestClient(_app(settings, overrides), raise_server_exceptions=False) as c:
        from_body = c.post(
            "/echo-context",
            headers=HEADERS,
            json={"thread_id": "thr_1", "correlation_id": "corr-b"},
        )
        assert from_body.status_code == 200, from_body.text
        assert from_body.headers["X-Correlation-ID"] == "corr-b"
        assert from_body.json()["correlation_id"] == "corr-b"
        both = c.post(
            "/echo-context",
            headers={**HEADERS, "X-Correlation-ID": "corr-h"},
            json={"thread_id": "thr_1", "correlation_id": "corr-b"},
        )
        assert both.headers["X-Correlation-ID"] == "corr-b"  # the body's wins, as before 0.2
        invalid = c.post(
            "/echo-context",
            headers=HEADERS,
            json={"thread_id": "thr_1", "correlation_id": "no spaces"},
        )
        assert invalid.status_code == 422
