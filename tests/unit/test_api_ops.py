from fastapi.testclient import TestClient

from tests.conftest import PG_AVAILABLE


def test_live_ready_version_metrics(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    # Readiness is the one endpoint whose *correct* answer depends on the machine, so assert
    # the mapping rather than a fixed status. Asserting 200 unconditionally made this unit
    # test require a live PostgreSQL without saying so: with the database down it failed
    # here, 503-with-postgres-not-ok — the endpoint working exactly as designed — while
    # every genuinely database-backed test skipped cleanly on PG_AVAILABLE.
    body = ready.json()
    assert body["dependencies"]["postgres"]["mandatory"] is True
    if PG_AVAILABLE:
        assert ready.status_code == 200
        assert body["status"] in ("ready", "degraded")
        assert body["dependencies"]["postgres"]["ok"] is True
    else:
        assert ready.status_code == 503
        assert body["status"] == "not_ready"
        assert body["dependencies"]["postgres"]["ok"] is False
    version = client.get("/version").json()
    assert version["version"] and version["api_version"] == "v1"
    assert version["providers"]["llm"] == "disabled"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200 and b"memory_http_requests_total" in metrics.content


def test_swagger_redoc_openapi_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Memory Service API"
    assert "ErrorEnvelope" in schema["components"]["schemas"]
    assert set(schema["components"]["securitySchemes"]) == {"ApiKeyAuth", "BearerAuth"}


def test_correlation_headers_round_trip(client: TestClient) -> None:
    r = client.get(
        "/version", headers={"X-Request-ID": "req_custom-1", "X-Correlation-ID": "corr-1"}
    )
    assert r.headers["X-Request-ID"] == "req_custom-1"
    assert r.headers["X-Correlation-ID"] == "corr-1"
    assert r.headers["X-Trace-ID"]
    r2 = client.get("/version", headers={"X-Request-ID": "bad id with spaces"})
    assert r2.headers["X-Request-ID"].startswith("req_")


def test_error_envelope_for_unknown_route(client: TestClient) -> None:
    r = client.get("/does-not-exist")
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "NOT_FOUND" and body["retryable"] is False and body["trace_id"]


def test_body_limit_returns_envelope(client: TestClient) -> None:
    r = client.post("/version", headers={"Content-Length": str(10**9)}, content=b"")
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "VALIDATION"


def test_domain_error_maps_to_envelope(settings) -> None:
    from fastapi import APIRouter

    from memory_service.api.app import create_app
    from memory_service.domain.errors import DependencyUnavailable, ScopeDenied

    app = create_app(settings)
    router = APIRouter()

    @router.get("/boom-scope")
    async def boom_scope():
        raise ScopeDenied("nope", details={"object": "thread:t1"})

    @router.get("/boom-dep")
    async def boom_dep():
        raise DependencyUnavailable("postgres down")

    @router.get("/boom-unhandled")
    async def boom_unhandled():
        raise RuntimeError("secret internal detail")

    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom-scope")
        assert r.status_code == 403 and r.json()["error"] == {
            "code": "SCOPE_DENIED",
            "message": "nope",
            "retryable": False,
            "trace_id": r.headers["X-Trace-ID"],
            "details": {"object": "thread:t1"},
        }
        r = c.get("/boom-dep")
        assert r.status_code == 503 and r.json()["error"]["retryable"] is True
        r = c.get("/boom-unhandled")
        assert r.status_code == 500
        assert "secret internal detail" not in r.text
