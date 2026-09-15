"""Hardening behaviours at the HTTP edge: rate limiting (per tenant, shared counter,
fail-open on cache outage), body size limit, and error envelopes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memory_service.api.app import create_app

pytestmark = pytest.mark.e2e
H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}


@pytest.fixture
def limited(make_settings, tmp_path):
    settings = make_settings(
        service={"rate_limit_per_minute": 5, "rate_limit_burst": 0, "max_body_bytes": 2048},
        tasks={"provider": "inline"},
        blob={"provider": "filesystem", "filesystem_root": str(tmp_path / "blob")},
    )
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, app.state.container


def test_rate_limit_per_tenant_with_envelope_and_fail_open(limited) -> None:
    client, container = limited
    codes = [client.get("/version", headers=H).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]
    r = client.get("/version", headers=H)
    body = r.json()["error"]
    assert body["code"] == "RATE_LIMIT" and body["retryable"] is True and body["trace_id"]
    assert r.headers["Retry-After"].isdigit() and r.headers["X-RateLimit-Remaining"] == "0"
    # another tenant has its own budget; health endpoints are never limited
    assert client.get("/version", headers={**H, "X-Memory-Tenant": "globex"}).status_code == 200
    assert client.get("/health/live").status_code == 200
    # a cache outage fails open: the limit is a hardening measure, not availability's master
    container.cache.available = False
    try:
        assert client.get("/version", headers=H).status_code == 200
    finally:
        container.cache.available = True
    assert client.get("/version", headers=H).status_code == 429


def test_body_size_limit_is_enforced_before_parsing(limited) -> None:
    client, _ = limited
    r = client.post(
        "/v1/observations",
        headers=H,
        content=b"x" * 4096,
    )
    assert r.status_code == 413 and r.json()["error"]["code"] == "VALIDATION"
