"""Hardening behaviours at the HTTP edge: rate limiting (per tenant, shared counter,
fail-open on cache outage), body size limit, and problem details."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from memory_service.config import constants
from tests.conftest import _test_overrides

pytestmark = pytest.mark.e2e
H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}


@pytest.fixture
def limited(make_settings, tmp_path, monkeypatch):
    # The rate, the burst allowance and the body cap are constants of the package, not
    # settings (config/constants.py; a tenant's quota narrows the rate); the test lowers them
    # the only way an operator cannot.
    monkeypatch.setattr(constants, "RATE_LIMIT_PER_MINUTE", 5)
    monkeypatch.setattr(constants, "RATE_LIMIT_BURST", 0)
    monkeypatch.setattr(constants, "MAX_BODY_BYTES", 2048)
    settings = make_settings(
        blob={"provider": "filesystem", "filesystem_root": str(tmp_path / "blob")},
    )
    app = create_app(settings, overrides=_test_overrides(tasks="inline", blob=None))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, app.state.container


def test_rate_limit_per_tenant_with_a_problem_and_fail_open(limited) -> None:
    client, container = limited
    codes = [client.get("/version", headers=H).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]
    r = client.get("/version", headers=H)
    body = r.json()
    assert body["code"] == "RATE_LIMIT" and body["retryable"] is True and body["trace_id"]
    assert r.headers["Retry-After"].isdigit() and r.headers["X-RateLimit-Remaining"] == "0"
    # another tenant has its own budget; health endpoints are never limited
    assert client.get("/version", headers={**H, "X-Trellis-Tenant": "globex"}).status_code == 200
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
        "/v1/messages",
        headers=H,
        content=b"x" * 4096,
    )
    assert r.status_code == 413 and r.json()["code"] == "VALIDATION"


def test_a_streamed_body_is_counted_against_the_limit(limited) -> None:
    """No Content-Length (chunked) is no way past the limit: the bytes are counted as they
    arrive and the same 413 problem answers once they pass it."""
    client, _ = limited

    def chunks():  # type: ignore[no-untyped-def]
        for _ in range(8):
            yield b"x" * 512

    r = client.post(
        "/v1/messages", headers={**H, "Content-Type": "application/json"}, content=chunks()
    )
    assert "content-length" not in {k.lower() for k in r.request.headers}
    assert r.status_code == 413 and r.headers["content-type"] == "application/problem+json"
    assert r.json()["code"] == "VALIDATION" and "2048" in r.json()["detail"]
    small = client.post(
        "/v1/messages",
        headers={**H, "Content-Type": "application/json"},
        content=iter(
            [
                b'{"scope": {"thread_id": "thr_s"}, ',
                b'"messages": [{"role": "USER", "content": "hi"}]}',
            ]
        ),
    )
    assert small.status_code == 202, small.text


async def test_an_upload_is_refused_as_it_passes_the_file_limit() -> None:
    import io

    from fastapi import UploadFile

    from memory_service.api.routers.v1.files import UPLOAD_CHUNK_BYTES, read_bounded
    from memory_service.domain.errors import ValidationFailed

    class Counting(io.BytesIO):
        reads = 0

        def read(self, size: int | None = -1) -> bytes:
            Counting.reads += 1
            return super().read(size)

    big = UploadFile(Counting(b"x" * (UPLOAD_CHUNK_BYTES * 10)))
    with pytest.raises(ValidationFailed, match="exceeds"):
        await read_bounded(big, UPLOAD_CHUNK_BYTES + 1)
    assert Counting.reads == 2, "stopped at the chunk that passed the limit"
    sized = UploadFile(io.BytesIO(b"abc"), size=10_000)
    with pytest.raises(ValidationFailed):
        await read_bounded(sized, 100)
    assert await read_bounded(UploadFile(io.BytesIO(b"abc")), 100) == b"abc"
