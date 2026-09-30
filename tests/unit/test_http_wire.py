"""What leaves the process on the wire, and what the two middlewares cost to get it there.

The middlewares are plain ASGI callables now; every behaviour they had as
``BaseHTTPMiddleware`` subclasses is asserted here rather than inferred, because the ones
that matter (the 413 before the body is read, fail-open when the cache is down, the
exemptions) only show up when something is wrong.
"""

from __future__ import annotations

import time
import types
from typing import Any

import pytest
from fastapi import APIRouter, Request
from fastapi.testclient import TestClient

from memory_service.api import middleware
from memory_service.api.app import create_app
from memory_service.api.routers.v1.retrieval import ContextResponse
from memory_service.config import constants
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.modules.context.builder import bundle_to_api

pytestmark = pytest.mark.unit

H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}

#: A judged LoCoMo run renders 9.1k characters at p50 and packs tens of items; this is a
#: bundle of that shape, built here so the size of a /v1/context response can be measured
#: without a database.
_ITEM_TEXT = (
    "On 8 May 2023 Caroline said she had finally booked the trip to Lisbon with her sister, "
    "after moving it twice because of work, and that they would be staying near Alfama. "
)


def _bundle(items: int = 40) -> ContextBundle:
    def item(i: int, kind: str) -> ContextItem:
        return ContextItem(
            item_id=f"mem_{i:026d}",
            representation=Representation.MEMORY if kind == "memory" else Representation.CHUNK,
            text=_ITEM_TEXT * 2,
            score=0.5 + i / 1000,
            relevance=0.5,
            retrievers=["dense", "bm25"],
            citation=f"{kind}_id:{i}",
            token_estimate=60,
            attributes={
                "subject": "user:caroline",
                "predicate": "travelled_to",
                "object": "Lisbon",
                "observed_at": "2023-05-08T00:00:00Z",
                "memory_type": "EPISODIC",
                "confidence": 0.9,
            },
        )

    half = items // 2
    return ContextBundle(
        query="Where did Caroline go in May and who was with her?",
        query_type=QueryType.DOCUMENT_MULTI_HOP,
        bundle_id="bnd_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
        conversation=ConversationWindow(
            thread_id="thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
            message_ids=[f"msg_{i}" for i in range(10)],
            rendered="USER: " + _ITEM_TEXT * 4,
            token_estimate=400,
        ),
        memories=[item(i, "memory") for i in range(half)],
        knowledge=[item(i, "chunk") for i in range(items - half)],
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=8000,
        token_estimate=4000,
    )


def _app_returning_a_bundle(settings, overrides):
    """The /v1/context response shape, served by the real app: same response model, same
    middleware stack, no database."""
    app = create_app(settings, overrides=overrides)
    router = APIRouter()
    payload = bundle_to_api(_bundle())

    @router.get("/bundle-shaped", response_model=ContextResponse)
    async def bundle_shaped() -> Any:
        return payload

    app.include_router(router)
    return app


def test_a_context_sized_response_is_compressed_on_the_wire(settings, overrides) -> None:
    """The measurement Phase 2 asks for: what a /v1/context-sized body costs off-box.

    Recorded in the assertion rather than in a comment, so it cannot quietly stop being
    true: the same response, gzipped, must be a fraction of its size. `rendered` repeats
    every item's text, which is exactly what a compressor is good at.
    """
    with TestClient(_app_returning_a_bundle(settings, overrides)) as client:
        plain = client.get("/bundle-shaped", headers={"Accept-Encoding": "identity"})
        zipped = client.get("/bundle-shaped", headers={"Accept-Encoding": "gzip"})
    assert plain.status_code == zipped.status_code == 200
    assert "content-encoding" not in plain.headers
    assert zipped.headers["content-encoding"] == "gzip"
    uncompressed = int(plain.headers["content-length"])
    compressed = int(zipped.headers["content-length"])
    assert plain.json() == zipped.json(), "compression must not change the body"
    # measured on this fixture: 49,773 B -> 1,853 B, a factor of 27
    assert uncompressed > 30_000, f"the fixture stopped being context-sized ({uncompressed} B)"
    assert compressed < uncompressed / 5, (
        f"gzip bought less than 5x on a context bundle: {uncompressed} B -> {compressed} B"
    )


def test_a_small_response_is_not_worth_compressing(settings, overrides) -> None:
    """Below the threshold the CPU and the 20-byte gzip header are pure loss."""
    with TestClient(create_app(settings, overrides=overrides)) as client:
        r = client.get("/health/live", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and "content-encoding" not in r.headers


# --- correlation ------------------------------------------------------------------------


def test_the_ids_survive_the_pure_asgi_middleware(settings, overrides) -> None:
    with TestClient(create_app(settings, overrides=overrides)) as client:
        r = client.get("/version", headers={"X-Request-ID": "req_a", "X-Correlation-ID": "corr-1"})
        assert r.headers["X-Request-ID"] == "req_a"
        assert r.headers["X-Correlation-ID"] == "corr-1"
        assert r.headers["X-Trace-ID"]
        bad = client.get("/version", headers={"X-Request-ID": "not a valid id"})
        assert bad.headers["X-Request-ID"].startswith("req_")


def test_the_request_sees_the_ids_the_middleware_set(settings, overrides) -> None:
    """``request.state`` is written through the ASGI scope now; the handlers that read it
    (build_context, the idempotency keys) must not notice the difference."""
    app = create_app(settings, overrides=overrides)
    router = APIRouter()

    @router.get("/state")
    async def state(request: Request) -> dict[str, Any]:
        return {
            "request_id": request.state.request_id,
            "trace_id": request.state.trace_id,
            "correlation_id": request.state.correlation_id,
            "idempotency_key": request.state.idempotency_key,
        }

    app.include_router(router)
    with TestClient(app) as client:
        r = client.get("/state", headers={"X-Request-ID": "req_b", "Idempotency-Key": "k1"})
    body = r.json()
    assert body["request_id"] == "req_b" == r.headers["X-Request-ID"]
    assert body["idempotency_key"] == "k1"
    assert body["trace_id"] == r.headers["X-Trace-ID"]


def test_the_body_limit_is_enforced_before_the_body_is_read(settings, overrides) -> None:
    with TestClient(create_app(settings, overrides=overrides)) as client:
        r = client.post("/version", headers={"Content-Length": str(10**9)}, content=b"")
    assert r.status_code == 413 and r.json()["code"] == "VALIDATION"


def test_a_request_is_counted_once_under_its_route(settings, overrides) -> None:
    """The route label is the matched path, read off the ASGI scope after the router put it
    there - not the raw URL, which would give every id its own time series."""
    from memory_service.observability.metrics import http_requests_total

    counter = http_requests_total.labels("GET", "/version", "200")
    before = counter._value.get()
    with TestClient(create_app(settings, overrides=overrides)) as client:
        client.get("/version")
        metrics = client.get("/metrics").text
    assert counter._value.get() == before + 1
    assert 'route="/version"' in metrics and "/metrics" not in metrics.split("\n")[0]


async def test_a_cancelled_request_is_still_counted() -> None:
    """A client that goes away cancels the task, and CancelledError is not an Exception. The
    request happened - it occupied a worker for as long as it lasted - and a run at 20 rps
    cares about exactly this population, so it is counted rather than dropped."""
    import asyncio

    from memory_service.api.middleware import CorrelationMiddleware
    from memory_service.observability.metrics import http_requests_total

    async def _cancelled(scope, receive, send) -> None:
        raise asyncio.CancelledError

    counter = http_requests_total.labels("GET", "unmatched", "499")
    before = counter._value.get()
    app = CorrelationMiddleware(_cancelled, max_body_bytes=1024)
    scope = {"type": "http", "method": "GET", "path": "/v1/context", "headers": []}
    with pytest.raises(asyncio.CancelledError):
        await app(scope, _noop_receive, _noop_send)
    assert counter._value.get() == before + 1


async def _noop_receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


async def _noop_send(message: dict[str, Any]) -> None:
    return None


# --- rate limit -------------------------------------------------------------------------


@pytest.fixture
def limited(make_settings, monkeypatch, overrides):
    monkeypatch.setattr(constants, "RATE_LIMIT_BURST", 0)
    # the window is a wall-clock minute; a test that straddles the boundary would see two
    frozen = types.SimpleNamespace(time=lambda: 1_790_553_605.0, perf_counter=time.perf_counter)
    monkeypatch.setattr(middleware, "time", frozen)
    settings = make_settings(service={"rate_limit_per_minute": 3})
    app = create_app(settings, overrides=overrides)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, app.state.container


def test_the_window_is_per_tenant_and_fails_open(limited) -> None:
    client, container = limited
    assert [client.get("/version", headers=H).status_code for _ in range(4)] == [200, 200, 200, 429]
    r = client.get("/version", headers=H)
    body = r.json()
    assert r.headers["content-type"] == "application/problem+json"
    assert body["code"] == "RATE_LIMIT" and body["retryable"] is True and body["trace_id"]
    assert body["status"] == 429 and body["instance"] == "/version"
    assert r.headers["Retry-After"].isdigit() and r.headers["X-RateLimit-Remaining"] == "0"
    assert client.get("/version", headers={**H, "X-Trellis-Tenant": "globex"}).status_code == 200
    assert client.get("/health/live").status_code == 200
    container.cache.available = False
    try:
        assert client.get("/version", headers=H).status_code == 200
    finally:
        container.cache.available = True
    assert client.get("/version", headers=H).status_code == 429


def test_the_remaining_budget_is_reported_while_there_is_one(limited) -> None:
    client, _ = limited
    r = client.get("/version", headers=H)
    assert r.headers["X-RateLimit-Limit"] == "3" and r.headers["X-RateLimit-Remaining"] == "2"


async def test_the_counter_and_its_expiry_are_one_round_trip() -> None:
    """A fixed window is INCR plus EXPIRE; issued separately that is two waits on a remote
    cache for every request that is not the first of its minute."""
    from memory_service.adapters.cache.memory_cache import MemoryCache

    cache = MemoryCache()
    assert await cache.incr_window("w", ttl_seconds=120) == 1
    assert await cache.incr_window("w", ttl_seconds=120) == 2
    assert int((await cache.get("w")).decode()) == 2


def test_the_window_is_shared_by_both_header_spellings(limited) -> None:
    """Alternating the deprecated and the new tenant header spends one budget, not two."""
    client, _ = limited
    old = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
    codes = [client.get("/version", headers=h).status_code for h in (H, old, H, old)]
    assert codes == [200, 200, 200, 429]


def test_the_llm_token_header_reports_what_the_request_spent(settings, overrides) -> None:
    from memory_service.modules.llm.cost import record_llm_tokens

    app = create_app(settings, overrides=overrides)
    router = APIRouter()

    @router.get("/spend")
    async def spend() -> dict[str, int]:
        record_llm_tokens(10, 20)
        return {}

    app.include_router(router)
    with TestClient(app) as client:
        r = client.get("/spend")
        assert r.headers["X-Trellis-LLM-Tokens"] == "30"
        assert r.headers["X-Memory-LLM-Tokens"] == "30"  # the old spelling, until 0.3.0
        quiet = client.get("/version")
        assert "X-Trellis-LLM-Tokens" not in quiet.headers
        assert "X-Memory-LLM-Tokens" not in quiet.headers


def test_a_conflicting_tenant_spelling_is_refused_before_any_bucket_is_touched(limited) -> None:
    """A client must not name a bucket per request with values the context builder is about
    to refuse; the refusal happens in the limiter, before the credential is even verified."""
    client, _ = limited
    conflict = {**H, "X-Memory-Tenant": "globex"}
    for _ in range(5):
        r = client.get("/version", headers=conflict)
        assert r.status_code == 422 and r.headers["content-type"] == "application/problem+json"
        assert r.json()["details"] == {"field": "X-Trellis-Tenant"}
    # the tenant's own budget is untouched by the refused requests
    assert [client.get("/version", headers=H).status_code for _ in range(4)] == [200, 200, 200, 429]
    # refused before the credential is read: no key, same answer; two keys, same answer
    unauthenticated = client.get(
        "/version", headers={"X-Trellis-Tenant": "acme", "X-Memory-Tenant": "globex"}
    )
    assert unauthenticated.status_code == 422
    two_keys = client.get("/version", headers=[("X-API-Key", "mk_a.1"), ("X-API-Key", "mk_b.2")])
    assert two_keys.status_code == 422 and two_keys.json()["details"] == {"field": "X-API-Key"}


def test_the_refusal_does_not_depend_on_the_rate_limiter(settings, overrides) -> None:
    """The check lives in the correlation middleware, so a deployment without a cache (or with
    the limiter off) refuses the same requests the same way."""
    with TestClient(create_app(settings, overrides=overrides)) as client:
        r = client.get("/version", headers={**H, "X-Memory-User": "someone-else"})
        assert r.status_code == 422 and r.json()["details"] == {"field": "X-Trellis-User"}
        assert r.headers["X-Request-ID"] and r.headers["X-Trace-ID"]
        assert client.get("/version", headers={**H, "X-Memory-User": "u1"}).status_code == 200


def test_an_alias_route_is_marked_on_the_responses_the_middleware_writes(
    settings, overrides
) -> None:
    """Deprecation and Link come from the path, so a 413 written before routing, a 422 for a
    body FastAPI could not even parse, and a route's own Response all carry them."""
    from memory_service.api.headers import ALIASES_DEPRECATED_AT

    with TestClient(create_app(settings, overrides=overrides), raise_server_exceptions=False) as c:
        too_big = c.post("/v1/files", headers={**H, "Content-Length": str(10**9)}, content=b"")
        unparsable = c.post(
            "/v1/files",
            headers={**H, "content-type": "application/json"},
            content=b"{not json",
        )
        canonical = c.post(
            "/v1/documents",
            headers={**H, "content-type": "application/json"},
            content=b"{not json",
        )
    assert too_big.status_code == 413 and unparsable.status_code == 422
    for r in (too_big, unparsable):
        assert r.headers["Deprecation"] == ALIASES_DEPRECATED_AT
        assert r.headers["Link"].endswith('; rel="successor-version"')
    assert unparsable.headers["Link"] == '</v1/documents>; rel="successor-version"'
    assert canonical.status_code == 422 and "Deprecation" not in canonical.headers
    with TestClient(create_app(settings, overrides=overrides), raise_server_exceptions=False) as c:
        wrong_method = c.get("/v1/files", headers=H)
    assert wrong_method.status_code == 405 and "Deprecation" not in wrong_method.headers
