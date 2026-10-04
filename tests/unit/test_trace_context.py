"""W3C Trace Context on the way in and out (ADR 0022).

Every response names a 32-hex trace id in ``X-Trace-ID`` and ``traceparent``; an incoming
``traceparent`` is continued, so an agent's trace and the service's trace are one trace.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from starlette.datastructures import Headers

from memory_service.api.app import create_app
from memory_service.api.middleware import _trace_context
from memory_service.observability.tracing import TraceParent, format_traceparent, parse_traceparent

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN = "00f067aa0ba902b7"
TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (f"00-{TRACE}-{SPAN}-01", TraceParent(TRACE, SPAN, "01")),
        (f"  00-{TRACE}-{SPAN}-00  ", TraceParent(TRACE, SPAN, "00")),
        # a future version may carry more fields; version 00 may not
        (f"01-{TRACE}-{SPAN}-01-extra", TraceParent(TRACE, SPAN, "01")),
        (f"00-{TRACE}-{SPAN}-01-extra", None),
        (f"ff-{TRACE}-{SPAN}-01", None),
        (f"00-{'0' * 32}-{SPAN}-01", None),
        (f"00-{TRACE}-{'0' * 16}-01", None),
        (f"00-{TRACE.upper()}-{SPAN}-01", None),
        (f"00-{TRACE[:-1]}-{SPAN}-01", None),
        ("", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_parse_traceparent(value: str | None, expected: TraceParent | None) -> None:
    assert parse_traceparent(value) == expected


def test_format_round_trips() -> None:
    assert parse_traceparent(format_traceparent(TRACE, SPAN, "01")) == TraceParent(
        TRACE, SPAN, "01"
    )


def test_precedence_without_an_active_span() -> None:
    incoming = Headers({"traceparent": f"00-{TRACE}-{SPAN}-01", "X-Trace-ID": "a" * 32})
    trace_id, traceparent = _trace_context(incoming)
    assert trace_id == TRACE
    version, parent_trace, parent_span, flags = traceparent.split("-")
    assert (version, parent_trace, flags) == ("00", TRACE, "01") and parent_span != SPAN

    # X-Trace-ID is a response header: the OpenTelemetry propagator does not read it, so a
    # request that relied on it would behave differently once tracing is on. Fresh id.
    trace_id, traceparent = _trace_context(Headers({"X-Trace-ID": "a" * 32}))
    assert trace_id != "a" * 32 and TRACEPARENT.match(traceparent) and traceparent.endswith("-00")

    # a proxy that sent two: the first is continued, as the OpenTelemetry propagator does
    other = f"00-{'b' * 32}-{SPAN}-01".encode()
    twice = Headers(
        raw=[(b"traceparent", f"00-{TRACE}-{SPAN}-01".encode()), (b"traceparent", other)]
    )
    trace_id, _ = _trace_context(twice)
    assert trace_id == TRACE


def test_the_active_span_wins_when_the_process_is_tracing() -> None:
    """With OpenTelemetry instrumenting the app, the server span has already continued the
    incoming traceparent; the exported trace is what the client must be told."""
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("server") as span:
        trace_id, traceparent = _trace_context(Headers({"traceparent": f"00-{TRACE}-{SPAN}-01"}))
    context = span.get_span_context()
    assert trace_id == format(context.trace_id, "032x") != TRACE
    # the span's own flags: sampled, plus the random-trace-id flag newer SDKs set
    flags = format(int(context.trace_flags), "02x")
    assert traceparent == f"00-{trace_id}-{format(context.span_id, '016x')}-{flags}"


def test_a_request_with_a_traceparent_is_answered_in_that_trace(settings, overrides) -> None:
    with TestClient(create_app(settings, overrides=overrides)) as client:
        r = client.get("/version", headers={"traceparent": f"00-{TRACE}-{SPAN}-01"})
    assert r.headers["X-Trace-ID"] == TRACE
    match = TRACEPARENT.match(r.headers["traceparent"])
    assert match and match.group(1) == TRACE and match.group(2) != SPAN


def test_a_request_without_one_gets_a_fresh_32_hex_trace(settings, overrides) -> None:
    with TestClient(create_app(settings, overrides=overrides)) as client:
        r = client.get("/version")
        again = client.get("/version", headers={"traceparent": "00-nonsense"})
    for response in (r, again):
        trace_id = response.headers["X-Trace-ID"]
        assert (
            re.fullmatch(r"[0-9a-f]{32}", trace_id) and trace_id != response.headers["X-Request-ID"]
        )
        match = TRACEPARENT.match(response.headers["traceparent"])
        assert match and match.group(1) == trace_id
    assert r.headers["X-Trace-ID"] != again.headers["X-Trace-ID"]


def test_the_early_413_carries_the_trace_headers_too(settings, overrides) -> None:
    with TestClient(create_app(settings, overrides=overrides)) as client:
        r = client.post(
            "/version",
            headers={"Content-Length": str(10**9), "traceparent": f"00-{TRACE}-{SPAN}-01"},
            content=b"",
        )
    assert r.status_code == 413 and r.headers["X-Trace-ID"] == TRACE
    assert r.headers["content-type"] == "application/problem+json"
    assert r.headers["traceparent"].startswith(f"00-{TRACE}-") and r.headers["X-Request-ID"]
    body = r.json()
    assert body["trace_id"] == TRACE and body["request_id"] == r.headers["X-Request-ID"]
    assert body["type"] == "urn:trellis:problem:payload-too-large"
    assert body["title"] == "Payload too large" and body["code"] == "PAYLOAD_TOO_LARGE"
    assert body["status"] == 413 and body["instance"] == "/version"


def test_an_unhandled_error_still_carries_the_ids(settings, overrides) -> None:
    """The 500 is written by Starlette's outermost error middleware, outside the correlation
    middleware; the handler copies the ids so a client can still quote them."""
    from fastapi import APIRouter

    app = create_app(settings, overrides=overrides)
    router = APIRouter()

    @router.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret internal detail")

    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get(
            "/boom", headers={"X-Request-ID": "req_boom", "traceparent": f"00-{TRACE}-{SPAN}-01"}
        )
    assert r.status_code == 500 and r.headers["content-type"] == "application/problem+json"
    assert r.headers["X-Request-ID"] == "req_boom" and r.headers["X-Trace-ID"] == TRACE
    assert r.headers["traceparent"].startswith(f"00-{TRACE}-") and r.headers["X-Correlation-ID"]
    assert r.json()["request_id"] == "req_boom" and r.json()["trace_id"] == TRACE
    assert "secret internal detail" not in r.text


def test_a_request_x_trace_id_is_a_response_header_and_not_continued(settings, overrides) -> None:
    with TestClient(create_app(settings, overrides=overrides)) as client:
        r = client.get("/version", headers={"X-Trace-ID": "a" * 32})
        schema = client.get("/openapi.json").json()
    assert r.headers["X-Trace-ID"] != "a" * 32 and re.fullmatch(
        r"[0-9a-f]{32}", r.headers["X-Trace-ID"]
    )
    for methods in schema["paths"].values():
        for op in methods.values():
            assert "X-Trace-ID" not in {p["name"] for p in op.get("parameters", [])}


def test_an_unhandled_error_is_answered_and_logged_with_its_ids(settings, overrides) -> None:
    from fastapi import APIRouter

    app = create_app(settings, overrides=overrides)
    router = APIRouter()

    @router.get("/old-boom")
    async def boom() -> None:
        raise RuntimeError("x")

    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/old-boom", headers={"X-Request-ID": "req_boom"})
    assert r.status_code == 500 and r.headers["content-type"] == "application/problem+json"
    assert "Deprecation" not in r.headers
    assert r.headers["X-Request-ID"] == "req_boom" == r.json()["request_id"]
