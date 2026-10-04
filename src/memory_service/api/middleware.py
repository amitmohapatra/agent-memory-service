"""Request middleware: ids and trace context, the body limit, rate limiting, the wire rules.

Both middlewares are plain ASGI callables rather than ``BaseHTTPMiddleware`` subclasses.
BaseHTTPMiddleware runs every request through an anyio task group with a memory-object
stream between the two halves, which buys a ``Request``/``Response`` API and costs a task,
two streams and a body round trip per request per middleware. At three workers and 20 rps
that is pure event-loop overhead on the path being measured.

``CorrelationMiddleware`` resolves the request, correlation and trace ids (W3C
``traceparent`` in and out, ADR 0022), refuses a request whose scope or credential headers
carry more than one value before anything reads them, answers the 413 before the body is
read (from Content-Length) or as it streams past the limit, and writes the id headers and
the LLM-token header on every response it sees. ``RateLimitMiddleware`` keeps the per-tenant
window with its burst, fails open on a cache outage, and answers the 429 with its headers.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from memory_service.api.errors import build_problem, problem_response
from memory_service.api.headers import (
    AUTHORIZATION_HEADER,
    CORRELATION_ID_HEADER,
    RATE_LIMIT_LIMIT_HEADER,
    RATE_LIMIT_REMAINING_HEADER,
    REQUEST_ID_HEADER,
    RETRY_AFTER_HEADER,
    correlation_headers,
    idempotency_key_of,
    refuse_ambiguous_headers,
    scope_header,
)
from memory_service.config.constants import HEADERS
from memory_service.domain.enums import ErrorCode
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.ids import is_valid_id, new_id
from memory_service.domain.tenancy import bare_credential
from memory_service.modules.llm.cost import LLMTokens, llm_accounting
from memory_service.observability.logging import bind_log_context, clear_log_context, get_logger
from memory_service.observability.metrics import http_request_seconds, http_requests_total
from memory_service.observability.tracing import (
    TRACEPARENT_HEADER,
    current_span_id,
    current_trace_flags,
    current_trace_id,
    format_traceparent,
    new_span_id,
    new_trace_id,
    parse_traceparent,
)

#: never counted, never logged, never rate limited: the probes an orchestrator runs
QUIET_PATHS = ("/health/live", "/health/ready", "/metrics")

log = get_logger("memory_service.http")


def _header_id(headers: Headers, name: str, kind: str) -> str:
    value = headers.get(name)
    if value and is_valid_id(value):
        return value
    return new_id(kind)


def _trace_context(headers: Headers) -> tuple[str, str]:
    """The request's trace id and the ``traceparent`` the response will carry.

    The exported trace wins when this process is tracing (its server span already continued
    an incoming ``traceparent``); otherwise the caller's ``traceparent``, else a fresh id.
    Never the request id, and never ``X-Trace-ID``, which is a response header: the one
    request header that names a trace is the one the OpenTelemetry propagator reads too, and
    it is read the way the propagator reads it (the first when a proxy sent two), so dev and
    production behave the same (ADR 0022).
    """
    active = current_trace_id()
    parent = parse_traceparent(headers.get(TRACEPARENT_HEADER))
    if active:
        trace_id, flags = active, current_trace_flags()
    elif parent:
        trace_id, flags = parent.trace_id, parent.flags
    else:
        trace_id, flags = new_trace_id(), "00"
    return trace_id, format_traceparent(trace_id, current_span_id() or new_span_id(), flags)


class CorrelationMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        request_id = _header_id(headers, REQUEST_ID_HEADER, "request")
        correlation_id = _header_id(headers, CORRELATION_ID_HEADER, "request")
        trace_id, traceparent = _trace_context(headers)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["correlation_id"] = correlation_id
        state["trace_id"] = trace_id
        state["traceparent"] = traceparent
        state["idempotency_key"] = None
        path = scope["path"]

        content_length = headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > self.max_body_bytes
        ):
            response = problem_response(
                build_problem(
                    code=ErrorCode.PAYLOAD_TOO_LARGE,
                    message=f"Body exceeds {self.max_body_bytes} bytes",
                    status=413,
                    retryable=False,
                    instance=path,
                    trace_id=trace_id,
                    request_id=request_id,
                ),
                headers=correlation_headers(state),
            )
            await response(scope, receive, send)
            return
        try:
            refuse_ambiguous_headers(headers)
            state["idempotency_key"] = idempotency_key_of(headers)
        except ValidationFailed as exc:
            # Before the credential is verified and before any bucket is touched: a client
            # must not name a tenant per request with values the context builder is about to
            # refuse, and the limiter and the authenticator must key on one credential.
            response = problem_response(
                build_problem(
                    code=exc.code,
                    message=exc.message,
                    status=exc.http_status,
                    retryable=exc.retryable,
                    instance=path,
                    trace_id=trace_id,
                    request_id=request_id,
                    details=exc.details,
                ),
                headers=correlation_headers(state),
            )
            await response(scope, receive, send)
            return

        clear_log_context()
        bind_log_context(request_id=request_id, trace_id=trace_id, correlation_id=correlation_id)
        try:
            with llm_accounting() as llm_tokens:
                await self._serve(scope, self._bounded(receive), send, state, llm_tokens)
        finally:
            clear_log_context()

    def _bounded(self, receive: Receive) -> Receive:
        """``receive`` counting the body as it streams in: a chunked request, or one whose
        Content-Length understates it, is stopped at the limit with the same 413 problem the
        Content-Length check answers, before the route has buffered past it. The exception is
        the HTTP one FastAPI's body reader re-raises (anything else it turns into a 400)."""
        seen = 0

        async def bounded() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_body_bytes:
                    raise StarletteHTTPException(
                        status_code=413, detail=f"Body exceeds {self.max_body_bytes} bytes"
                    )
            return message

        return bounded

    async def _serve(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        state: dict[str, Any],
        llm_tokens: LLMTokens,
    ) -> None:
        """The request inside its LLM accounting scope: metrics, logs and response headers."""
        started = time.perf_counter()
        recorded = False

        def record(status: int) -> None:
            nonlocal recorded
            recorded = True
            elapsed = time.perf_counter() - started
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            http_requests_total.labels(scope["method"], route, str(status)).inc()
            http_request_seconds.labels(scope["method"], route).observe(elapsed)
            if route not in QUIET_PATHS:
                log.info(
                    "request.completed",
                    method=scope["method"],
                    route=route,
                    status=status,
                    duration_ms=round(elapsed * 1000, 2),
                )

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                # from the state, not the locals: a body may have named the correlation id
                # (build_context writes the effective one back), and the response must echo
                # the id the logs carry
                response_headers.update(correlation_headers(state))
                if llm_tokens.total:
                    response_headers[HEADERS.llm_tokens] = str(llm_tokens.total)
                record(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as exc:
            # Nothing answered, so the count is the one the error handler will produce.
            #
            # BaseException rather than Exception because CancelledError is not an
            # Exception: a client that disconnects and a request the server timed out are
            # both cancellations, and catching only Exception drops them out of
            # http_requests_total entirely - which is the one population a run at the target
            # rate is looking for. They are counted as 499 (client closed request) rather
            # than 500: nothing on this side failed, and a gate that reads failures off this
            # counter should not be told one did.
            if not recorded:
                record(499 if isinstance(exc, asyncio.CancelledError) else 500)
            raise


class RateLimitMiddleware:
    """Per-tenant (and per-API-key) request budget: a fixed one-minute window counted in
    the shared cache, so every instance of the service enforces the same budget. Health,
    readiness and metrics are exempt. A cache outage fails *open* for this middleware —
    losing the cache must degrade rate limiting, never availability — and is logged once
    per window. The limit is a hardening measure against runaway clients, not a security
    boundary (authorization is).

    The count and its expiry are one round trip: the cache pipelines them. Two commands
    meant two network waits per request against a remote Dragonfly, on every request that
    was not the first of a window.
    """

    def __init__(self, app: ASGIApp, *, per_minute: int, burst: int) -> None:
        self.app = app
        self.per_minute = per_minute
        self.burst = burst
        self._warned_window = -1

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in QUIET_PATHS:
            await self.app(scope, receive, send)
            return
        container = getattr(getattr(scope.get("app"), "state", None), "container", None)
        cache = getattr(container, "cache", None)
        if cache is None:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        api_key = bare_credential(
            headers.get(HEADERS.api_key) or headers.get(AUTHORIZATION_HEADER) or "-"
        )
        # A tenant's own quota, when the platform set one: read from this process, never
        # from a store, so the override costs the request nothing (modules/tenancy/registry.py).
        # The tenant is the header's, or the one the caller's key names. A tenant quota
        # applies even where the service default is off.
        registry = getattr(container, "services", {}).get("tenant_registry")
        named = scope_header(headers, HEADERS.tenant)
        if registry is not None:
            tenant, per_minute = registry.quota_for(named, api_key)
        else:
            tenant, per_minute = named or "-", None
        if per_minute is None:
            per_minute = self.per_minute
        if per_minute <= 0:
            await self.app(scope, receive, send)
            return
        window = int(time.time() // 60)
        key = f"ratelimit:{tenant}:{hash_key(api_key)}:{window}"
        try:
            count = await cache.incr_window(key, ttl_seconds=120)
        except Exception as exc:
            if self._warned_window != window:
                self._warned_window = window
                log.warning("ratelimit.cache_unavailable", error=str(exc))
            await self.app(scope, receive, send)
            return
        limit = per_minute + self.burst
        if count > limit:
            retry_after = 60 - int(time.time() % 60)
            state = scope.get("state", {})
            response = problem_response(
                build_problem(
                    code=ErrorCode.RATE_LIMIT,
                    message="Too many requests for this tenant; retry after the window",
                    status=429,
                    retryable=True,
                    instance=scope["path"],
                    trace_id=state.get("trace_id"),
                    request_id=state.get("request_id"),
                    details={"limit_per_minute": per_minute, "window_seconds": 60},
                ),
                headers={
                    RETRY_AFTER_HEADER: str(retry_after),
                    RATE_LIMIT_LIMIT_HEADER: str(per_minute),
                    RATE_LIMIT_REMAINING_HEADER: "0",
                },
            )
            await response(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers[RATE_LIMIT_LIMIT_HEADER] = str(per_minute)
                response_headers[RATE_LIMIT_REMAINING_HEADER] = str(max(0, limit - count))
            await send(message)

        await self.app(scope, receive, send_wrapper)


def hash_key(value: str) -> str:
    import hashlib

    return hashlib.blake2b(value.encode(), digest_size=8).hexdigest()
