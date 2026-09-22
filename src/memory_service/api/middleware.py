"""Request middleware: correlation headers, timing, metrics, log context, body limit.

Both middlewares are plain ASGI callables rather than ``BaseHTTPMiddleware`` subclasses.
BaseHTTPMiddleware runs every request through an anyio task group with a memory-object
stream between the two halves, which buys a ``Request``/``Response`` API and costs a task,
two streams and a body round trip per request per middleware. At three workers and 20 rps
that is pure event-loop overhead on the path being measured. The semantics below are the
ones that were there before, to the header: the same ids, the same 413 envelope before the
body is read, the same per-tenant window with the same burst, fail-open on a cache outage,
the same 429 body and headers, and the same exemptions.
"""

from __future__ import annotations

import time

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from memory_service.domain.ids import is_valid_id, new_id
from memory_service.modules.llm.cost import start_llm_accounting
from memory_service.observability.logging import bind_log_context, clear_log_context, get_logger
from memory_service.observability.metrics import http_request_seconds, http_requests_total
from memory_service.observability.tracing import current_trace_id

HEADER_REQUEST_ID = "X-Request-ID"
HEADER_TRACE_ID = "X-Trace-ID"
HEADER_CORRELATION_ID = "X-Correlation-ID"
HEADER_IDEMPOTENCY_KEY = "Idempotency-Key"
HEADER_LLM_TOKENS = "X-Memory-LLM-Tokens"

#: never counted, never logged, never rate limited: the probes an orchestrator runs
QUIET_PATHS = ("/health/live", "/health/ready", "/metrics")

log = get_logger("memory_service.http")


def _header_id(headers: Headers, name: str, kind: str) -> str:
    value = headers.get(name)
    if value and is_valid_id(value):
        return value
    return new_id(kind)


class CorrelationMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        request_id = _header_id(headers, HEADER_REQUEST_ID, "request")
        correlation_id = _header_id(headers, HEADER_CORRELATION_ID, "request")
        trace_id = headers.get(HEADER_TRACE_ID) or current_trace_id() or request_id
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["correlation_id"] = correlation_id
        state["trace_id"] = trace_id
        state["idempotency_key"] = headers.get(HEADER_IDEMPOTENCY_KEY)

        content_length = headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > self.max_body_bytes
        ):
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "VALIDATION",
                        "message": f"Body exceeds {self.max_body_bytes} bytes",
                        "retryable": False,
                        "trace_id": trace_id,
                        "details": {},
                    }
                },
            )
            await response(scope, receive, send)
            return

        clear_log_context()
        bind_log_context(request_id=request_id, trace_id=trace_id, correlation_id=correlation_id)
        llm_tokens = start_llm_accounting()
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
                response_headers[HEADER_REQUEST_ID] = request_id
                response_headers[HEADER_TRACE_ID] = trace_id
                response_headers[HEADER_CORRELATION_ID] = correlation_id
                if llm_tokens.total:
                    response_headers[HEADER_LLM_TOKENS] = str(llm_tokens.total)
                record(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Nothing answered, so the count is the one the error handler will produce.
            if not recorded:
                record(500)
            raise
        finally:
            clear_log_context()


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
        if scope["type"] != "http" or self.per_minute <= 0 or scope["path"] in QUIET_PATHS:
            await self.app(scope, receive, send)
            return
        cache = getattr(getattr(scope.get("app"), "state", None), "container", None)
        cache = getattr(cache, "cache", None)
        if cache is None:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        tenant = headers.get("x-memory-tenant") or "-"
        api_key = headers.get("x-api-key") or headers.get("authorization") or "-"
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
        limit = self.per_minute + self.burst
        if count > limit:
            retry_after = 60 - int(time.time() % 60)
            response = JSONResponse(
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.per_minute),
                    "X-RateLimit-Remaining": "0",
                },
                content={
                    "error": {
                        "code": "RATE_LIMIT",
                        "message": "Too many requests for this tenant; retry after the window",
                        "retryable": True,
                        "trace_id": scope.get("state", {}).get("trace_id", ""),
                        "details": {"limit_per_minute": self.per_minute, "window_seconds": 60},
                    }
                },
            )
            await response(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers["X-RateLimit-Limit"] = str(self.per_minute)
                response_headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
            await send(message)

        await self.app(scope, receive, send_wrapper)


def hash_key(value: str) -> str:
    import hashlib

    return hashlib.blake2b(value.encode(), digest_size=8).hexdigest()
