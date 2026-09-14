"""Request middleware: correlation headers, timing, metrics, log context, body limit."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from memory_service.domain.ids import is_valid_id, new_id
from memory_service.observability.logging import bind_log_context, clear_log_context, get_logger
from memory_service.observability.metrics import http_request_seconds, http_requests_total
from memory_service.observability.tracing import current_trace_id

HEADER_REQUEST_ID = "X-Request-ID"
HEADER_TRACE_ID = "X-Trace-ID"
HEADER_CORRELATION_ID = "X-Correlation-ID"
HEADER_IDEMPOTENCY_KEY = "Idempotency-Key"

log = get_logger("memory_service.http")


def _header_id(request: Request, name: str, kind: str) -> str:
    value = request.headers.get(name)
    if value and is_valid_id(value):
        return value
    return new_id(kind)


class CorrelationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, max_body_bytes: int) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = _header_id(request, HEADER_REQUEST_ID, "request")
        correlation_id = _header_id(request, HEADER_CORRELATION_ID, "request")
        trace_id = request.headers.get(HEADER_TRACE_ID) or current_trace_id() or request_id
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        request.state.trace_id = trace_id
        request.state.idempotency_key = request.headers.get(HEADER_IDEMPOTENCY_KEY)

        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > self.max_body_bytes
        ):
            return JSONResponse(
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

        clear_log_context()
        bind_log_context(request_id=request_id, trace_id=trace_id, correlation_id=correlation_id)
        started = time.perf_counter()
        route = "unmatched"
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - started
            route_obj = request.scope.get("route")
            route = getattr(route_obj, "path", None) or route
            http_requests_total.labels(request.method, route, str(status)).inc()
            http_request_seconds.labels(request.method, route).observe(elapsed)
            if route not in ("/health/live", "/health/ready", "/metrics"):
                log.info(
                    "request.completed",
                    method=request.method,
                    route=route,
                    status=status,
                    duration_ms=round(elapsed * 1000, 2),
                )
            clear_log_context()
        response.headers[HEADER_REQUEST_ID] = request_id
        response.headers[HEADER_TRACE_ID] = trace_id
        response.headers[HEADER_CORRELATION_ID] = correlation_id
        return response
