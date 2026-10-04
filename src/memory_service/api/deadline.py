"""Per-request deadlines: a request that runs past its route class's budget is answered 504.

Without one, a request was bounded only by its client: a slow dependency (a model queue, a
saturated pool, a gateway taking its 30 s three times) held the connection, the pooled
database connection under it and the worker's attention for as long as the slowest thing it
waited on, and an overloaded pod accepted new work at the rate it shed none. The deadline
turns that into a fast, retryable answer: ``TIMEOUT`` problem details, status 504,
``retryable: true``, built by the same function every other problem is built with.

Route classes (``constants.OVERLOAD``): a *read* is a GET or one of the read-only POSTs
(``READ_POSTS``), a *verify* is ``/v1/verify`` (NLI plus a judge that has its own, shorter
deadline), everything else is a *write*. Uploads (``POST /v1/documents``) stream a body of up
to the upload limit over whatever the client's link is, and the probes and ``/metrics`` must
answer even when everything else is slow, so those carry no deadline here.

Plain ASGI, like the other middlewares (``api/middleware.py`` says why). The deadline covers
the application only after the response has not started: once the first byte is out, the
status is sent and cannot become a 504, so a late timeout then just ends the response.
"""

from __future__ import annotations

import asyncio
from typing import Final, Literal

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from memory_service.api.errors import build_problem, problem_response
from memory_service.api.headers import correlation_headers
from memory_service.config.constants import OverloadTuning
from memory_service.domain.enums import ErrorCode
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import request_deadline_exceeded_total

log = get_logger(__name__)

RouteClass = Literal["read", "write", "verify"]

#: POSTs that only read (a body carries the query, nothing is written)
READ_POSTS: Final = frozenset({"/v1/context", "/v1/recall", "/v1/tools/hints"})
VERIFY_PATHS: Final = frozenset({"/v1/verify"})
#: no deadline: the orchestrator's probes, the scrape, and uploads (a slow link is not a
#: slow service)
EXEMPT_PATHS: Final = frozenset({"/health/live", "/health/ready", "/metrics"})
UPLOAD_PATHS: Final = frozenset({"/v1/documents"})


def route_class(method: str, path: str) -> RouteClass | None:
    """The deadline class of a request, or ``None`` when it has no deadline."""
    if path in EXEMPT_PATHS or (method == "POST" and path in UPLOAD_PATHS):
        return None
    if path in VERIFY_PATHS:
        return "verify"
    if method in ("GET", "HEAD", "OPTIONS") or (method == "POST" and path in READ_POSTS):
        return "read"
    return "write"


class DeadlineMiddleware:
    def __init__(self, app: ASGIApp, *, limits: OverloadTuning) -> None:
        self.app = app
        self.budgets: dict[RouteClass, float] = {
            "read": limits.read_deadline_seconds,
            "write": limits.write_deadline_seconds,
            "verify": limits.verify_deadline_seconds,
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        kind = route_class(scope["method"], scope["path"])
        if kind is None:
            await self.app(scope, receive, send)
            return
        started = False

        async def tracked(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        budget = self.budgets[kind]
        deadline = asyncio.timeout(budget)
        try:
            async with deadline:
                await self.app(scope, receive, tracked)
        except TimeoutError:
            if not deadline.expired():
                raise  # a timeout of the application's own, not this deadline
            request_deadline_exceeded_total.labels(kind).inc()
            log.warning(
                "request.deadline_exceeded",
                path=scope["path"],
                route_class=kind,
                budget_seconds=budget,
                response_started=started,
            )
            if started:
                return
            state = scope.setdefault("state", {})
            response = problem_response(
                build_problem(
                    code=ErrorCode.TIMEOUT,
                    message=f"The request exceeded its {budget:g} s {kind} deadline",
                    status=504,
                    retryable=True,
                    instance=scope["path"],
                    trace_id=state.get("trace_id"),
                    request_id=state.get("request_id"),
                ),
                headers=correlation_headers(state),
            )
            await response(scope, receive, send)
