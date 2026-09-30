"""RFC 9457 problem details and the exception handlers that produce them.

Every error response is ``application/problem+json`` with this shape::

    {"type": "urn:trellis:problem:scope-denied", "title": "Outside the caller's scope",
     "status": 403, "detail": "...", "instance": "/v1/recall",
     "code": "SCOPE_DENIED", "retryable": false, "trace_id": "...", "request_id": "...",
     "details": {...}}

``code`` is the stable machine-readable category (``ErrorCode``) and ``type`` is its URN; the
rest are the standard members plus the extensions a client acts on. One builder serves the
handlers below and the two responses the middleware writes before a route runs (413 and
429), so there is exactly one error shape on the wire (ADR 0022).
"""

from __future__ import annotations

from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from memory_service.api.headers import correlation_headers
from memory_service.config.constants import MAX_BODY_BYTES
from memory_service.domain.enums import ErrorCode
from memory_service.domain.errors import MemoryServiceError
from memory_service.observability.logging import get_logger

log = get_logger(__name__)

PROBLEM_MEDIA_TYPE: Final = "application/problem+json"
PROBLEM_TYPE_PREFIX: Final = "urn:trellis:problem:"

#: One title per category: the same words for every occurrence, as RFC 9457 asks.
PROBLEM_TITLES: Final[dict[ErrorCode, str]] = {
    ErrorCode.VALIDATION: "Invalid request",
    ErrorCode.AUTHENTICATION: "Authentication required",
    ErrorCode.AUTHORIZATION: "Not permitted",
    ErrorCode.SCOPE_DENIED: "Outside the caller's scope",
    ErrorCode.NOT_FOUND: "Not found",
    ErrorCode.CONFLICT: "Conflict",
    ErrorCode.RATE_LIMIT: "Too many requests",
    ErrorCode.DEPENDENCY_UNAVAILABLE: "A dependency is unavailable",
    ErrorCode.TIMEOUT: "Operation timed out",
    ErrorCode.RETRYABLE_PROCESSING: "Processing did not complete",
    ErrorCode.CORRUPT_SOURCE: "Source could not be processed",
    ErrorCode.INTERNAL: "Internal error",
}


def problem_type(code: ErrorCode) -> str:
    """``urn:trellis:problem:<code in kebab case>``: a stable identifier, not a URL to fetch."""
    return PROBLEM_TYPE_PREFIX + code.value.lower().replace("_", "-")


class Problem(BaseModel):
    """RFC 9457 problem details with the service's extension members."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(
        ...,
        description="URN of the category: urn:trellis:problem:<code in kebab case>.",
        examples=["urn:trellis:problem:scope-denied"],
    )
    title: str = Field(
        ...,
        description="Short summary of the category; the same for every occurrence.",
        examples=["Outside the caller's scope"],
    )
    status: int = Field(..., description="The HTTP status code.", examples=[403])
    detail: str = Field(
        ...,
        description="What went wrong in this occurrence. Never contains source text.",
        examples=["thread thr_01J... is outside the caller's scope"],
    )
    instance: str = Field(..., description="The request path.", examples=["/v1/recall"])
    code: ErrorCode = Field(..., description="Stable machine-readable category.")
    retryable: bool = Field(..., description="Whether the same request may succeed if retried.")
    trace_id: str | None = Field(
        default=None,
        description="W3C trace id (32 hex) shared with the response's traceparent and X-Trace-ID.",
    )
    request_id: str | None = Field(
        default=None, description="The request's X-Request-ID, for support tickets and logs."
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Category-specific structured context."
    )


class ProblemResponse(JSONResponse):
    media_type = PROBLEM_MEDIA_TYPE


PROBLEM_SCHEMA_REF: Final = f"#/components/schemas/{Problem.__name__}"


def build_problem(
    *,
    code: ErrorCode,
    message: str,
    status: int,
    retryable: bool,
    instance: str,
    trace_id: str | None,
    request_id: str | None,
    details: dict[str, Any] | None = None,
) -> Problem:
    return Problem(
        type=problem_type(code),
        title=PROBLEM_TITLES[code],
        status=status,
        detail=message,
        instance=instance,
        code=code,
        retryable=retryable,
        trace_id=trace_id,
        request_id=request_id,
        details=details or {},
    )


def problem_response(problem: Problem, *, headers: dict[str, str] | None = None) -> ProblemResponse:
    return ProblemResponse(
        status_code=problem.status, content=problem.model_dump(mode="json"), headers=headers
    )


ERROR_EXAMPLES: dict[int, dict[str, Any]] = {
    401: {
        "code": ErrorCode.AUTHENTICATION,
        "detail": "Missing or invalid credentials",
        "retryable": False,
    },
    403: {
        "code": ErrorCode.SCOPE_DENIED,
        "detail": "Access denied (SCOPE_DENIED for an object outside the caller's scope, "
        "AUTHORIZATION for a refused role)",
        "retryable": False,
    },
    404: {"code": ErrorCode.NOT_FOUND, "detail": "Thread not found", "retryable": False},
    409: {
        "code": ErrorCode.CONFLICT,
        "detail": "Idempotency key reused with a different payload",
        "retryable": False,
    },
    413: {
        "code": ErrorCode.VALIDATION,
        "detail": f"Body exceeds {MAX_BODY_BYTES} bytes",
        "retryable": False,
    },
    422: {"code": ErrorCode.VALIDATION, "detail": "Request validation failed", "retryable": False},
    429: {"code": ErrorCode.RATE_LIMIT, "detail": "Too many requests", "retryable": True},
    503: {
        "code": ErrorCode.DEPENDENCY_UNAVAILABLE,
        "detail": "PostgreSQL unavailable",
        "retryable": True,
    },
    504: {"code": ErrorCode.TIMEOUT, "detail": "Operation exceeded its budget", "retryable": True},
}


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for the given statuses (used by every public route)."""
    out: dict[int | str, dict[str, Any]] = {}
    for status in statuses:
        example = ERROR_EXAMPLES.get(status, ERROR_EXAMPLES[503])
        problem = build_problem(
            code=example["code"],
            message=example["detail"],
            status=status,
            retryable=example["retryable"],
            instance="/v1/recall",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            request_id="req_01J8ZZZZZZZZZZZZZZZZZZZZZZ",
        )
        out[status] = {
            "description": example["detail"],
            "content": {
                PROBLEM_MEDIA_TYPE: {
                    "schema": {"$ref": PROBLEM_SCHEMA_REF},
                    "example": problem.model_dump(mode="json"),
                }
            },
        }
    return out


def _problem(
    request: Request,
    *,
    code: ErrorCode,
    message: str,
    retryable: bool,
    status: int,
    details: dict[str, Any] | None = None,
) -> ProblemResponse:
    return problem_response(
        build_problem(
            code=code,
            message=message,
            status=status,
            retryable=retryable,
            instance=request.url.path,
            trace_id=getattr(request.state, "trace_id", None),
            request_id=getattr(request.state, "request_id", None),
            details=details,
        )
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(MemoryServiceError)
    async def _domain_error(request: Request, exc: MemoryServiceError) -> ProblemResponse:
        if exc.http_status >= 500:
            log.error("request.failed", code=exc.code, message=exc.message, path=request.url.path)
        return _problem(
            request,
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            status=exc.http_status,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> ProblemResponse:
        errors = [
            {"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return _problem(
            request,
            code=ErrorCode.VALIDATION,
            message="Request validation failed",
            retryable=False,
            status=422,
            details={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> ProblemResponse:
        mapping = {
            401: ErrorCode.AUTHENTICATION,
            403: ErrorCode.AUTHORIZATION,
            404: ErrorCode.NOT_FOUND,
            405: ErrorCode.VALIDATION,
            409: ErrorCode.CONFLICT,
            413: ErrorCode.VALIDATION,
            422: ErrorCode.VALIDATION,
            429: ErrorCode.RATE_LIMIT,
        }
        code = mapping.get(
            exc.status_code, ErrorCode.INTERNAL if exc.status_code >= 500 else ErrorCode.VALIDATION
        )
        return _problem(
            request,
            code=code,
            message=str(exc.detail),
            retryable=exc.status_code in (429, 503, 504),
            status=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> ProblemResponse:
        # Starlette's outermost ServerErrorMiddleware runs this handler after the correlation
        # middleware has cleared the log context and outside its response wrapper, so the ids
        # it kept in the request state go into the log line and the headers by hand.
        state = request.scope.get("state", {})
        log.exception(
            "request.unhandled",
            path=request.url.path,
            request_id=state.get("request_id"),
            trace_id=state.get("trace_id"),
            correlation_id=state.get("correlation_id"),
        )
        response = _problem(
            request,
            code=ErrorCode.INTERNAL,
            message="Internal error",
            retryable=False,
            status=500,
        )
        response.headers.update(correlation_headers(state))
        return response
