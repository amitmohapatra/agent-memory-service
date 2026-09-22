"""Typed error envelope and exception handlers.

Every error response has the shape::

    {"error": {"code": "SCOPE_DENIED", "message": "...", "retryable": false,
               "trace_id": "...", "details": {...}}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from memory_service.domain.enums import ErrorCode
from memory_service.domain.errors import MemoryServiceError
from memory_service.observability.logging import get_logger

log = get_logger(__name__)


class ErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ErrorCode = Field(..., description="Stable machine-readable category.")
    message: str = Field(..., description="Human-readable summary. Never contains source text.")
    retryable: bool = Field(..., description="Whether the same request may succeed if retried.")
    trace_id: str | None = Field(default=None, description="Correlates with logs/traces.")
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: ErrorBody


ERROR_EXAMPLES: dict[int, dict[str, Any]] = {
    401: {
        "code": "AUTHENTICATION",
        "message": "Missing or invalid credentials",
        "retryable": False,
    },
    403: {"code": "SCOPE_DENIED", "message": "Access denied", "retryable": False},
    404: {"code": "NOT_FOUND", "message": "Thread not found", "retryable": False},
    409: {
        "code": "CONFLICT",
        "message": "Idempotency key reused with a different payload",
        "retryable": False,
    },
    422: {"code": "VALIDATION", "message": "Request validation failed", "retryable": False},
    429: {"code": "RATE_LIMIT", "message": "Too many requests", "retryable": True},
    503: {"code": "DEPENDENCY_UNAVAILABLE", "message": "PostgreSQL unavailable", "retryable": True},
    504: {"code": "TIMEOUT", "message": "Operation exceeded its budget", "retryable": True},
}


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for the given statuses (used by every public route)."""
    out: dict[int | str, dict[str, Any]] = {}
    for status in statuses:
        example = ERROR_EXAMPLES.get(status, ERROR_EXAMPLES[503])
        out[status] = {
            "model": ErrorEnvelope,
            "description": example["message"],
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            **example,
                            "trace_id": "req_01J8ZZZZZZZZZZZZZZZZZZZZZZ",
                            "details": {},
                        }
                    }
                }
            },
        }
    return out


def _trace_id(request: Request) -> str | None:
    return getattr(request.state, "trace_id", None)


def _envelope(
    request: Request,
    *,
    code: ErrorCode,
    message: str,
    retryable: bool,
    status: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message,
            retryable=retryable,
            trace_id=_trace_id(request),
            details=details or {},
        )
    )
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(MemoryServiceError)
    async def _domain_error(request: Request, exc: MemoryServiceError) -> JSONResponse:
        if exc.http_status >= 500:
            log.error("request.failed", code=exc.code, message=exc.message, path=request.url.path)
        return _envelope(
            request,
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            status=exc.http_status,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return _envelope(
            request,
            code=ErrorCode.VALIDATION,
            message="Request validation failed",
            retryable=False,
            status=422,
            details={"errors": errors},
        )

    @app.exception_handler(ValidationError)
    async def _model_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        # A pydantic model validated inside a handler (a form field parsed by hand, a value
        # coerced into a domain model) is still the caller's input; it used to fall through
        # to the generic handler and come back as a 500 with no detail.
        errors = [
            {"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors(include_url=False, include_input=False)
        ]
        return _envelope(
            request,
            code=ErrorCode.VALIDATION,
            message="Request validation failed",
            retryable=False,
            status=422,
            details={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
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
        return _envelope(
            request,
            code=code,
            message=str(exc.detail),
            retryable=exc.status_code in (429, 503, 504),
            status=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("request.unhandled", path=request.url.path)
        return _envelope(
            request,
            code=ErrorCode.INTERNAL,
            message="Internal error",
            retryable=False,
            status=500,
        )
