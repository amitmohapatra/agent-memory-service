"""SDK exceptions mapped from the API's RFC 9457 problem details."""

from __future__ import annotations

from typing import Any


class MemoryError(Exception):
    """Base SDK error. ``code`` mirrors the API error code; ``retryable`` tells you what to do;
    ``trace_id`` and ``request_id`` are what to quote to whoever runs the service."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "INTERNAL",
        status: int = 500,
        retryable: bool = False,
        trace_id: str | None = None,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.retryable = retryable
        self.trace_id = trace_id
        self.request_id = request_id
        self.details = details or {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code}, status={self.status}, "
            f"message={self.message!r})"
        )


class AuthenticationError(MemoryError):
    pass


class AuthorizationError(MemoryError):
    pass


class NotFoundError(MemoryError):
    pass


class ConflictError(MemoryError):
    pass


class ValidationError(MemoryError):
    pass


class PayloadTooLargeError(ValidationError):
    """413: the body or the file is larger than the service accepts; never retryable as is."""


class RateLimitedError(MemoryError):
    pass


class DependencyUnavailableError(MemoryError):
    pass


class TimeoutError(MemoryError):
    pass


_BY_CODE: dict[str, type[MemoryError]] = {
    "AUTHENTICATION": AuthenticationError,
    "AUTHORIZATION": AuthorizationError,
    "SCOPE_DENIED": AuthorizationError,
    "NOT_FOUND": NotFoundError,
    "CONFLICT": ConflictError,
    "VALIDATION": ValidationError,
    "PAYLOAD_TOO_LARGE": PayloadTooLargeError,
    "RATE_LIMIT": RateLimitedError,
    "DEPENDENCY_UNAVAILABLE": DependencyUnavailableError,
    "RETRYABLE_PROCESSING": DependencyUnavailableError,
    "TIMEOUT": TimeoutError,
}


#: What makes a dict an RFC 9457 problem rather than some other JSON body.
_PROBLEM_MEMBERS = frozenset({"code", "type", "title", "detail"})


def error_from_problem(status: int, body: Any) -> MemoryError:
    """The typed exception for an error response.

    The service answers RFC 9457 problems: ``code`` and ``detail`` at the top level. A plain
    problem from a gateway in front of it (``type``/``title``/``detail``, no ``code``) keeps
    its words, and a 0.1 server's nested ``{"error": {"code", "message", ...}}`` envelope is
    read the same way, so an SDK ahead of its service still raises the right class.
    """
    problem: dict[str, Any] = {}
    if isinstance(body, dict):
        nested = body.get("error")
        if body.keys() & _PROBLEM_MEMBERS:
            problem = body
        elif isinstance(nested, dict):
            problem = nested
        else:
            # a gateway's own shape: whatever it calls message or request_id is kept
            problem = body
    code = str(problem.get("code") or ("PAYLOAD_TOO_LARGE" if status == 413 else "INTERNAL"))
    cls = _BY_CODE.get(code, MemoryError)
    # the members are typed here, at the boundary: a gateway's body is not the service's
    retryable = problem.get("retryable")
    details = problem.get("details")
    return cls(
        str(
            problem.get("detail")
            or problem.get("message")
            or problem.get("title")
            or f"HTTP {status}"
        ),
        code=code,
        status=status,
        retryable=retryable if isinstance(retryable, bool) else status in (429, 503, 504),
        trace_id=_text(problem.get("trace_id")),
        request_id=_text(problem.get("request_id")),
        details=details if isinstance(details, dict) else {},
    )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
