"""SDK exceptions mapped from the API's RFC 9457 problem details."""

from __future__ import annotations

from typing import Any


class MemoryError(Exception):
    """Base SDK error. ``code`` mirrors the API error code; ``retryable`` tells you what to do;
    ``retry_after`` is the seconds the service asked for (``Retry-After``), when it said;
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
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.retryable = retryable
        self.trace_id = trace_id
        self.request_id = request_id
        self.details = details or {}
        self.retry_after = retry_after

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


class CircuitOpenError(DependencyUnavailableError):
    """The client's circuit breaker is open: recent calls failed for want of the service, so
    this one was not sent. ``retry_after`` is how long is left on the open circuit.

    A :class:`DependencyUnavailableError` (and retryable), so code that degrades on an
    unavailable service degrades on this too - at once, instead of after the timeout."""


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
    "TIMEOUT": TimeoutError,
    "CORRUPT_SOURCE": ValidationError,
    "CIRCUIT_OPEN": CircuitOpenError,
}


#: The class, code and retryability an error status stands for when its body names no
#: ``code`` - a gateway, a proxy or a load balancer answered instead of the service. 504 is
#: the service's own TIMEOUT; 500 stays the base class and is not retried (the request may
#: have done its work), while 502 and 503 say the service is not there to ask.
_BY_STATUS: dict[int, tuple[type[MemoryError], str, bool]] = {
    400: (ValidationError, "VALIDATION", False),
    401: (AuthenticationError, "AUTHENTICATION", False),
    403: (AuthorizationError, "AUTHORIZATION", False),
    404: (NotFoundError, "NOT_FOUND", False),
    409: (ConflictError, "CONFLICT", False),
    413: (PayloadTooLargeError, "PAYLOAD_TOO_LARGE", False),
    422: (ValidationError, "VALIDATION", False),
    429: (RateLimitedError, "RATE_LIMIT", True),
    502: (DependencyUnavailableError, "DEPENDENCY_UNAVAILABLE", True),
    503: (DependencyUnavailableError, "DEPENDENCY_UNAVAILABLE", True),
    504: (TimeoutError, "TIMEOUT", True),
}
#: Any other status: the base class, not retried.
_UNKNOWN_STATUS: tuple[type[MemoryError], str, bool] = (MemoryError, "INTERNAL", False)

#: What makes a dict an RFC 9457 problem rather than some other JSON body.
_PROBLEM_MEMBERS = frozenset({"code", "type", "title", "detail"})


def error_from_problem(status: int, body: Any, *, retry_after: float | None = None) -> MemoryError:
    """The typed exception for an error response.

    The service answers RFC 9457 problems: ``code`` and ``detail`` at the top level. A plain
    problem from a gateway in front of it (``type``/``title``/``detail``, no ``code``) keeps
    its words, and a 0.1 server's nested ``{"error": {"code", "message", ...}}`` envelope is
    read the same way, so an SDK ahead of its service still raises the right class.

    A body that names no ``code`` (or one this SDK does not know) is classed by its HTTP
    status, so a gateway's 404 is a :class:`NotFoundError` and its 503 a retryable
    :class:`DependencyUnavailableError`, never a bare :class:`MemoryError`.
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
    status_cls, status_code, status_retryable = _BY_STATUS.get(status, _UNKNOWN_STATUS)
    named = problem.get("code")
    code = str(named) if isinstance(named, str) and named else status_code
    cls = _BY_CODE.get(code, status_cls)
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
        retryable=retryable if isinstance(retryable, bool) else status_retryable,
        trace_id=_text(problem.get("trace_id")),
        request_id=_text(problem.get("request_id")),
        details=details if isinstance(details, dict) else {},
        retry_after=retry_after,
    )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
