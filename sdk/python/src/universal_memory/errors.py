"""SDK exceptions mapped from the API error envelope."""

from __future__ import annotations

from typing import Any


class MemoryError(Exception):
    """Base SDK error. ``code`` mirrors the API error code; ``retryable`` tells you what to do."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "INTERNAL",
        status: int = 500,
        retryable: bool = False,
        trace_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.retryable = retryable
        self.trace_id = trace_id
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


class RateLimitedError(MemoryError):
    pass


class DependencyUnavailableError(MemoryError):
    pass


class TimeoutError(MemoryError):
    pass


class InsufficientEvidence(MemoryError):
    pass


_BY_CODE: dict[str, type[MemoryError]] = {
    "AUTHENTICATION": AuthenticationError,
    "AUTHORIZATION": AuthorizationError,
    "SCOPE_DENIED": AuthorizationError,
    "NOT_FOUND": NotFoundError,
    "CONFLICT": ConflictError,
    "VALIDATION": ValidationError,
    "RATE_LIMIT": RateLimitedError,
    "DEPENDENCY_UNAVAILABLE": DependencyUnavailableError,
    "RETRYABLE_PROCESSING": DependencyUnavailableError,
    "TIMEOUT": TimeoutError,
    "INSUFFICIENT_EVIDENCE": InsufficientEvidence,
}


def error_from_envelope(status: int, body: dict[str, Any] | None) -> MemoryError:
    err = (body or {}).get("error") or {}
    code = str(err.get("code") or "INTERNAL")
    cls = _BY_CODE.get(code, MemoryError)
    return cls(
        str(err.get("message") or f"HTTP {status}"),
        code=code,
        status=status,
        retryable=bool(err.get("retryable", status in (429, 503, 504))),
        trace_id=err.get("trace_id"),
        details=err.get("details") or {},
    )
