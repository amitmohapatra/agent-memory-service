"""Domain exceptions. The API layer maps these onto the typed error envelope."""

from __future__ import annotations

from memory_service.domain.enums import ErrorCode


class MemoryServiceError(Exception):
    """Base class. ``code`` maps to :class:`ErrorCode`; ``retryable`` tells clients what to do."""

    code: ErrorCode = ErrorCode.INTERNAL
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str | None = None, *, details: dict[str, object] | None = None):
        super().__init__(message or self.__class__.__doc__ or self.code)
        self.message = message or self.code.value
        self.details = details or {}


class ValidationFailed(MemoryServiceError):
    """The request is malformed or violates a domain invariant."""

    code = ErrorCode.VALIDATION
    http_status = 422


class AuthenticationFailed(MemoryServiceError):
    """The calling service could not be authenticated."""

    code = ErrorCode.AUTHENTICATION
    http_status = 401


class AuthorizationFailed(MemoryServiceError):
    """The principal is not allowed to perform this operation."""

    code = ErrorCode.AUTHORIZATION
    http_status = 403


class ScopeDenied(AuthorizationFailed):
    """Access to the requested scope is denied."""

    code = ErrorCode.SCOPE_DENIED
    http_status = 403


class NotFound(MemoryServiceError):
    """The requested object does not exist in the caller's scope."""

    code = ErrorCode.NOT_FOUND
    http_status = 404


class Conflict(MemoryServiceError):
    """The request conflicts with current state (e.g. idempotency key reused with new body)."""

    code = ErrorCode.CONFLICT
    http_status = 409


class DependencyUnavailable(MemoryServiceError):
    """A mandatory backing store or provider is unavailable."""

    code = ErrorCode.DEPENDENCY_UNAVAILABLE
    http_status = 503
    retryable = True


class CorruptSource(MemoryServiceError):
    """Source bytes failed checksum / parse validation."""

    code = ErrorCode.CORRUPT_SOURCE
    http_status = 422


class InsufficientEvidence(MemoryServiceError):
    """Retrieval could not assemble the evidence required to answer safely."""

    code = ErrorCode.INSUFFICIENT_EVIDENCE
    http_status = 200  # returned as a typed result, not an HTTP failure


class ProviderNotConfigured(DependencyUnavailable):
    """An optional provider was requested but is not enabled/configured."""
