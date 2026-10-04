"""The API layer's wire headers.

The header names the API layer reads or writes (the scope headers live in
``config.constants.HEADERS``, the W3C ``traceparent`` in ``observability.tracing``), the
request-scoped ids the correlation middleware keeps in ``scope["state"]``, and the one rule
on the scope headers: a request whose values for one header are not one value is refused,
because a gateway that stamps a header and a client behind it that sends its own must not
leave the client choosing its own tenant or user.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from memory_service.config.constants import HEADERS
from memory_service.domain.errors import ValidationFailed
from memory_service.observability.tracing import TRACEPARENT_HEADER

REQUEST_ID_HEADER: Final = HEADERS.request_id
TRACE_ID_HEADER: Final = "X-Trace-ID"
CORRELATION_ID_HEADER: Final = "X-Correlation-ID"
AUTHORIZATION_HEADER: Final = "Authorization"
IDEMPOTENCY_KEY_HEADER: Final = "Idempotency-Key"
IDEMPOTENT_REPLAYED_HEADER: Final = "Idempotent-Replayed"
RATE_LIMIT_LIMIT_HEADER: Final = "X-RateLimit-Limit"
RATE_LIMIT_REMAINING_HEADER: Final = "X-RateLimit-Remaining"
RETRY_AFTER_HEADER: Final = "Retry-After"
LINK_HEADER: Final = "Link"
LOCATION_HEADER: Final = "Location"
ETAG_HEADER: Final = "ETag"
IF_NONE_MATCH_HEADER: Final = "If-None-Match"
CACHE_CONTROL_HEADER: Final = "Cache-Control"

#: The longest Idempotency-Key a client may send (the store keeps 300 characters, and the
#: service's own derived keys need room beside a client's in the same column).
IDEMPOTENCY_KEY_MAX_CHARS: Final = 255

#: response header -> the ``scope["state"]`` key the correlation middleware keeps it under
CORRELATION_STATE: Final[Mapping[str, str]] = {
    REQUEST_ID_HEADER: "request_id",
    TRACE_ID_HEADER: "trace_id",
    CORRELATION_ID_HEADER: "correlation_id",
    TRACEPARENT_HEADER: "traceparent",
}


def correlation_headers(state: Mapping[str, Any]) -> dict[str, str]:
    """The four id headers every response carries, from what the middleware resolved."""
    return {header: str(state[key]) for header, key in CORRELATION_STATE.items() if state.get(key)}


def _values(headers: Mapping[str, str], name: str) -> list[str]:
    """Every value sent for ``name``: all of them from a multi-valued mapping (Starlette's
    ``Headers``), the one from a plain mapping. Both must look up case-insensitively."""
    getlist = getattr(headers, "getlist", None)
    if getlist is not None:
        return list(getlist(name))
    value = headers.get(name)
    return [value] if value is not None else []


def scope_values(headers: Mapping[str, str], name: str) -> list[str]:
    """Every non-blank value sent for ``name``, duplicates included: a proxy that appends
    rather than replaces leaves the client's line in front, and an empty header is no
    header."""
    return [value for value in (raw.strip() for raw in _values(headers, name)) if value]


def scope_header(headers: Mapping[str, str], name: str) -> str | None:
    """The value of ``name``; the first when several were sent.

    This is the lenient reader the rate limiter counts with, after the correlation middleware
    has refused every request whose values disagree (:func:`refuse_ambiguous_headers`).
    Everything that acts on the value goes through :func:`require_one_value`.
    """
    values = scope_values(headers, name)
    return values[0] if values else None


def require_one_value(headers: Mapping[str, str], name: str) -> str | None:
    """:func:`scope_header`, refusing a request whose values for ``name`` disagree."""
    values = scope_values(headers, name)
    if len(set(values)) > 1:
        raise ValidationFailed(
            f"{name} was sent more than once with different values; send one",
            details={"field": name},
        )
    return values[0] if values else None


def refuse_ambiguous_headers(headers: Mapping[str, str]) -> None:
    """The check the correlation middleware runs before anything reads a header: the scope
    headers and the credential headers, whose first value keys the
    rate limiter's bucket and whose last value the authenticator would otherwise verify."""
    for name in (HEADERS.tenant, HEADERS.workspace, HEADERS.user):
        require_one_value(headers, name)
    for name in (HEADERS.api_key, AUTHORIZATION_HEADER):
        if len(set(_values(headers, name))) > 1:
            raise ValidationFailed(
                f"{name} was sent more than once with different values; send one",
                details={"field": name},
            )


def idempotency_key_of(headers: Mapping[str, str]) -> str | None:
    """The request's ``Idempotency-Key``: absent or blank is none, longer than
    :data:`IDEMPOTENCY_KEY_MAX_CHARS` or sent twice with different values is refused."""
    values = scope_values(headers, IDEMPOTENCY_KEY_HEADER)
    if len(set(values)) > 1:
        raise ValidationFailed(
            f"{IDEMPOTENCY_KEY_HEADER} was sent more than once with different values; send one",
            details={"field": IDEMPOTENCY_KEY_HEADER},
        )
    if values and len(values[0]) > IDEMPOTENCY_KEY_MAX_CHARS:
        raise ValidationFailed(
            f"{IDEMPOTENCY_KEY_HEADER} is at most {IDEMPOTENCY_KEY_MAX_CHARS} characters",
            details={"field": IDEMPOTENCY_KEY_HEADER},
        )
    return values[0] if values else None
