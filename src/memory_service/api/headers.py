"""The API layer's wire headers, and the deprecation window on them (ADR 0022).

The header names the API layer reads or writes (the scope headers live in
``config.constants.HEADERS``, the W3C ``traceparent`` in ``observability.tracing``), the
request-scoped ids the correlation middleware keeps in ``scope["state"]``, and the two rules
of the deprecation window that ends in ``constants.ALIASES_REMOVED_IN``: the ``X-Memory-*``
spellings of the scope headers are read as aliases of ``X-Trellis-*`` (a request whose values
for one header, across spellings and duplicates, are not one value is refused, because a
gateway that stamps one spelling strips only that spelling and a client behind it must not
choose its own tenant or user), and the two alias routes are answered with ``Deprecation``
and ``Link`` headers on every response, whichever layer produced it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from memory_service.config.constants import ALIASES_REMOVED_IN, DEPRECATED_HEADER_ALIASES, HEADERS
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
DEPRECATION_HEADER: Final = "Deprecation"
LINK_HEADER: Final = "Link"

#: response header -> the ``scope["state"]`` key the correlation middleware keeps it under
CORRELATION_STATE: Final[Mapping[str, str]] = {
    REQUEST_ID_HEADER: "request_id",
    TRACE_ID_HEADER: "trace_id",
    CORRELATION_ID_HEADER: "correlation_id",
    TRACEPARENT_HEADER: "traceparent",
}

#: RFC 9745 Deprecation header value (a structured-field date) for the alias routes
ALIASES_DEPRECATED_AT: Final = "@1790553600"  # 2026-09-28T00:00:00Z
#: (method, route path) of an alias route -> the route it is an alias of
DEPRECATED_ROUTES: Final[Mapping[tuple[str, str], str]] = {
    ("POST", "/v1/files"): "/v1/documents",
    ("POST", "/v1/tools/record"): "/v1/tools/invocations",
}


def correlation_headers(state: Mapping[str, Any]) -> dict[str, str]:
    """The four id headers every response carries, from what the middleware resolved."""
    return {header: str(state[key]) for header, key in CORRELATION_STATE.items() if state.get(key)}


def route_path(scope: Mapping[str, Any]) -> str:
    """The path the router matched: ``scope["path"]`` without the mount's ``root_path``."""
    root = scope.get("root_path") or ""
    path: str = scope["path"]
    return path[len(root) :] if root and path.startswith(root) else path


def deprecation_headers_for(method: str, path: str, root_path: str = "") -> dict[str, str]:
    """``Deprecation`` and ``Link`` for an alias route's response; empty for any other route."""
    successor = DEPRECATED_ROUTES.get((method.upper(), path))
    if successor is None:
        return {}
    return {
        DEPRECATION_HEADER: ALIASES_DEPRECATED_AT,
        LINK_HEADER: f'<{root_path}{successor}>; rel="successor-version"',
    }


def alias_route(path: str) -> dict[str, Any]:
    """The decorator keywords an alias route shares: the pairing is written once, here."""
    successor = DEPRECATED_ROUTES[("POST", path)]
    return {
        "deprecated": True,
        "summary": f"Deprecated alias of POST {successor}",
        "description": f"Deprecated alias of `POST {successor}` (ADR 0022); removed in "
        f"{ALIASES_REMOVED_IN}.",
    }


def _values(headers: Mapping[str, str], name: str) -> list[str]:
    """Every value sent for ``name``: all of them from a multi-valued mapping (Starlette's
    ``Headers``), the one from a plain mapping. Both must look up case-insensitively."""
    getlist = getattr(headers, "getlist", None)
    if getlist is not None:
        return list(getlist(name))
    value = headers.get(name)
    return [value] if value is not None else []


def scope_values(headers: Mapping[str, str], name: str) -> list[str]:
    """Every non-blank value sent for ``name`` under either spelling, new spelling first,
    duplicates included: a proxy that appends rather than replaces leaves the client's line
    in front, and an empty header is no header."""
    values = _values(headers, name)
    alias = DEPRECATED_HEADER_ALIASES.get(name)
    if alias is not None:
        values += _values(headers, alias)
    return [value for value in (raw.strip() for raw in values) if value]


def scope_header(headers: Mapping[str, str], name: str) -> str | None:
    """The value of ``name``, under whichever spelling was sent; the first when several were.

    This is the lenient reader the rate limiter counts with, after the correlation middleware
    has refused every request whose values disagree (:func:`refuse_ambiguous_headers`).
    Everything that acts on the value goes through :func:`require_one_spelling`.
    """
    values = scope_values(headers, name)
    return values[0] if values else None


def require_one_spelling(headers: Mapping[str, str], name: str) -> str | None:
    """:func:`scope_header`, refusing a request whose values for ``name`` disagree."""
    values = scope_values(headers, name)
    if len(set(values)) > 1:
        raise ValidationFailed(
            f"{name} was sent more than once with different values (its deprecated spelling "
            f"{DEPRECATED_HEADER_ALIASES.get(name, name)} counts); send one",
            details={"field": name},
        )
    return values[0] if values else None


def refuse_ambiguous_headers(headers: Mapping[str, str]) -> None:
    """The check the correlation middleware runs before anything reads a header: the scope
    headers under both spellings, and the credential headers, whose first value keys the
    rate limiter's bucket and whose last value the authenticator would otherwise verify."""
    for name in (HEADERS.tenant, HEADERS.workspace, HEADERS.user):
        require_one_spelling(headers, name)
    for name in (HEADERS.api_key, AUTHORIZATION_HEADER):
        if len(set(_values(headers, name))) > 1:
            raise ValidationFailed(
                f"{name} was sent more than once with different values; send one",
                details={"field": name},
            )
