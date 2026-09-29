"""HTTP transport: headers, trace context, retries for retryable errors, problem mapping."""

from __future__ import annotations

import asyncio
import random
import re
import secrets
from collections.abc import Mapping, MutableMapping
from typing import Any

import httpx

from trellis.memory import errors as sdk_errors
from trellis.memory.errors import MemoryError, error_from_problem
from trellis.memory.models import Scope

try:  # the ``otel`` extra: the W3C propagator writes traceparent for the active span
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    _PROPAGATOR: TraceContextTextMapPropagator | None = TraceContextTextMapPropagator()
except ImportError:  # pragma: no cover - the default install; tests patch the name instead
    _PROPAGATOR = None

HEADER_TENANT = "X-Trellis-Tenant"
HEADER_WORKSPACE = "X-Trellis-Workspace"
HEADER_USER = "X-Trellis-User"
HEADER_API_KEY = "X-API-Key"
HEADER_IDEMPOTENCY = "Idempotency-Key"
HEADER_REQUEST_ID = "X-Request-ID"
HEADER_CORRELATION = "X-Correlation-ID"
HEADER_TRACEPARENT = "traceparent"
_REQUEST_ID = HEADER_REQUEST_ID.lower()

_TRACE_ID = re.compile(r"[0-9a-fA-F]{32}")


def w3c_trace_id(value: str | None) -> str | None:
    """``value`` lower-cased when it is a 32-hex, non-zero trace id (what the service
    continues); ``None`` for anything else."""
    if value and _TRACE_ID.fullmatch(value) and set(value) != {"0"}:
        return value.lower()
    return None


def trace_headers(scope: Scope | None, headers: MutableMapping[str, str]) -> None:
    """Add ``traceparent`` unless the caller sent one: the active span's when OpenTelemetry is
    tracing (the W3C propagator only, never baggage), else one built from ``scope.trace_id``
    when that is a W3C trace id. The service continues either, so an agent's Langfuse trace
    and the service's Datadog trace share one id. Without an active span the parent span id
    is synthetic (no exporter emits it) and the request is marked sampled, so a parent-based
    sampler on the service keeps the request that is being correlated."""
    if HEADER_TRACEPARENT in headers:
        return
    if _PROPAGATOR is not None:
        _PROPAGATOR.inject(headers)
    if HEADER_TRACEPARENT in headers or scope is None:
        return
    trace_id = w3c_trace_id(scope.trace_id)
    if trace_id:
        headers[HEADER_TRACEPARENT] = f"00-{trace_id}-{secrets.token_hex(8)}-01"


class Transport:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "trellis-memory-python",
    ) -> None:
        headers = {"User-Agent": user_agent, "Accept": "application/json, application/problem+json"}
        if api_key:
            headers[HEADER_API_KEY] = api_key
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        if client is None:
            client = httpx.AsyncClient(
                base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
            )
            owns = True
        else:
            client.headers.update(headers)
            # An injected client is the caller's to own, but a caller who forgot the base URL
            # gets "Request URL is missing an 'http://' or 'https://' protocol" from deep
            # inside httpx. Fill it in; an explicit one is left alone.
            if not str(client.base_url):
                client.base_url = base_url.rstrip("/")
            owns = False
        self._client = client
        self._owns_client = owns
        self.max_retries = max_retries

    @staticmethod
    def scope_headers(scope: Scope) -> dict[str, str]:
        h: dict[str, str] = {}
        if scope.tenant_id:
            h[HEADER_TENANT] = scope.tenant_id
        if scope.workspace_id:
            h[HEADER_WORKSPACE] = scope.workspace_id
        if scope.user_id:
            h[HEADER_USER] = scope.user_id
        if scope.correlation_id:
            h[HEADER_CORRELATION] = scope.correlation_id
        elif scope.trace_id and not w3c_trace_id(scope.trace_id):
            # An opaque trace id is a correlation id in all but name: the service continues
            # only W3C ids (ADR 0022), which travel as traceparent, and echoes this one.
            h[HEADER_CORRELATION] = scope.trace_id
        return h

    @staticmethod
    def scope_params(scope: Scope) -> dict[str, str]:
        """Lineage for body-less (GET/DELETE) routes; security fields stay in headers."""
        fields = (
            "thread_id",
            "session_id",
            "work_id",
            "task_id",
            "agent_id",
            "agent_group_id",
            "agent_run_id",
            "parent_agent_run_id",
        )
        return {f: v for f in fields if (v := getattr(scope, f, None))}

    async def request(
        self,
        method: str,
        path: str,
        *,
        scope: Scope | None = None,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        files: Any | None = None,
        data: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """The decoded body of a successful response (None for an empty one)."""
        response = await self._perform(
            method,
            path,
            scope=scope,
            json=json,
            params=params,
            idempotency_key=idempotency_key,
            files=files,
            data=data,
            headers=headers,
        )
        return _decoded(response)

    async def request_text(
        self,
        method: str,
        path: str,
        *,
        scope: Scope | None = None,
        params: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        """The body of a successful response as text, for the routes that do not answer JSON
        (``/metrics``). Errors still arrive as problem documents and raise the same way."""
        response = await self._perform(method, path, scope=scope, params=params, headers=headers)
        return response.text

    async def request_page(
        self,
        path: str,
        *,
        scope: Scope | None = None,
        params: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[Any, str | None]:
        """A list route's body and the cursor of the next page, read from its ``Link``
        header (``rel="next"``), which every paged route sends (ADR 0023)."""
        response = await self._perform("GET", path, scope=scope, params=params, headers=headers)
        return _decoded(response), next_cursor(response.headers.get("link"))

    async def _perform(
        self,
        method: str,
        path: str,
        *,
        scope: Scope | None = None,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        files: Any | None = None,
        data: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        # Header names are case-insensitive on the wire, so they are folded once here: a
        # caller's "x-request-id" or "Traceparent" is honoured rather than joined by a second
        # spelling. A plain dict is what httpx merges anyway, and far cheaper than its Headers.
        merged = {name.lower(): value for name, value in (headers or {}).items()}
        # One request id per logical call, kept across its retries so the service's logs show
        # them as the attempts of one request.
        if _REQUEST_ID not in merged:
            merged[_REQUEST_ID] = secrets.token_hex(16)
        if scope is not None:
            merged.update({k.lower(): v for k, v in self.scope_headers(scope).items()})
            if json is None and method.upper() in ("GET", "DELETE"):
                params = {**self.scope_params(scope), **(params or {})}
        trace_headers(scope, merged)
        if idempotency_key:
            merged[HEADER_IDEMPOTENCY.lower()] = idempotency_key
        safe_to_retry = idempotency_key is not None or method.upper() == "GET"
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._client.request(
                    method, path, json=json, params=params, headers=merged, files=files, data=data
                )
            except httpx.TransportError as exc:
                # A connection that never opened is safe to retry; a request that may have
                # reached the service (a read or write timeout, a dropped connection) only
                # when the write is idempotent, or the retry could duplicate it.
                never_sent = isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout)
                if (never_sent or safe_to_retry) and attempt <= self.max_retries:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise _transport_error(exc) from exc
            if response.status_code < 400:
                return response
            err = error_from_problem(response.status_code, _safe_json(response))
            if err.retryable and safe_to_retry and attempt <= self.max_retries:
                await asyncio.sleep(_backoff(attempt))
                continue
            raise err

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')


def _decoded(response: httpx.Response) -> Any:
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def next_cursor(link_header: str | None) -> str | None:
    """The ``cursor`` of the ``rel="next"`` link, or None on the last page."""
    if not link_header:
        return None
    for part in link_header.split(","):
        match = _NEXT_LINK.search(part)
        if match is None:
            continue
        query = httpx.URL(match.group(1)).params
        value = query.get("cursor")
        return str(value) if value else None
    return None


def _transport_error(exc: httpx.TransportError) -> MemoryError:
    """The typed exception for a request that got no response: a timeout is a timeout, the
    rest is the service being unreachable. ``status`` is 0 because there was none."""
    message = str(exc) or type(exc).__name__
    if isinstance(exc, httpx.TimeoutException):
        return sdk_errors.TimeoutError(message, code="TIMEOUT", status=0, retryable=True)
    return sdk_errors.DependencyUnavailableError(
        message, code="DEPENDENCY_UNAVAILABLE", status=0, retryable=True
    )


def _backoff(attempt: int) -> float:
    return min(2.0, 0.1 * (2 ** (attempt - 1))) + random.uniform(0, 0.05)  # noqa: S311


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None
