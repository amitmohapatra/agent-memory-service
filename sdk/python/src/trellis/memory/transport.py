"""HTTP transport: headers, trace context, retries for retryable errors, problem mapping,
and the circuit breaker."""

from __future__ import annotations

import asyncio
import random
import re
import secrets
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final

import httpx

from trellis.memory import errors as sdk_errors
from trellis.memory.breaker import DEFAULT_FAILURE_THRESHOLD, DEFAULT_OPEN_SECONDS, CircuitBreaker
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
#: The status of a conditional GET whose answer has not changed.
NOT_MODIFIED: Final = 304
_REQUEST_ID = HEADER_REQUEST_ID.lower()

_TRACE_ID = re.compile(r"[0-9a-fA-F]{32}")

#: The POSTs that read and change nothing: the body carries a query, not a write, so a retry
#: cannot duplicate anything and they retry like a GET. (``/v1/verify`` with a run records the
#: judge's feedback, which the service keys on the run and the bundle.)
READ_ONLY_POSTS: Final = frozenset({"/v1/context", "/v1/recall", "/v1/verify", "/v1/tools/hints"})
#: The longest a ``Retry-After`` is waited before the next attempt; a longer one is waited
#: this long, so a caller is never parked for a minute inside one call.
RETRY_AFTER_CAP_SECONDS: Final = 30.0
#: Full-jitter exponential backoff for failures that carry no advice of their own: attempt
#: ``n`` sleeps a uniform draw from ``[0, min(cap, base * 2**(n-1))]``. The jitter spreads
#: the retries of many clients that failed together instead of synchronising them.
BACKOFF_BASE_SECONDS: Final = 0.5
BACKOFF_CAP_SECONDS: Final = 8.0
#: Opening a connection either works quickly or not at all; the overall timeout bounds the
#: rest of the request.
CONNECT_TIMEOUT_SECONDS: Final = 5.0
#: Idle connections are kept this long for the next call. Explicit because httpx's default
#: (5 s) is shorter than the gap between an agent's turns, so most turns paid a new
#: connection; a connection the service closed first is discarded by httpx before reuse.
KEEPALIVE_EXPIRY_SECONDS: Final = 30.0
#: The wait between attempts; a name of its own so tests can stand it in.
_sleep = asyncio.sleep
POOL_LIMITS: Final = httpx.Limits(
    max_connections=100, max_keepalive_connections=20, keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS
)


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
        timeout: float | httpx.Timeout = 10.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "trellis-memory-python",
        circuit_failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        circuit_open_seconds: float = DEFAULT_OPEN_SECONDS,
    ) -> None:
        headers = {"User-Agent": user_agent, "Accept": "application/json, application/problem+json"}
        if api_key:
            headers[HEADER_API_KEY] = api_key
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        if client is None:
            client = httpx.AsyncClient(
                base_url=base_url.rstrip("/"),
                timeout=request_timeout(timeout),
                headers=headers,
                limits=POOL_LIMITS,
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
        #: Per client: one service, one circuit. ``circuit_failure_threshold=0`` disables it.
        self.breaker = CircuitBreaker(circuit_failure_threshold, circuit_open_seconds)

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
        """Lineage for body-less routes; security fields stay in headers."""
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
        timeout: float | None = None,  # noqa: ASYNC109 - the httpx request timeout
    ) -> Any:
        """The decoded body of a successful response (None for an empty one). ``timeout``
        replaces the client's for this call (each attempt), when given."""
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
            timeout=timeout,
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

    async def request_conditional(
        self,
        path: str,
        *,
        scope: Scope | None = None,
        params: dict[str, Any] | None = None,
        etag: str | None = None,
    ) -> tuple[Any | None, str | None]:
        """A GET asked with ``If-None-Match`` when ``etag`` is given: the decoded body and the
        answer's ``ETag``, or ``None`` and the ETag to keep when nothing changed (304). A
        route that sends no ``ETag`` is simply read in full each time."""
        headers = {"If-None-Match": etag} if etag else None
        response = await self._perform("GET", path, scope=scope, params=params, headers=headers)
        tag = response.headers.get("etag")
        if response.status_code == NOT_MODIFIED:
            return None, tag or etag
        return _decoded(response), tag

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
        timeout: float | None = None,  # noqa: ASYNC109 - the httpx request timeout
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
            # a body-less call (GET, DELETE, or a POST like accepting an approval
            # suggestion) has nowhere else to carry its lineage
            if json is None and files is None and data is None:
                params = {**self.scope_params(scope), **(params or {})}
        trace_headers(scope, merged)
        if idempotency_key:
            merged[HEADER_IDEMPOTENCY.lower()] = idempotency_key
        request: dict[str, Any] = {
            "json": json,
            "params": params,
            "headers": merged,
            "files": files,
            "data": data,
        }
        # httpx reads an explicit timeout=None as "wait forever": passed only when given
        if timeout is not None:
            request["timeout"] = request_timeout(timeout)
        probe = self.breaker.acquire()
        try:
            response = await self._attempts(
                method, path, request, safe_to_retry=_safe_to_retry(method, path, idempotency_key)
            )
        except MemoryError as err:
            if err.status == 0 or err.status >= 500:
                self.breaker.record_failure()
            elif err.status != 429:
                self.breaker.record_success()
            raise
        else:
            self.breaker.record_success()
            return response
        finally:
            if probe:
                self.breaker.release()

    async def _attempts(
        self, method: str, path: str, request: dict[str, Any], *, safe_to_retry: bool
    ) -> httpx.Response:
        """The response of the first attempt that got one worth returning, retrying what is
        retryable while attempts remain; the typed error otherwise."""
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._client.request(method, path, **request)
            except httpx.TransportError as exc:
                # A connection that never opened is safe to retry; a request that may have
                # reached the service (a read or write timeout, a dropped connection) only
                # when the call is idempotent, or the retry could duplicate it.
                never_sent = isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout)
                if (never_sent or safe_to_retry) and attempt <= self.max_retries:
                    await _sleep(backoff(attempt))
                    continue
                raise _transport_error(exc) from exc
            if response.status_code < 400:
                return response
            err = error_from_problem(
                response.status_code,
                _safe_json(response),
                retry_after=retry_after(response.headers.get("retry-after")),
            )
            # A 429 is refused before any work is done, so even a write without a key may be
            # sent again; anything else is retried only where a retry duplicates nothing.
            may_resend = safe_to_retry or response.status_code == 429
            if err.retryable and may_resend and attempt <= self.max_retries:
                await _sleep(
                    min(err.retry_after, RETRY_AFTER_CAP_SECONDS)
                    if err.retry_after is not None
                    else backoff(attempt)
                )
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


def _safe_to_retry(method: str, path: str, idempotency_key: str | None) -> bool:
    """Whether a request the service may already have received can be sent again."""
    verb = method.upper()
    if idempotency_key is not None or verb == "GET":
        return True
    return verb == "POST" and path in READ_ONLY_POSTS


def backoff(attempt: int) -> float:
    """Full-jitter exponential backoff before retry ``attempt`` (1-based)."""
    ceiling = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
    return random.uniform(0, ceiling)  # noqa: S311 - jitter, not a secret


def retry_after(value: str | None) -> float | None:
    """Seconds a ``Retry-After`` asks for: delta-seconds or an HTTP date; None when absent or
    unreadable."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def request_timeout(timeout: float | httpx.Timeout) -> httpx.Timeout:
    """``timeout`` for the whole request, with connecting bounded by
    ``CONNECT_TIMEOUT_SECONDS`` when that is shorter; an ``httpx.Timeout`` is used as given."""
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT_SECONDS))


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None
