"""Retries, Retry-After, the status fallback of the error classes, the circuit breaker and the
zero-configuration defaults."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest
import respx

from trellis.memory import (
    AuthenticationError,
    AuthorizationError,
    CircuitOpenError,
    ConflictError,
    DependencyUnavailableError,
    MemoryClient,
    MemoryContext,
    MemoryError,
    NotFoundError,
    RateLimitedError,
    TimeoutError,
    ValidationError,
)
from trellis.memory import transport as transport_module
from trellis.memory.errors import error_from_problem

URL = "http://memory.test"
UNAVAILABLE = {"code": "DEPENDENCY_UNAVAILABLE", "detail": "db", "retryable": True}
LIMITED = {"code": "RATE_LIMIT", "detail": "slow down", "retryable": True}
BUNDLE = {"bundle_id": "b1", "evidence_status": "COMPLETE", "token_estimate": 0}


@pytest.fixture
def client() -> MemoryClient:
    return MemoryClient(URL, api_key="k", max_retries=2)


def _ctx(client: MemoryClient) -> MemoryContext:
    return client.bind(tenant_id="acme", user_id="u1")


# -------------------------------------------------------------------------- Retry-After


@respx.mock
async def test_a_rate_limit_is_waited_out_for_as_long_as_the_service_asks(
    client: MemoryClient, slept: list[float]
) -> None:
    route = respx.post(f"{URL}/v1/context").mock(
        side_effect=[
            httpx.Response(429, json=LIMITED, headers={"Retry-After": "2"}),
            httpx.Response(200, json=BUNDLE),
        ]
    )
    await _ctx(client).context("q", format="full")
    assert route.call_count == 2 and slept == [2.0]


@respx.mock
async def test_a_long_retry_after_is_capped(client: MemoryClient, slept: list[float]) -> None:
    respx.post(f"{URL}/v1/context").mock(
        side_effect=[
            httpx.Response(429, json=LIMITED, headers={"Retry-After": "120"}),
            httpx.Response(200, json=BUNDLE),
        ]
    )
    await _ctx(client).context("q", format="full")
    assert slept == [transport_module.RETRY_AFTER_CAP_SECONDS]


@respx.mock
async def test_an_exhausted_rate_limit_carries_its_retry_after(
    client: MemoryClient, slept: list[float]
) -> None:
    respx.post(f"{URL}/v1/context").mock(
        return_value=httpx.Response(429, json=LIMITED, headers={"Retry-After": "7"})
    )
    with pytest.raises(RateLimitedError) as exc:
        await _ctx(client).context("q")
    assert exc.value.retry_after == 7.0 and exc.value.retryable
    assert slept == [7.0, 7.0]


def test_retry_after_reads_seconds_and_http_dates() -> None:
    assert transport_module.retry_after("3") == 3.0
    assert transport_module.retry_after("-1") == 0.0
    assert transport_module.retry_after(None) is None
    assert transport_module.retry_after("soon") is None
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=20), usegmt=True)
    waited = transport_module.retry_after(later)
    assert waited is not None and 15 <= waited <= 20


@respx.mock
async def test_a_rate_limited_write_without_a_key_is_resent(
    client: MemoryClient, slept: list[float]
) -> None:
    """A 429 is refused before any work: resending duplicates nothing even without a key."""
    route = respx.post(f"{URL}/v1/tools/invocations").mock(
        side_effect=[
            httpx.Response(429, json=LIMITED, headers={"Retry-After": "1"}),
            httpx.Response(201, json={"invocation_id": "tiv_1"}),
        ]
    )
    result = await _ctx(client).agent("bot").record_tool("search", {"q": "x"})
    assert result.invocation_id == "tiv_1" and route.call_count == 2
    assert "idempotency-key" not in route.calls.last.request.headers


# -------------------------------------------------------------------------- backoff


def test_backoff_is_full_jitter_under_a_growing_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    draws: list[tuple[float, float]] = []
    monkeypatch.setattr(
        transport_module.random, "uniform", lambda lo, hi: draws.append((lo, hi)) or hi
    )
    waits = [transport_module.backoff(n) for n in range(1, 8)]
    assert all(lo == 0 for lo, _ in draws)  # full jitter: anywhere from no wait at all
    assert waits[:5] == [0.5, 1.0, 2.0, 4.0, 8.0]
    assert waits[5:] == [transport_module.BACKOFF_CAP_SECONDS] * 2


@respx.mock
async def test_a_retry_without_advice_backs_off(client: MemoryClient, slept: list[float]) -> None:
    respx.get(f"{URL}/v1/jobs/job_1").mock(return_value=httpx.Response(503, json=UNAVAILABLE))
    with pytest.raises(DependencyUnavailableError):
        await _ctx(client).advanced.job("job_1")
    assert len(slept) == 2
    assert 0 <= slept[0] <= 0.5 and 0 <= slept[1] <= 1.0


# -------------------------------------------------------------------------- read-only POSTs

READS: list[tuple[str, dict[str, Any], Callable[[MemoryContext], Awaitable[Any]]]] = [
    ("/v1/context", BUNDLE, lambda ctx: ctx.context("q", format="full")),
    ("/v1/recall", {"items": []}, lambda ctx: ctx.search("q")),
    ("/v1/verify", {"claims": []}, lambda ctx: ctx.verify("an answer", bundle_id="b1")),
    ("/v1/tools/hints", {"tools": []}, lambda ctx: ctx.tool_hints("a task")),
]


@respx.mock
@pytest.mark.parametrize(("path", "body", "call"), READS, ids=[r[0] for r in READS])
async def test_a_read_only_post_is_retried_on_unavailable_and_timeouts(
    client: MemoryClient,
    path: str,
    body: dict[str, Any],
    call: Callable[[MemoryContext], Awaitable[Any]],
) -> None:
    route = respx.post(f"{URL}{path}").mock(
        side_effect=[
            httpx.Response(503, json=UNAVAILABLE),
            httpx.ReadTimeout("slow"),
            httpx.Response(200, json=body),
        ]
    )
    await call(_ctx(client))
    assert route.call_count == 3
    assert "idempotency-key" not in route.calls.last.request.headers


@respx.mock
async def test_a_write_without_a_key_is_not_retried_on_unavailable(client: MemoryClient) -> None:
    route = respx.post(f"{URL}/v1/tools/invocations").mock(
        return_value=httpx.Response(503, json=UNAVAILABLE)
    )
    with pytest.raises(DependencyUnavailableError):
        await _ctx(client).agent("bot").record_tool("search", {"q": "x"})
    assert route.call_count == 1


# -------------------------------------------------------------------------- error classes


@pytest.mark.parametrize(
    ("status", "cls", "code", "retryable"),
    [
        (400, ValidationError, "VALIDATION", False),
        (401, AuthenticationError, "AUTHENTICATION", False),
        (403, AuthorizationError, "AUTHORIZATION", False),
        (404, NotFoundError, "NOT_FOUND", False),
        (409, ConflictError, "CONFLICT", False),
        (422, ValidationError, "VALIDATION", False),
        (429, RateLimitedError, "RATE_LIMIT", True),
        (500, MemoryError, "INTERNAL", False),
        (502, DependencyUnavailableError, "DEPENDENCY_UNAVAILABLE", True),
        (503, DependencyUnavailableError, "DEPENDENCY_UNAVAILABLE", True),
        (504, TimeoutError, "TIMEOUT", True),
        (418, MemoryError, "INTERNAL", False),
    ],
)
@pytest.mark.parametrize("body", [None, {"title": "From the gateway"}, {"code": "SOMETHING_NEW"}])
def test_a_body_without_a_known_code_is_classed_by_its_status(
    status: int, cls: type[MemoryError], code: str, retryable: bool, body: Any
) -> None:
    err = error_from_problem(status, body)
    assert type(err) is cls and err.retryable is retryable and err.status == status
    named = body.get("code") if isinstance(body, dict) else None
    assert err.code == (named or code)


def test_the_service_s_own_retryable_wins_over_the_status() -> None:
    err = error_from_problem(503, {"code": "DEPENDENCY_UNAVAILABLE", "retryable": False})
    assert isinstance(err, DependencyUnavailableError) and err.retryable is False
    err = error_from_problem(422, {"code": "CORRUPT_SOURCE", "detail": "bad bytes"})
    assert isinstance(err, ValidationError) and err.code == "CORRUPT_SOURCE"


# -------------------------------------------------------------------------- circuit breaker


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _breaking_client(threshold: int = 2) -> tuple[MemoryClient, Clock]:
    client = MemoryClient(URL, api_key="k", max_retries=0)
    clock = Clock()
    breaker = client.transport.breaker
    breaker.threshold, breaker._clock = threshold, clock
    return client, clock


@respx.mock
async def test_the_circuit_opens_after_failed_calls_and_fails_fast() -> None:
    client, clock = _breaking_client()
    route = respx.get(f"{URL}/v1/jobs/job_1").mock(
        side_effect=[httpx.Response(503, json=UNAVAILABLE), httpx.ConnectError("refused")]
    )
    ctx = _ctx(client)
    with pytest.raises(DependencyUnavailableError):
        await ctx.advanced.job("job_1")
    with pytest.raises(DependencyUnavailableError):
        await ctx.advanced.job("job_1")
    assert client.transport.breaker.state == "open"
    clock.now += 10
    with pytest.raises(CircuitOpenError) as exc:
        await ctx.advanced.job("job_1")
    assert route.call_count == 2  # nothing was sent
    err = exc.value
    assert isinstance(err, DependencyUnavailableError) and err.retryable
    assert err.code == "CIRCUIT_OPEN" and err.status == 0 and err.retry_after == 20.0


@respx.mock
async def test_client_errors_and_rate_limits_do_not_open_the_circuit(slept: list[float]) -> None:
    client, _ = _breaking_client()
    respx.get(f"{URL}/v1/jobs/missing").mock(
        return_value=httpx.Response(404, json={"code": "NOT_FOUND", "retryable": False})
    )
    respx.get(f"{URL}/v1/jobs/busy").mock(
        return_value=httpx.Response(429, json=LIMITED, headers={"Retry-After": "1"})
    )
    ctx = _ctx(client)
    for _ in range(3):
        with pytest.raises(NotFoundError):
            await ctx.advanced.job("missing")
        with pytest.raises(RateLimitedError):
            await ctx.advanced.job("busy")
    assert client.transport.breaker.state == "closed"


@respx.mock
async def test_an_answer_between_failures_resets_the_count() -> None:
    client, _ = _breaking_client()
    respx.get(f"{URL}/v1/jobs/job_1").mock(
        side_effect=[
            httpx.Response(503, json=UNAVAILABLE),
            httpx.Response(404, json={"code": "NOT_FOUND"}),
            httpx.Response(503, json=UNAVAILABLE),
        ]
    )
    ctx = _ctx(client)
    for expected in (DependencyUnavailableError, NotFoundError, DependencyUnavailableError):
        with pytest.raises(expected):
            await ctx.advanced.job("job_1")
    assert client.transport.breaker.state == "closed"


@respx.mock
async def test_half_open_lets_one_probe_through_and_its_success_closes() -> None:
    client, clock = _breaking_client(threshold=1)
    route = respx.get(f"{URL}/v1/jobs/job_1").mock(
        side_effect=[
            httpx.Response(503, json=UNAVAILABLE),
            httpx.Response(200, json={"job_id": "job_1", "status": "SUCCEEDED"}),
        ]
    )
    ctx = _ctx(client)
    with pytest.raises(DependencyUnavailableError):
        await ctx.advanced.job("job_1")
    clock.now += 31
    breaker = client.transport.breaker
    assert breaker.state == "half_open"
    assert breaker.acquire() is True  # a probe is in flight ...
    with pytest.raises(CircuitOpenError):  # ... so every other call still fails fast
        await ctx.advanced.job("job_1")
    breaker.release()
    await ctx.advanced.job("job_1")  # this call is the probe, and it succeeds
    assert breaker.state == "closed" and route.call_count == 2


@respx.mock
async def test_a_failed_probe_opens_the_circuit_again() -> None:
    client, clock = _breaking_client(threshold=1)
    respx.get(f"{URL}/v1/jobs/job_1").mock(return_value=httpx.Response(502, text="bad gateway"))
    ctx = _ctx(client)
    with pytest.raises(DependencyUnavailableError):
        await ctx.advanced.job("job_1")
    clock.now += 31
    with pytest.raises(DependencyUnavailableError) as exc:
        await ctx.advanced.job("job_1")
    assert not isinstance(exc.value, CircuitOpenError)  # the probe was sent
    assert client.transport.breaker.state == "open"


@respx.mock
async def test_a_threshold_of_zero_disables_the_breaker() -> None:
    client, _ = _breaking_client(threshold=0)
    route = respx.get(f"{URL}/v1/jobs/job_1").mock(
        return_value=httpx.Response(503, json=UNAVAILABLE)
    )
    for _ in range(5):
        with pytest.raises(DependencyUnavailableError) as exc:
            await _ctx(client).advanced.job("job_1")
        assert not isinstance(exc.value, CircuitOpenError)
    assert route.call_count == 5


def test_each_client_has_its_own_circuit() -> None:
    first, second = MemoryClient(URL), MemoryClient(URL)
    assert first.transport.breaker is not second.transport.breaker


# -------------------------------------------------------------------------- zero config


@respx.mock
async def test_the_environment_names_the_service_and_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_URL", "http://memory.env:9000/")
    monkeypatch.setenv("TRELLIS_API_KEY", "env-key")
    route = respx.get("http://memory.env:9000/health/live").respond(200, json={"ok": True})
    async with MemoryClient() as client:
        await client.alive()
    assert route.calls.last.request.headers["X-API-Key"] == "env-key"


@respx.mock
async def test_explicit_arguments_win_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_URL", "http://memory.env:9000")
    monkeypatch.setenv("TRELLIS_API_KEY", "env-key")
    route = respx.get(f"{URL}/health/live").respond(200, json={"ok": True})
    await MemoryClient(URL, api_key="given").alive()
    assert route.calls.last.request.headers["X-API-Key"] == "given"
    bearer = respx.get(f"{URL}/version").respond(200, json={})
    await MemoryClient(URL, bearer_token="tok").version()
    assert "X-API-Key" not in bearer.calls.last.request.headers  # a token is the credential


def test_without_configuration_the_client_points_at_the_local_stack() -> None:
    client = MemoryClient()
    assert str(client.transport._client.base_url) == "http://localhost:8080"
    assert "x-api-key" not in client.transport._client.headers


def test_the_pool_keeps_connections_and_bounds_connecting() -> None:
    client = MemoryClient(URL, timeout=12.0)
    http = client.transport._client
    assert http.timeout == httpx.Timeout(12.0, connect=5.0)
    assert http._transport._pool._keepalive_expiry == 30.0  # type: ignore[attr-defined]
    assert MemoryClient(URL, timeout=2.0).transport._client.timeout.connect == 2.0


@respx.mock
async def test_a_call_may_carry_its_own_timeout(client: MemoryClient) -> None:
    route = respx.post(f"{URL}/v1/context").respond(200, json=BUNDLE)
    await _ctx(client).context("q", format="full", timeout=1.5)
    assert route.calls.last.request.extensions["timeout"]["read"] == 1.5
    await _ctx(client).context("q", format="full")
    assert route.calls.last.request.extensions["timeout"]["read"] == 10.0
