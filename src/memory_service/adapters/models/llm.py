"""Generative LLM adapter: whatever endpoint the call resolves to, through bifrost-sdk.

Each call resolves its endpoint first (``modules.llm.endpoints``: the acting agent's
registered ``Endpoint``, else the service's own ``LLM_*``, else none) and goes out through
one ``bifrost-sdk`` client per endpoint: any OpenAI-compatible API (OpenAI, OpenRouter, a
LiteLLM proxy, Ollama, vLLM, llama.cpp, the Anthropic and Gemini compatible endpoints, a
Bifrost gateway). Three strategies pick how, each by name from one registry and none by an
if/else here: the route (``ROUTES``: ``openai-compatible``, ``bifrost``), the auth
(``bifrost_sdk.AUTH_STRATEGIES``: ``bearer``, ``google``, ``aws``) and the model's profile
(``trellis.contracts.ModelProfile``: a reasoning model is sent ``max_completion_tokens`` and no
``temperature``, shaped by the sdk from that data, never from the model's name). No provider
SDK is imported anywhere in this code base (``tests/unit/test_architecture.py``).

Bounded by construction: one timeout per call, ``max_retries`` retries with exponential
backoff on transient failures (429/5xx/timeouts/connection errors), and a circuit breaker per
endpoint that fails fast for ``circuit_open_seconds`` after ``circuit_failure_threshold``
consecutive failures, so one agent's broken endpoint never trips another's. All three come
from the shared ``bifrost-sdk`` client. What is this service's is the resolution, the per-use
model routing, the metrics and spans, and the mapping of the client's errors into the
domain's. Every call is traced (model, use, tokens, latency), metered and logged without
prompt text unless ``log_source_text`` is on.
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Hashable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import httpx
from bifrost_sdk import (
    RETRYABLE,
    Bifrost,
    CircuitOpen,
    Gateway,
    GatewayError,
    Options,
    RateLimited,
    Unreachable,
    auth_strategy,
)
from bifrost_sdk import ModelProfile as WireProfile

from memory_service.adapters.models.catalog import ModelCatalog, profile_of
from memory_service.config.constants import LLM, LLM_TRANSPORT, LLMTransport, LLMTuning
from memory_service.config.settings import LLMSettings
from memory_service.domain.errors import (
    DependencyUnavailable,
    ProviderNotConfigured,
    ValidationFailed,
)
from memory_service.modules.llm.cost import record_llm_tokens
from memory_service.modules.llm.endpoints import EndpointChoice, EndpointResolver
from memory_service.modules.llm.policy import current_model_identity, model_for
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import llm_requests_total, llm_seconds, llm_tokens_total
from memory_service.observability.tracing import span
from memory_service.ports.credentials import (
    CredentialResolver,
    ModelIdentity,
    ResolvedCredential,
)
from memory_service.ports.llm import LLMUsageRecorder
from memory_service.ports.models import LLMCompletion, LLMMessage, ProviderInfo
from trellis.contracts import Endpoint, ModelProfile

log = get_logger(__name__)

#: The most endpoints with an open client at once; the least recently used is closed (after
#: its calls finish) and opened again when next needed.
MAX_ENDPOINT_CLIENTS = 256
#: The registration probe's budget: a reachable endpoint lists its models well inside it.
PROBE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class Route:
    """How a memory call travels on one kind of endpoint."""

    gateway: Gateway
    #: per-call options (Bifrost's ``x-bf-*`` headers); None where the endpoint has none
    options: Options | None = None


#: The routes by the name ``LLM_GATEWAY`` and an ``Endpoint.gateway`` give. On Bifrost a
#: memory call keeps its content out of the gateway's logs, and the sdk sends the gateway's
#: deny-all MCP scope on every completion (no injected tools, no Agent Mode).
ROUTES: dict[str, Route] = {
    "openai-compatible": Route("openai-compatible"),
    "bifrost": Route("bifrost", Options(content_logging=False)),
}


def route_of(endpoint: Endpoint) -> Route:
    try:
        return ROUTES[endpoint.gateway]
    except KeyError:
        raise ValueError(
            f"unknown gateway {endpoint.gateway!r}; known: {', '.join(sorted(ROUTES))}"
        ) from None


def auth_of(endpoint: Endpoint) -> httpx.Auth | None:
    """The endpoint's auth strategy by name; a bearer endpoint with no key (a local server)
    sends no ``Authorization`` at all."""
    key = endpoint.api_key.get_secret_value() if endpoint.api_key else None
    if endpoint.auth == "bearer" and not key:
        return None
    return auth_strategy(endpoint.auth, key)


def wire_profile(profile: ModelProfile) -> WireProfile:
    """The two facts the sdk shapes a request from."""
    return WireProfile(reasoning_output=profile.reasoning_output, temperature=profile.temperature)


def _http_client(
    endpoint: Endpoint, timeout: float, transport: httpx.AsyncBaseTransport | None
) -> httpx.AsyncClient:
    """The endpoint's HTTP client, its auth applied to every request (raises ``ValueError``
    for an unknown auth or one built wrong, e.g. a key given to ``google``)."""
    return httpx.AsyncClient(
        base_url=endpoint.base_url.rstrip("/"),
        headers={"Content-Type": "application/json"},
        auth=auth_of(endpoint),
        timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
        transport=transport,
    )


class HttpEndpointProbe:
    """``ports.credentials.EndpointProbe``: ``GET /models`` with the endpoint's own route and
    auth, before an endpoint is stored. An address only the harness's machine resolves (its
    loopback address, a compose service name of its own) is refused here, not at the first
    memory call."""

    def __init__(self, *, http: httpx.AsyncBaseTransport | None = None) -> None:
        self.http = http

    async def __call__(self, endpoint: Endpoint) -> None:
        try:
            route_of(endpoint)
            client = _http_client(endpoint, PROBE_TIMEOUT_SECONDS, self.http)
        except (ValueError, ImportError) as exc:
            raise ValidationFailed(f"The endpoint cannot be used: {exc}") from exc
        where = f"{endpoint.base_url.rstrip('/')}/models"
        async with client:
            try:
                response = await client.get("/models")
            except httpx.HTTPError as exc:
                raise ValidationFailed(
                    f"The memory service cannot reach {where} ({type(exc).__name__}). "
                    "Register an address the service can reach: one only the harness's "
                    "machine resolves (its loopback address, a name in its own network) "
                    "does not work from here.",
                    details={"base_url": endpoint.base_url},
                ) from exc
        if not response.is_success:
            raise ValidationFailed(
                f"GET {where} answered {response.status_code}: check the endpoint's key and "
                "auth, and that base_url is the API root (it usually ends in /v1).",
                details={"base_url": endpoint.base_url, "status": response.status_code},
            )


@dataclass(frozen=True)
class Selection:
    """The model a call goes to and what it accepts."""

    model: str
    profile: ModelProfile


@dataclass(eq=False)
class _Connection:
    """One endpoint's client: its HTTP client, the sdk client per profile over it, and how
    many calls are using it (a retired connection closes when the last one finishes)."""

    endpoint: Endpoint = field(repr=False)
    route: Route
    revision: int | None
    http: httpx.AsyncClient = field(repr=False)
    gateways: dict[WireProfile, Bifrost] = field(default_factory=dict, repr=False)
    users: int = 0
    retired: bool = False

    async def close(self) -> None:
        for gateway in self.gateways.values():
            await gateway.aclose()
        await self.http.aclose()


_BAD_REQUEST = 400


class LLMCallFailed(DependencyUnavailable):
    """The gateway answered but the call could not be completed (bad status or payload)."""


class LLMOutputInvalid(LLMCallFailed):
    """A structured call returned output that does not satisfy the requested schema."""

    retryable = False


def _text_or_none(data: dict[str, Any]) -> str | None:
    """The response text for a log line. Never raises: logging is not a control path."""
    try:
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content")
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else None


_NO_RESPONSE_FORMAT = re.compile(
    r"response_format|json_schema|structured output|constrained decoding", re.IGNORECASE
)


def _unsupported_response_format(exc: LLMCallFailed) -> bool:
    """Whether the gateway refused the request *because* of the JSON envelope.

    Narrow on purpose: a 400 that is not about the envelope is a real bad request, and
    retrying it without constrained decoding would hide a genuine bug behind a slower path.
    """
    if exc.details.get("status") != _BAD_REQUEST:
        return False
    return bool(_NO_RESPONSE_FORMAT.search(str(exc.details.get("body") or "")))


def _json_only_instruction(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema as a prompt, for models that cannot be constrained by the API."""
    return {
        "role": "user",
        "content": (
            "Reply with a single JSON object and nothing else - no prose, no code fence. "
            f"It must match this JSON Schema:\n{json.dumps(schema)}"
        ),
    }


class DisabledLLM:
    """The LLM port when no endpoint can ever resolve: every call is a ProviderNotConfigured.

    Wired when the service has no endpoint of its own and the operator does not accept
    agents' (``models.llm.accept_agent_endpoints=false``)."""

    info = ProviderInfo(
        name="disabled-llm", version="0", license="Apache-2.0", origin="internal", locality="local"
    )
    enabled = False

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        use: str = "generic",
    ) -> LLMCompletion:
        raise ProviderNotConfigured("no model endpoint is configured")

    async def structured(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        use: str = "generic",
    ) -> dict[str, Any]:
        raise ProviderNotConfigured("no model endpoint is configured")

    async def ping(self) -> bool:
        return False

    async def close(self) -> None:
        return None


class EndpointLLM:
    info = ProviderInfo(
        name="llm-endpoint",
        version="openai-compatible",
        license="Apache-2.0",
        origin="bifrost-sdk",
        locality="remote",
        data_residency="endpoint",
    )
    enabled = True

    def __init__(
        self,
        settings: LLMSettings,
        *,
        log_source_text: bool = False,
        http: httpx.AsyncBaseTransport | None = None,
        transport: LLMTransport = LLM_TRANSPORT,
        credentials: CredentialResolver | None = None,
        resolver: EndpointResolver | None = None,
        usage: LLMUsageRecorder | None = None,
        tuning: LLMTuning = LLM,
    ) -> None:
        self.settings = settings
        #: models, output budget, timeout and retries (``constants.LLM`` unless a container
        #: override replaced it)
        self.tuning = tuning
        self.transport = transport
        self.credentials = credentials
        self._resolver = resolver
        self.usage = usage
        self.log_source_text = log_source_text
        #: the HTTP transport every endpoint client uses (a test's stand-in; None: the network)
        self._http = http
        #: Models that answered 400 to the JSON-schema envelope, so later structured calls
        #: put the schema in the prompt from the first attempt instead of paying a wasted
        #: round trip each time (DeepSeek did, on every one of ~600 calls in one run).
        self._schema_in_prompt: set[str] = set()
        self._credential_attempt: ContextVar[
            tuple[ModelIdentity, ResolvedCredential | None] | None
        ] = ContextVar("memory_credential_attempt", default=None)
        self._connections: OrderedDict[Hashable, _Connection] = OrderedDict()
        self._catalog = ModelCatalog()
        if settings.endpoint is not None:
            # The service's own endpoint is built now, so an unknown LLM_GATEWAY or LLM_AUTH
            # (or a missing auth extra) fails at startup rather than at the first call.
            self._remember("service", self._open(settings.endpoint, None))

    @property
    def resolver(self) -> EndpointResolver:
        """The one resolution order (``modules.llm.endpoints``): the container's, else one
        over this adapter's settings and credentials."""
        return self._resolver or EndpointResolver(self.settings, self.credentials)

    # ------------------------------------------------------------------ port
    async def resolve_model(self, use: str) -> str:
        """The model a call for ``use`` goes to now (``auto`` resolved through the endpoint),
        for a caller that records which model wrote something."""
        async with self._endpoint() as (connection, key):
            return (await self._select(use, connection, key)).model

    async def _select(self, use: str, connection: _Connection, key: Hashable) -> Selection:
        """The tenant policy's model for ``use``, else the endpoint's, else the service's
        (``constants.LLM``), with ``auto`` chosen from the endpoint's ``/models``."""
        endpoint = connection.endpoint
        configured = model_for(use, self.tuning, endpoint.model)
        if configured != "auto":
            return Selection(configured, endpoint.profile)
        listed = await self._catalog.resolve((key, connection.revision), connection.http)
        return Selection(listed.id, profile_of(listed, endpoint.profile))

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        use: str = "generic",
    ) -> LLMCompletion:
        return await self._chat(
            [m.model_dump() for m in messages],
            use=use,
            max_tokens=min(max_tokens, self.tuning.max_tokens),
            temperature=temperature,
            source=messages,
        )

    async def structured(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        use: str = "generic",
    ) -> dict[str, Any]:
        turns = [m.model_dump() for m in messages]
        async with self._endpoint() as (connection, key):
            selection = await self._select(use, connection, key)
        model = selection.model
        response_format: dict[str, Any] | None = None
        if model not in self._schema_in_prompt:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            }
        else:
            # This model already refused the envelope once; do not pay the 400 again.
            turns = [*turns, _json_only_instruction(schema)]
        last_error: Exception | None = None
        # One bounded repair round: feed the validation error back once. A refused envelope
        # is not an attempt: the fallback asks the same question again with the schema in
        # the prompt and keeps its repair round. It used to consume it, so an invalid first
        # JSON after the fallback raised where the same output from a constrained model
        # would have been repaired.
        repairs_left = 1
        for _ in range(4):  # refusal + answer + repair, with margin; never unbounded
            try:
                # Omit the key entirely on the fallback attempt. Passing
                # ``response_format=None`` still puts ``"response_format": null`` on the
                # wire, which the provider that refused the envelope refuses again.
                envelope: dict[str, Any] = (
                    {"response_format": response_format} if response_format else {}
                )
                completion = await self._chat(
                    turns,
                    use=use,
                    selected=selection,
                    max_tokens=min(max_tokens, self.tuning.max_tokens),
                    source=messages,
                    **envelope,
                )
            except LLMCallFailed as exc:
                # Not every model behind the gateway implements constrained decoding.
                # DeepSeek answers `400 "This response_format type is unavailable now"`,
                # and because the envelope is attached to every structured call, one
                # unsupported model turned *all* of them into hard failures — 288 of 304 in
                # one benchmark run. Ask for the JSON in the prompt instead and parse it;
                # the validate-and-repair loop below is unchanged, so the contract the
                # caller relies on is the same either way.
                if response_format is None or not _unsupported_response_format(exc):
                    raise
                log.info("llm.structured_fallback", use=use, reason="response_format_unsupported")
                self._schema_in_prompt.add(model)
                response_format = None
                turns = [*turns, _json_only_instruction(schema)]
                continue
            text = completion.text
            try:
                parsed = _parse_json(text)
                _validate(parsed, schema)
                return parsed
            except LLMOutputInvalid as exc:
                last_error = exc
                if repairs_left == 0:
                    break
                repairs_left -= 1
                turns = [
                    *turns,
                    {"role": "assistant", "content": text[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "That output was invalid: "
                            f"{exc.message}. Return only JSON matching the schema."
                        ),
                    },
                ]
        llm_requests_total.labels(use, "invalid_output").inc()
        raise last_error or LLMOutputInvalid("structured output invalid")

    async def ping(self) -> bool:
        """The service's own endpoint answers ``/models`` (False without one)."""
        own = self._connections.get("service")
        return own is not None and await own.gateways[wire_profile(own.endpoint.profile)].ping()

    async def close(self) -> None:
        await self._catalog.close()
        connections = tuple(self._connections.values())
        self._connections.clear()
        for connection in connections:
            await connection.close()

    async def _validate_attempt(self, request: httpx.Request) -> None:
        binding = self._credential_attempt.get()
        if binding is not None and self.credentials is not None:
            await self.credentials.confirm(*binding)

    # ------------------------------------------------------------------ endpoints
    def _open(self, endpoint: Endpoint, revision: int | None) -> _Connection:
        route = route_of(endpoint)
        http = _http_client(endpoint, self.tuning.timeout_seconds, self._http)
        # httpx invokes request hooks for every SDK retry, after its backoff. This keeps
        # retries in the shared SDK while denying a newly retired endpoint before send.
        http.event_hooks["request"].append(self._validate_attempt)
        connection = _Connection(endpoint, route, revision, http)
        self._gateway(connection, endpoint.profile)
        return connection

    def _gateway(self, connection: _Connection, profile: ModelProfile) -> Bifrost:
        """The sdk client over ``connection`` that shapes requests for ``profile``."""
        wire = wire_profile(profile)
        gateway = connection.gateways.get(wire)
        if gateway is None:
            gateway = Bifrost(
                connection.endpoint.base_url,
                gateway=connection.route.gateway,
                profile=wire,
                timeout=self.tuning.timeout_seconds,
                max_retries=self.tuning.max_retries,
                backoff_seconds=self.transport.retry_backoff_seconds,
                max_tokens=self.tuning.max_tokens,
                circuit_failure_threshold=self.transport.circuit_failure_threshold,
                circuit_open_seconds=self.transport.circuit_open_seconds,
                client=connection.http,
            )
            connection.gateways[wire] = gateway
        return gateway

    def _remember(self, key: Hashable, connection: _Connection) -> None:
        self._connections[key] = connection
        self._connections.move_to_end(key)

    async def _retire(self, key: Hashable) -> None:
        connection = self._connections.pop(key, None)
        if connection is None:
            return
        connection.retired = True
        if connection.users == 0:
            await connection.close()

    async def _connection(self, choice: EndpointChoice) -> tuple[_Connection, Hashable]:
        """The open client for ``choice``'s endpoint: one per registered row (replaced when
        the row's revision moves) and one for the service's own."""
        credential = choice.credential
        key: Hashable = credential.identity if credential is not None else "service"
        revision = credential.revision if credential is not None else None
        current = self._connections.get(key)
        if current is not None and current.revision != revision:
            await self._retire(key)
            current = None
        if current is None:
            assert choice.endpoint is not None
            try:
                current = self._open(choice.endpoint, revision)
            except (ValueError, ImportError) as exc:
                raise ProviderNotConfigured(
                    f"The {choice.source} endpoint cannot be used: {exc}"
                ) from exc
            self._remember(key, current)
            while len(self._connections) > MAX_ENDPOINT_CLIENTS:
                await self._retire(next(iter(self._connections)))
        self._connections.move_to_end(key)
        return current, key

    @asynccontextmanager
    async def _endpoint(self) -> AsyncIterator[tuple[_Connection, Hashable]]:
        """Resolve the call's endpoint, hold its client for the call, and reject the result
        if the registration it was resolved from changed meanwhile."""
        identity = current_model_identity()
        choice = await self.resolver.resolve(identity)
        if choice.endpoint is None:
            raise ProviderNotConfigured(choice.reason, details={"source": choice.source})
        # Confirmed only while registrations are honoured: with accept_agent_endpoints off a
        # registration appearing mid-call changes nothing about where the call goes.
        binding = (
            (identity, choice.credential)
            if identity is not None
            and self.credentials is not None
            and self.settings.accept_agent_endpoints
            else None
        )
        connection, key = await self._connection(choice)
        connection.users += 1
        token = self._credential_attempt.set(binding)
        try:
            yield connection, key
            if binding is not None and self.credentials is not None:
                await self.credentials.confirm(*binding)
        finally:
            self._credential_attempt.reset(token)
            connection.users -= 1
            if connection.retired and connection.users == 0:
                await connection.close()

    # ------------------------------------------------------------------ internals
    async def _chat(
        self,
        turns: list[dict[str, Any]],
        *,
        use: str,
        max_tokens: int,
        temperature: float = 0.0,
        source: Sequence[LLMMessage],
        selected: Selection | None = None,
        **extra: Any,
    ) -> LLMCompletion:
        """One gateway call, in this service's error vocabulary.

        Returns the extracted completion rather than the raw payload: both callers want the
        text, and the outcome recorded here depends on whether there is any.
        """
        async with self._endpoint() as (connection, key):
            choice = selected or await self._select(use, connection, key)
            data, model, started, current = await self._send(
                connection, choice, turns, use, max_tokens, temperature, extra
            )
            # Extract before recording, because the outcome is not known until the content
            # is. A reasoning model that spends its whole budget thinking answers 200 with
            # an empty string, and counting that ok — which is what recording here used to
            # do — made the error rate say the feature was healthy while every caller saw
            # it fail.
            try:
                completion = self._completion(data)
            except LLMCallFailed:
                # Not a gateway problem and not transient: the budget is too small for this
                # model and will be next time too. The shared client already recorded the
                # call as a success, so this does not reach the breaker — which is right:
                # tripping on it would take out the uses that are working.
                self._failure(use, "empty_output")
                raise
            tokens = self._success(use, data, time.perf_counter() - started, current, model, source)
        await self._record_usage(use, tokens)
        return completion

    async def _send(
        self,
        connection: _Connection,
        choice: Selection,
        turns: list[dict[str, Any]],
        use: str,
        max_tokens: int,
        temperature: float,
        extra: dict[str, Any],
    ) -> tuple[dict[str, Any], str, float, Any]:
        """One request on ``connection``, shaped by ``choice``'s profile: its output ceiling
        caps the budget, the sdk sends a reasoning model ``max_completion_tokens`` and no
        temperature, and ``tool_choice`` goes only to a model that takes tools."""
        model, profile = choice.model, choice.profile
        if profile.max_output_tokens is not None:
            max_tokens = min(max_tokens, profile.max_output_tokens)
        if profile.tool_calling:
            extra = {"tool_choice": "none", **extra}
        gateway = self._gateway(connection, profile)
        started = time.perf_counter()
        with span("llm.chat", use=use, model=model) as current:
            try:
                data = await gateway.complete(
                    turns,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    options=connection.route.options,
                    **extra,
                )
            except CircuitOpen as exc:
                # Nothing was sent: the shared client's breaker is open after consecutive
                # failures, so this call fails immediately instead of paying the timeout.
                llm_requests_total.labels(use, "circuit_open").inc()
                log.warning("llm.circuit_open", use=use, retry_after=exc.retry_after)
                raise DependencyUnavailable("llm circuit open", details=exc.details) from exc
            except RateLimited as exc:
                # Backpressure, not brokenness. The retries are already spent by the time
                # this surfaces; the shared breaker deliberately does not count it.
                self._failure(use, "exhausted")
                # Carry how long to wait. The SDK parses it — Gemini puts the delay in the
                # response body rather than a Retry-After header — and stores it on the
                # exception, but `details` is only whatever the gateway literally said, so
                # forwarding that alone dropped it. Nothing above this adapter could tell
                # "come back in 30 seconds" from "this is broken", which is the difference
                # between backpressure and an outage. The key matches the one the circuit
                # breaker already uses above.
                raise LLMCallFailed(
                    "llm gateway rate limited the request",
                    details={
                        **exc.details,
                        **(
                            {"retry_after_seconds": round(exc.retry_after, 1)}
                            if exc.retry_after is not None
                            else {}
                        ),
                    },
                ) from exc
            except Unreachable as exc:
                self._failure(use, "exhausted")
                raise DependencyUnavailable(str(exc)) from exc
            except GatewayError as exc:
                status = exc.details.get("status")
                # A retryable status only reaches here once the client has given up on it;
                # anything else was refused on the first attempt and is named by its status.
                if status is None:
                    outcome = "bad_payload"
                elif status in RETRYABLE:
                    outcome = "exhausted"
                else:
                    outcome = f"http_{status}"
                self._failure(use, outcome)
                raise LLMCallFailed(str(exc), details=exc.details) from exc
        return data, model, started, current

    async def _record_usage(self, use: str, tokens: int) -> None:
        """Count the call against its tenant's day: one upsert, never fatal to the call."""
        identity = current_model_identity()
        if self.usage is None or identity is None:
            return
        try:
            await self.usage.record(identity.tenant_id, use, tokens)
        except Exception as exc:
            log.warning("llm.usage_not_recorded", use=use, error=type(exc).__name__)

    def _success(
        self,
        use: str,
        data: dict[str, Any],
        elapsed: float,
        current: Any,
        model: str,
        messages: Sequence[LLMMessage],
    ) -> int:
        """Record a successful call; returns its total tokens."""
        usage = data.get("usage") or {}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        identity = current_model_identity()
        tenant = identity.tenant_id if identity is not None else "operator"
        llm_requests_total.labels(use, "ok").inc()
        llm_seconds.labels(use).observe(elapsed)
        if in_tok:
            llm_tokens_total.labels(tenant, use, "input").inc(int(in_tok))
        if out_tok:
            llm_tokens_total.labels(tenant, use, "output").inc(int(out_tok))
        record_llm_tokens(in_tok, out_tok)
        current.set_attribute("llm.input_tokens", int(in_tok or 0))
        current.set_attribute("llm.output_tokens", int(out_tok or 0))
        current.set_attribute("llm.latency_ms", round(elapsed * 1000, 1))
        fields: dict[str, Any] = {
            "use": use,
            "model": data.get("model") or model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": round(elapsed * 1000, 1),
        }
        if self.log_source_text:
            # Read defensively rather than through _completion(): that raises on an
            # exhausted output budget, and raising *here* would mean the failure came out
            # of the success path — after the metrics above, outside every except clause
            # in _chat, so none of the failure accounting ran. Turning logging on to
            # investigate a problem must not change how that problem presents.
            fields["prompt"] = [m.content for m in messages]
            fields["response"] = _text_or_none(data)
        log.info("llm.call", **fields)
        return int(in_tok or 0) + int(out_tok or 0)

    def _failure(self, use: str, outcome: str) -> None:
        """Record a failed call under the outcome that caused it.

        Whether it also opens the circuit is the shared client's decision now, and the rule
        it applies is the one this service learned: a 429 is the gateway healthy and asking
        for less, not the gateway broken. Counting it converted "slow down" into "stop" —
        measured on a real run, 17 rate limits tripped the breaker and the next 62 calls
        failed with "circuit open" having sent nothing at all.
        """
        llm_requests_total.labels(use, outcome).inc()

    @staticmethod
    def _completion(data: dict[str, Any]) -> LLMCompletion:
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
            if message.get("tool_calls") or message.get("function_call"):
                raise LLMCallFailed("memory model response unexpectedly requested a tool")
            content = message.get("content")
            if isinstance(content, list):  # some gateways return content parts
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMCallFailed("llm gateway returned no choices") from exc
        usage = data.get("usage") or {}
        # Reasoning models spend the output budget on thinking before they emit anything, so
        # a too-small `max_tokens` comes back 200 OK with an empty string and a `length`
        # finish reason. Measured against gemini-3.6-flash: "Reply with exactly: OK" burned 57
        # reasoning tokens, so max_tokens=16 produced no content at all. Returning "" here let
        # that reach every call site as a silently degraded answer. Say what happened instead.
        if not content and choice.get("finish_reason") == "length":
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            raise LLMCallFailed(
                "llm returned no content: the output budget was exhausted before any text "
                "was produced (raise models.llm.max_tokens)",
                details={
                    "finish_reason": "length",
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": reasoning,
                },
            )
        return LLMCompletion(
            text=str(content or ""),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            model=data.get("model"),
        )


# ---------------------------------------------------------------------- JSON helpers


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    try:
        parsed = json.loads(stripped)
    except ValueError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise LLMOutputInvalid("output is not JSON") from None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except ValueError as exc:
            raise LLMOutputInvalid("output is not JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMOutputInvalid("output is not a JSON object")
    return parsed


_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Minimal JSON-schema check (type, required, properties, items, enum) — enough to reject
    malformed model output before it reaches domain code; no external validator needed."""
    typ = schema.get("type")
    if typ:
        expected = _TYPES.get(typ) if isinstance(typ, str) else None
        if expected is not None and not isinstance(value, expected):
            raise LLMOutputInvalid(f"{path}: expected {typ}")
        if typ == "integer" and isinstance(value, bool):
            raise LLMOutputInvalid(f"{path}: expected integer")
    if "enum" in schema and value not in schema["enum"]:
        raise LLMOutputInvalid(f"{path}: not in enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise LLMOutputInvalid(f"{path}: missing {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{i}]")
