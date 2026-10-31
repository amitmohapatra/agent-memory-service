"""Conformance: every registered route, auth and profile strategy, against a stub of each
wire format, through the one adapter the service calls models with.

The parameters are read from the registries themselves (``ROUTES``,
``bifrost_sdk.AUTH_STRATEGIES``), so a strategy added without a stub here fails this suite
rather than shipping untested. Each stub checks what its real counterpart would:

- the OpenAI chat format, and the Anthropic- and Gemini-compatible ones (content as text or as
  parts, reasoning tokens in the usage) and Bifrost's (its ``extra_fields``);
- the Bifrost route's per-call headers, and that no other endpoint gets an ``x-bf-*`` header;
- a bearer key; a Google access token, refreshed once it expires; an AWS SigV4 signature,
  recomputed by the stub from the request it received, body included;
- the request's shape from the profile: ``max_completion_tokens`` and no ``temperature`` for
  a reasoning model or one that takes no temperature, the output ceiling, and never a
  ``tool_choice`` (no memory call sends tools, and an endpoint may refuse one without them).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from bifrost_sdk import AUTH_STRATEGIES

from memory_service.adapters.models.llm import ROUTES, EndpointLLM, HttpEndpointProbe
from memory_service.config.constants import LLM, LLMTransport
from memory_service.config.settings import LLMSettings
from memory_service.domain.errors import ValidationFailed
from memory_service.ports.models import LLMMessage
from trellis.contracts import Endpoint, ModelProfile

pytestmark = pytest.mark.contract

BASE = "http://models.test/v1"
NO_BACKOFF = LLMTransport(retry_backoff_seconds=0.0)
AWS_KEY, AWS_SECRET, AWS_REGION = (
    "AKIDEXAMPLE",
    "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
    "us-east-1",
)

# ----------------------------------------------------------------------------- wire formats


def _openai(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


def _anthropic_compatible(text: str) -> dict[str, Any]:
    """Anthropic's OpenAI SDK compatibility layer: the OpenAI shape, Anthropic's ids."""
    return {
        "id": "msg_01",
        "object": "chat.completion",
        "model": "claude-sonnet",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }


def _gemini_compatible(text: str) -> dict[str, Any]:
    """Gemini's OpenAI-compatible endpoint, as some gateways relay it: content as parts and
    the thinking budget in the usage."""
    return {
        "object": "chat.completion",
        "model": "gemini-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
    }


def _bifrost(text: str) -> dict[str, Any]:
    return {**_openai(text), "extra_fields": {"provider": "openai", "latency": 12}}


WIRES: dict[str, Callable[[str], dict[str, Any]]] = {
    "openai": _openai,
    "anthropic-compatible": _anthropic_compatible,
    "gemini-compatible": _gemini_compatible,
    "bifrost": _bifrost,
}

# ----------------------------------------------------------------------------- profiles

PROFILES: dict[str, ModelProfile] = {
    "plain chat": ModelProfile(),
    "reasoning": ModelProfile(reasoning_output=True),
    "no temperature": ModelProfile(temperature=False),
    "ceiling": ModelProfile(max_output_tokens=64),
}

# ----------------------------------------------------------------------------- auth stubs


class _GoogleCredentials:
    """``google.auth.default()``'s credentials: a token that expires and is refreshed."""

    def __init__(self) -> None:
        import google.auth.credentials

        class Credentials(google.auth.credentials.Credentials):
            refreshes = 0

            def refresh(self, request: Any) -> None:
                type(self).refreshes += 1
                self.token = f"ya29.token-{type(self).refreshes}"
                self.expiry = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=1)

        self.credentials = Credentials()

    def expire(self) -> None:
        self.credentials.expiry = dt.datetime(2000, 1, 1)


def _sigv4_signature(request: httpx.Request) -> str:
    """The signature AWS computes for ``request``, from what arrived: method, path, the
    signed headers' values, the body's hash, the date and the scope."""
    auth = request.headers["authorization"]
    fields = dict(part.split("=", 1) for part in auth.removeprefix("AWS4-HMAC-SHA256 ").split(", "))
    signed = fields["SignedHeaders"].split(";")
    date = request.headers["x-amz-date"]
    scope = f"{date[:8]}/{AWS_REGION}/bedrock/aws4_request"
    assert fields["Credential"] == f"{AWS_KEY}/{scope}"
    canonical_headers = "".join(f"{name}:{request.headers[name].strip()}\n" for name in signed)
    canonical = "\n".join(
        [
            request.method,
            request.url.path,
            request.url.query.decode(),
            canonical_headers,
            fields["SignedHeaders"],
            hashlib.sha256(request.content).hexdigest(),
        ]
    )
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = f"AWS4{AWS_SECRET}".encode()
    for part in (date[:8], AWS_REGION, "bedrock", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()


def _check_bearer(request: httpx.Request, state: dict[str, Any]) -> None:
    assert request.headers["authorization"] == "Bearer sk-conformance"


def _check_google(request: httpx.Request, state: dict[str, Any]) -> None:
    token = request.headers["authorization"].removeprefix("Bearer ")
    assert token.startswith("ya29.token-")
    state.setdefault("tokens", []).append(token)


def _check_aws(request: httpx.Request, state: dict[str, Any]) -> None:
    sent = request.headers["authorization"].rsplit("Signature=", 1)[1]
    assert sent == _sigv4_signature(request), "the signature does not cover what was sent"
    assert "content-type" in request.headers["authorization"]
    state["signed"] = state.get("signed", 0) + 1


AUTH_CHECKS: dict[str, Callable[[httpx.Request, dict[str, Any]], None]] = {
    "bearer": _check_bearer,
    "google": _check_google,
    "aws": _check_aws,
}


@pytest.fixture
def google(monkeypatch: pytest.MonkeyPatch) -> Iterator[_GoogleCredentials]:
    import google.auth

    stub = _GoogleCredentials()
    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (stub.credentials, "proj"))
    yield stub


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for name, value in {
        "AWS_ACCESS_KEY_ID": AWS_KEY,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET,
        "AWS_DEFAULT_REGION": AWS_REGION,
        "AWS_CONFIG_FILE": str(tmp_path / "none"),
        "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "none"),
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)


def test_every_registered_strategy_has_a_stub() -> None:
    assert set(AUTH_CHECKS) == set(AUTH_STRATEGIES)
    assert set(ROUTES) == {"openai-compatible", "bifrost"}
    assert {route.gateway for route in ROUTES.values()} == {"openai-compatible", "bifrost"}


# ----------------------------------------------------------------------------- the matrix


def _endpoint(route: str, auth: str, profile: ModelProfile) -> Endpoint:
    return Endpoint(
        base_url=BASE,
        gateway=route,
        auth=auth,
        model="conformance-model",
        api_key="sk-conformance" if auth == "bearer" else None,
        profile=profile,
    )


@pytest.mark.parametrize("wire", sorted(WIRES))
@pytest.mark.parametrize("profile", sorted(PROFILES))
@pytest.mark.parametrize("auth", sorted(AUTH_STRATEGIES))
@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_a_memory_call_conforms_on_every_route_auth_profile_and_wire(
    route: str, auth: str, profile: str, wire: str, google: _GoogleCredentials, aws: None
) -> None:
    shape = PROFILES[profile]
    state: dict[str, Any] = {}
    bodies: list[dict[str, Any]] = []

    def endpoint(request: httpx.Request) -> httpx.Response:
        AUTH_CHECKS[auth](request, state)
        bf = {k: v for k, v in request.headers.items() if k.lower().startswith("x-bf-")}
        if route == "bifrost":
            assert bf == {
                "x-bf-disable-content-logging": "true",
                "x-bf-mcp-include-clients": "",
                "x-bf-mcp-include-tools": "",
            }
        else:
            assert bf == {}, "a Bifrost header went to another endpoint"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=WIRES[wire]('{"answer": "pong"}'))

    llm = EndpointLLM(
        LLMSettings(endpoint=_endpoint(route, auth, shape)),
        http=httpx.MockTransport(endpoint),
        transport=NO_BACKOFF,
        tuning=LLM,
    )
    try:
        out = await llm.complete([LLMMessage(role="user", content="ping")], max_tokens=200)
        assert out.text == '{"answer": "pong"}'
        assert (out.input_tokens, out.output_tokens) == (11, 3)
        if auth == "google":
            google.expire()
        parsed = await llm.structured(
            [LLMMessage(role="user", content="ping")],
            schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            max_tokens=200,
        )
        assert parsed == {"answer": "pong"}
    finally:
        await llm.close()

    budget = min(200, shape.max_output_tokens or 200)
    for body in bodies:
        assert body["model"] == "conformance-model"
        if shape.reasoning_output or not shape.temperature:
            assert body["max_completion_tokens"] == budget and "max_tokens" not in body
            assert "temperature" not in body
        else:
            assert body["max_tokens"] == budget and body["temperature"] == 0.0
        # no tools, so no tool_choice: an endpoint may refuse one without tools (400)
        assert "tool_choice" not in body and "tools" not in body
    assert "response_format" in bodies[1]
    if auth == "google":
        assert state["tokens"] == ["ya29.token-1", "ya29.token-2"], "refreshed once expired"
    if auth == "aws":
        assert state["signed"] == 2


# ----------------------------------------------------------------------------- registration


def _probe(handler: Callable[[httpx.Request], httpx.Response]) -> HttpEndpointProbe:
    return HttpEndpointProbe(http=httpx.MockTransport(handler))


@pytest.mark.parametrize("auth", sorted(AUTH_STRATEGIES))
async def test_the_probe_lists_models_with_the_endpoint_s_own_auth(
    auth: str, google: _GoogleCredentials, aws: None
) -> None:
    seen: list[str] = []

    def endpoint(request: httpx.Request) -> httpx.Response:
        AUTH_CHECKS[auth](request, {})
        seen.append(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    await _probe(endpoint)(_endpoint("openai-compatible", auth, ModelProfile()))
    assert seen == ["/v1/models"]


async def test_the_probe_refuses_an_endpoint_the_service_cannot_reach() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ValidationFailed, match="cannot reach") as raised:
        await _probe(unreachable)(Endpoint(base_url=BASE, api_key="k"))
    assert "loopback" in raised.value.message and raised.value.details["base_url"] == BASE


@pytest.mark.parametrize("status", [401, 404])
async def test_the_probe_refuses_an_endpoint_that_does_not_answer_models(status: int) -> None:
    with pytest.raises(ValidationFailed, match=f"answered {status}"):
        await _probe(lambda request: httpx.Response(status))(Endpoint(base_url=BASE, api_key="k"))


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"gateway": "litellm"}, "unknown gateway"),
        ({"auth": "kerberos"}, "unknown auth"),
        ({"auth": "google", "api_key": "k"}, "takes no key"),
    ],
)
async def test_the_probe_refuses_a_route_or_auth_it_has_no_strategy_for(
    fields: dict[str, str], why: str
) -> None:
    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing is sent for an endpoint that cannot be built")

    with pytest.raises(ValidationFailed, match=why):
        await _probe(never)(Endpoint(base_url=BASE, **fields))  # type: ignore[arg-type]


async def test_the_sigv4_stub_rejects_a_body_the_signature_does_not_cover(aws: None) -> None:
    """The stub above is not vacuous: change one byte after signing and it disagrees."""
    from bifrost_sdk import AwsSigV4Auth

    captured: list[httpx.Request] = []

    def endpoint(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(auth=AwsSigV4Auth(), transport=httpx.MockTransport(endpoint)) as c:
        await c.post(f"{BASE}/chat/completions", json={"model": "m"})
    signed = captured[0]
    sent = signed.headers["authorization"].rsplit("Signature=", 1)[1]
    assert sent == _sigv4_signature(signed)
    tampered = httpx.Request(
        signed.method, signed.url, headers=signed.headers, content=b'{"model":"x"}'
    )
    assert sent != _sigv4_signature(tampered)
