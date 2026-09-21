"""Contract tests for the Bifrost LLM adapter — the only LLM path.

The gateway is mocked with respx; the ``bifrost``-marked test at the end hits a running
Bifrost when ``MEMORY__MODELS__LLM__ENABLED=true`` and a model/virtual key are configured.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
import respx
from pydantic import SecretStr

from memory_service.adapters.models.llm import BifrostLLM, DisabledLLM, LLMOutputInvalid
from memory_service.config.settings import LLMSettings, Settings
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured
from memory_service.modules.llm.assist import LLMAssist
from memory_service.observability.metrics import llm_requests_total
from memory_service.ports.models import LLMMessage, LLMProvider

pytestmark = pytest.mark.contract

BASE = "http://bifrost.test/v1"
SCHEMA = {
    "type": "object",
    "properties": {"worthy": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["worthy"],
}


def _settings(**overrides: object) -> LLMSettings:
    base: dict[str, object] = {
        "enabled": True,
        "provider": "bifrost",
        "base_url": BASE,
        "api_key": SecretStr("vk-test"),
        "model": "openai/gpt-4.1",
        "fast_model": "openai/gpt-4.1-mini",
        "max_retries": 2,
        "retry_backoff_seconds": 0.0,
        "circuit_failure_threshold": 3,
        "circuit_open_seconds": 60,
        "uses": ["ambiguous_worthiness", "summaries"],
    }
    base.update(overrides)
    return LLMSettings(**base)  # type: ignore[arg-type]


def _chat(content: str, *, model: str = "openai/gpt-4.1", usage: bool = True) -> dict:
    body = {"model": model, "choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 12, "completion_tokens": 5}
    return body


def _messages() -> list[LLMMessage]:
    return [LLMMessage(role="system", content="s"), LLMMessage(role="user", content="u")]


def _counter(use: str, outcome: str) -> float:
    return llm_requests_total.labels(use, outcome)._value.get()  # noqa: SLF001


def test_protocol_conformance() -> None:
    assert isinstance(BifrostLLM(_settings()), LLMProvider)
    assert isinstance(DisabledLLM(), LLMProvider)


async def test_disabled_raises_provider_not_configured() -> None:
    with pytest.raises(ProviderNotConfigured):
        await DisabledLLM().complete(_messages())
    with pytest.raises(ProviderNotConfigured):
        await DisabledLLM().structured(_messages(), schema=SCHEMA)
    assert (await DisabledLLM().ping()) is False


def test_construction_guards() -> None:
    with pytest.raises(ProviderNotConfigured):
        BifrostLLM(LLMSettings(enabled=False))
    with pytest.raises(ProviderNotConfigured):
        BifrostLLM(LLMSettings(enabled=True, provider="bifrost", model=None))


def test_settings_reject_non_bifrost_provider() -> None:
    with pytest.raises(ValueError, match="bifrost"):
        Settings(models={"llm": {"enabled": True, "provider": "disabled"}})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model"):
        Settings(models={"llm": {"enabled": True, "provider": "bifrost", "model": None}})  # type: ignore[arg-type]
    defaults = LLMSettings()
    assert defaults.model.startswith("anthropic/claude") and defaults.fast_model.startswith(
        "anthropic/claude"
    )


@respx.mock
async def test_complete_sends_bearer_virtual_key_and_maps_usage() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat("hello"))
    )
    llm = BifrostLLM(_settings())
    before = _counter("generic", "ok")
    out = await llm.complete(_messages(), max_tokens=64, temperature=0.2)
    assert out.text == "hello" and out.input_tokens == 12 and out.output_tokens == 5
    assert out.model == "openai/gpt-4.1"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer vk-test"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-4.1" and body["max_tokens"] == 64
    assert body["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert _counter("generic", "ok") == before + 1
    await llm.close()


@respx.mock
async def test_fast_model_is_used_for_fast_uses() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat("x", model="openai/gpt-4.1-mini"))
    )
    llm = BifrostLLM(_settings())
    await llm.complete(_messages(), use="ambiguous_worthiness")
    assert json.loads(route.calls.last.request.content)["model"] == "openai/gpt-4.1-mini"
    await llm.complete(_messages(), use="summaries")
    assert json.loads(route.calls.last.request.content)["model"] == "openai/gpt-4.1"


@respx.mock
async def test_structured_requests_json_schema_and_validates() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat('{"worthy": true, "reason": "fact"}'))
    )
    llm = BifrostLLM(_settings())
    out = await llm.structured(_messages(), schema=SCHEMA, use="ambiguous_worthiness")
    assert out == {"worthy": True, "reason": "fact"}
    body = json.loads(route.calls.last.request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["temperature"] == 0.0


@respx.mock
async def test_structured_repairs_once_then_fails() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_chat("not json at all")),
            httpx.Response(200, json=_chat('```json\n{"worthy": "yes"}\n```')),
        ]
    )
    llm = BifrostLLM(_settings())
    with pytest.raises(LLMOutputInvalid):
        await llm.structured(_messages(), schema=SCHEMA)
    assert route.call_count == 2
    repair = json.loads(route.calls.last.request.content)["messages"]
    assert repair[-1]["role"] == "user" and "invalid" in repair[-1]["content"]

    route.side_effect = [
        httpx.Response(200, json=_chat('{"reason": "x"}')),  # missing required key
        httpx.Response(200, json=_chat('{"worthy": false}')),
    ]
    assert await llm.structured(_messages(), schema=SCHEMA) == {"worthy": False}


@respx.mock
async def test_retries_transient_status_then_succeeds() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(503, text="busy"),
            httpx.Response(429, text="slow down"),
            httpx.Response(200, json=_chat("ok")),
        ]
    )
    llm = BifrostLLM(_settings())
    assert (await llm.complete(_messages())).text == "ok"
    assert route.call_count == 3


@respx.mock
async def test_retries_are_bounded_and_non_retryable_status_fails_fast() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(502))
    llm = BifrostLLM(_settings(max_retries=1))
    with pytest.raises(DependencyUnavailable, match="502"):
        await llm.complete(_messages())
    assert route.call_count == 2

    route.return_value = httpx.Response(400, json={"error": "bad request"})
    with pytest.raises(DependencyUnavailable, match="400"):
        await llm.complete(_messages())
    assert route.call_count == 3  # no retry on 4xx other than 408/409/425/429


@respx.mock
async def test_timeouts_and_circuit_breaker() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    llm = BifrostLLM(_settings(max_retries=0, circuit_failure_threshold=2))
    for _ in range(2):
        with pytest.raises(DependencyUnavailable, match="unreachable"):
            await llm.complete(_messages())
    assert route.call_count == 2
    before = _counter("generic", "circuit_open")
    with pytest.raises(DependencyUnavailable, match="circuit open") as info:
        await llm.complete(_messages())
    assert route.call_count == 2  # failed fast, no request made
    assert info.value.details["retry_after_seconds"] > 0
    assert _counter("generic", "circuit_open") == before + 1


@respx.mock
async def test_bad_payloads_fail_loudly() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, text="<html>")
    )
    llm = BifrostLLM(_settings(max_retries=0))
    with pytest.raises(DependencyUnavailable, match="non-JSON"):
        await llm.complete(_messages())
    route.return_value = httpx.Response(200, json={"choices": []})
    with pytest.raises(DependencyUnavailable, match="no choices"):
        await llm.complete(_messages())


@respx.mock
async def test_usage_is_logged_without_prompt_text_by_default() -> None:
    # capture_logs only sees events when structlog is on its default configuration; another
    # test in the session may have reconfigured it, so reset first and restore afterwards.
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat("SECRET-ANSWER"))
    )
    with capture_logs() as events:
        await BifrostLLM(_settings()).complete(
            [LLMMessage(role="user", content="SECRET-PROMPT")], use="summaries"
        )
    call = next(e for e in events if e.get("event") == "llm.call")
    assert call["use"] == "summaries" and call["input_tokens"] == 12
    assert "SECRET-PROMPT" not in json.dumps(call) and "SECRET-ANSWER" not in json.dumps(call)
    with capture_logs() as events:
        await BifrostLLM(_settings(), log_source_text=True).complete(
            [LLMMessage(role="user", content="SECRET-PROMPT")]
        )
    call = next(e for e in events if e.get("event") == "llm.call")
    assert call["prompt"] == ["SECRET-PROMPT"] and call["response"] == "SECRET-ANSWER"


@respx.mock
async def test_ping_uses_models_endpoint() -> None:
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    assert await BifrostLLM(_settings()).ping() is True
    respx.get(f"{BASE}/models").mock(side_effect=httpx.ConnectError("down"))
    assert await BifrostLLM(_settings()).ping() is False


@respx.mock
async def test_assist_falls_back_to_none_on_any_failure() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(500))
    llm = BifrostLLM(_settings(max_retries=0))
    assist = LLMAssist(llm, _settings())
    assert assist.wants("ambiguous_worthiness") and not assist.wants("reflection")
    out = await assist.structured("ambiguous_worthiness", system="s", user="u", schema=SCHEMA)
    assert out is None and route.call_count == 1
    assert await assist.structured("reflection", system="s", user="u", schema=SCHEMA) is None
    assert route.call_count == 1  # a use that is not enabled never calls the gateway
    route.return_value = httpx.Response(200, json=_chat('{"worthy": true}'))
    assert await assist.structured("summaries", system="s", user="u", schema=SCHEMA) == {
        "worthy": True
    }
    assert LLMAssist.disabled().wants("summaries") is False


async def _rate_limited(settings) -> str | None:
    """Whether the provider is refusing traffic right now, and what it said."""
    async with httpx.AsyncClient(base_url=settings.base_url, timeout=30) as client:
        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": settings.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8,
                },
            )
        except httpx.HTTPError as exc:
            return f"gateway unreachable: {exc}"
    if response.status_code != 429 and '"code":"429"' not in response.text:
        return None
    # The gateway wraps the provider's message; the useful part is the tail of that message
    # ("... limit: 20 ... Please retry in 28.9s."), not the routing envelope around it.
    try:
        message = str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        message = response.text
    return message.strip().splitlines()[-1][-160:] or message[-160:]


@pytest.mark.bifrost
async def test_live_bifrost_roundtrip() -> None:
    """Hits the running gateway. Needs MEMORY__MODELS__LLM__ENABLED=true plus model and key."""
    settings = Settings().models.llm
    if not settings.enabled or not settings.model:
        pytest.skip("Bifrost not configured (MEMORY__MODELS__LLM__*)")
    llm = BifrostLLM(settings)
    if not await llm.ping():
        pytest.skip(f"Bifrost not reachable at {settings.base_url}")
    if (limited := await _rate_limited(settings)) is not None:
        # The provider refusing traffic is not a defect in this adapter. Established up
        # front rather than caught below: the SDK honours the delay Gemini puts in the
        # response *body*, so a rate-limited call no longer fails fast — it waits the ~30s
        # it was asked for, twice, and then surfaces as LLMCallFailed. Catching that would
        # mean either skipping on every gateway failure (hiding real ones) or reporting a
        # free tier's "limit: 20" as a bug in this code.
        pytest.skip(f"model provider is rate limited: {limited}")
    # max_tokens=8 was enough when every model emitted text immediately. A reasoning model
    # spends the output budget on thinking first — measured against gemini-3.6-flash, "Reply
    # with exactly: OK" consumed 57 reasoning tokens — so a small budget returns 200 OK with
    # an empty string and finish_reason="length". The adapter now raises on exactly that
    # rather than handing back "", which is what this test hit. Ask for a real budget.
    out = await llm.complete(
        [LLMMessage(role="user", content="Reply with the single word: pong")], max_tokens=1024
    )
    assert "pong" in out.text.lower()
    assert out.input_tokens and out.output_tokens
    data = await llm.structured(
        [
            LLMMessage(
                role="user",
                content="Is 'I prefer dark mode' worth remembering about a user? Answer in JSON.",
            )
        ],
        schema=SCHEMA,
    )
    assert data["worthy"] is True
    if os.environ.get("MEMORY_BIFROST_RECORD"):
        print({"model": out.model, "input_tokens": out.input_tokens})  # noqa: T201
    await llm.close()
