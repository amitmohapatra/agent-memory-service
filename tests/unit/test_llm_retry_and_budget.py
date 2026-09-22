"""Gateway failure modes that are silent by default.

Both were found by pointing the service at a real gateway with a reasoning model behind a
free-tier quota — the configuration most first-time users will have.
"""

from __future__ import annotations

import httpx
import pytest

from memory_service.adapters.models.llm import BifrostLLM, LLMCallFailed
from memory_service.config.constants import LLMTransport
from memory_service.config.settings import LLMSettings


def _settings(**over: object) -> LLMSettings:
    base = {
        "enabled": True,
        "base_url": "http://gateway/v1",
        "model": "gemini/gemini-3.6-flash",
        "max_retries": 1,
    }
    return LLMSettings(**{**base, **over})  # type: ignore[arg-type]


#: no backoff between retries: these tests count requests, not seconds
FAST = LLMTransport(retry_backoff_seconds=0.0)
#: the breaker opens after two failures so a test can reach it in three calls
TWO_STRIKES = LLMTransport(retry_backoff_seconds=0.0, circuit_failure_threshold=2)


def _llm(handler) -> BifrostLLM:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://gateway/v1")
    return BifrostLLM(_settings(), client=client, transport=FAST)


# ``Retry-After`` parsing moved to the shared gateway client when this service stopped
# carrying its own copy; its spellings are covered in bifrost-sdk's tests. What is still
# this service's own — the breaker, the metric outcomes, the empty-answer rule — is below.


@pytest.mark.anyio
async def test_an_empty_answer_from_an_exhausted_budget_is_an_error_not_an_empty_string() -> None:
    """A reasoning model can spend the whole output budget before emitting any text.

    The gateway returns 200 with content="" and finish_reason="length". Returning that as a
    completion pushed an empty answer into every call site as if the model had said nothing
    on purpose. Measured: gemini-3.6-flash burned 57 reasoning tokens answering "Reply with
    exactly: OK", so max_tokens=16 produced no content at all.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {
                    "completion_tokens": 16,
                    "completion_tokens_details": {"reasoning_tokens": 16},
                },
            },
        )

    from memory_service.ports.models import LLMMessage

    llm = _llm(handler)
    with pytest.raises(LLMCallFailed, match="output budget was exhausted"):
        await llm.complete([LLMMessage(role="user", content="hi")], max_tokens=16)
    await llm.close()


@pytest.mark.anyio
async def test_a_short_answer_that_simply_finished_is_not_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        )

    from memory_service.ports.models import LLMMessage

    llm = _llm(handler)
    result = await llm.complete([LLMMessage(role="user", content="hi")])
    assert result.text == "OK"
    await llm.close()


@pytest.mark.anyio
async def test_a_rate_limit_does_not_open_the_circuit() -> None:
    """429 is backpressure, not brokenness — the same rule as the harness client.

    Measured on a real judged benchmark run: 17 rate limits opened the breaker and the
    following 62 calls failed instantly with "circuit open", having sent nothing. The scores
    silently fell back to a different metric, and the run still claimed it had measured the
    thing the model was there to measure.
    """
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] <= 4:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}]}
        )

    from memory_service.ports.models import LLMMessage

    settings = _settings(max_retries=0)
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://gateway/v1")
    llm = BifrostLLM(settings, client=client, transport=TWO_STRIKES)

    for _ in range(4):
        with pytest.raises(Exception):  # noqa: B017 - either error type is acceptable here
            await llm.complete([LLMMessage(role="user", content="hi")])
    result = await llm.complete([LLMMessage(role="user", content="hi")])
    await llm.close()

    assert result.text == "recovered"
    assert seen["n"] == 5, "an open circuit would have stopped the fifth request being sent"


@pytest.mark.anyio
async def test_a_server_error_still_opens_the_circuit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    from memory_service.domain.errors import DependencyUnavailable
    from memory_service.ports.models import LLMMessage

    settings = _settings(max_retries=0)
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://gateway/v1")
    llm = BifrostLLM(settings, client=client, transport=TWO_STRIKES)

    for _ in range(2):
        with pytest.raises(Exception):  # noqa: B017
            await llm.complete([LLMMessage(role="user", content="hi")])
    with pytest.raises(DependencyUnavailable, match="circuit open"):
        await llm.complete([LLMMessage(role="user", content="hi")])
    await llm.close()
