"""Test helpers for LLM-assisted paths: a Bifrost adapter pointed at a respx-mocked gateway.

Usage::

    with mocked_gateway(['{"worthy": true}', '{"worthy": false}']) as gw:
        assist = gw.assist(uses=["ambiguous_worthiness"])
        ...
        assert gw.route.call_count == 1
        assert gw.prompts()[0]["messages"][1]["content"].startswith("...")

Each string in ``replies`` is served, in order, as the assistant content of one chat
completion; ``failing=True`` makes the gateway answer 503 so the native fallback is
exercised.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import respx
from pydantic import SecretStr

from memory_service.adapters.models.llm import BifrostLLM
from memory_service.config.settings import LLMSettings, LLMUse
from memory_service.modules.llm.assist import LLMAssist

BASE = "http://bifrost.test/v1"


def llm_settings(uses: Sequence[LLMUse] = (), **overrides: Any) -> LLMSettings:
    base: dict[str, Any] = {
        "enabled": True,
        "base_url": BASE,
        "api_key": SecretStr("vk-test"),
        "model": "test/strong",
        "fast_model": "test/fast",
        "max_retries": 0,
        "retry_backoff_seconds": 0.0,
        "timeout_seconds": 5.0,
        "uses": list(uses),
    }
    base.update(overrides)
    return LLMSettings(**base)


def chat_response(content: str, *, model: str = "test/strong") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }


@dataclass
class Gateway:
    route: respx.Route

    def assist(self, uses: Sequence[LLMUse], **overrides: Any) -> LLMAssist:
        settings = llm_settings(uses, **overrides)
        return LLMAssist(BifrostLLM(settings), settings)

    def prompts(self) -> list[dict[str, Any]]:
        """Request bodies sent to the gateway, in order."""
        return [json.loads(call.request.content) for call in self.route.calls]


@contextmanager
def mocked_gateway(
    replies: Sequence[str | dict[str, Any]] = (), *, failing: bool = False
) -> Iterator[Gateway]:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/chat/completions")
        if failing:
            route.mock(return_value=httpx.Response(503, text="down"))
        elif replies:
            responses = [
                httpx.Response(
                    200,
                    json=chat_response(r if isinstance(r, str) else json.dumps(r)),
                )
                for r in replies
            ]
            # keep serving the last reply once the scripted ones are used up
            route.side_effect = [*responses, *([responses[-1]] * 50)]
        else:
            route.mock(return_value=httpx.Response(200, json=chat_response("{}")))
        yield Gateway(route=route)
