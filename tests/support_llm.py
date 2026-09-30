"""Test helpers for LLM-assisted paths: a Bifrost adapter pointed at a respx-mocked gateway.

Usage::

    with mocked_gateway(['{"worthy": true}', '{"worthy": false}']) as gw:
        assist = gw.assist(uses=["query_expansion"])     # what bound work may use
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
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import Any

import httpx
import respx
from pydantic import SecretStr

from memory_service.adapters.models.llm import BifrostLLM
from memory_service.config.constants import LLMTransport
from memory_service.config.settings import LLMSettings, LLMUse
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.policy import DEFAULT_ACCESS, ModelAccess, ModelBinding, bind
from memory_service.ports.credentials import ModelIdentity

BASE = "http://bifrost.test/v1"
#: retries without sleeping: these tests count calls, not seconds
NO_BACKOFF = LLMTransport(retry_backoff_seconds=0.0)
#: what the mocked gateway lists on /models: the strong and the fast model it picks
STRONG = "gemini/gemini-3.8-pro"
FAST = "gemini/gemini-3.8-flash"
CATALOG = {"data": [{"id": STRONG}, {"id": FAST}]}


def llm_settings(**overrides: Any) -> LLMSettings:
    """The gateway at ``BASE`` with an operator key (so an unbound call can pay)."""
    base: dict[str, Any] = {"base_url": BASE, "api_key": SecretStr("vk-test")}
    base.update(overrides)
    return LLMSettings(**base)


class StaticPolicies:
    """What ``ModelPolicies`` resolves, fixed: these uses, reads assisted, a key that pays.
    Stands in for the tenant policy a test would otherwise have to store."""

    def __init__(
        self,
        uses: Sequence[str],
        *,
        read_assist: bool = True,
        has_key: bool = True,
        models: dict[str, str] | None = None,
    ) -> None:
        self.access_value = ModelAccess(frozenset(uses), read_assist, has_key, dict(models or {}))

    async def access(self, identity: ModelIdentity) -> ModelAccess:
        return self.access_value

    async def tenants_with_keys(self) -> list[str]:
        return []


def chat_response(content: str, *, model: str = STRONG) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }


@dataclass
class Gateway:
    route: respx.Route
    bindings: ExitStack

    def assist(self, uses: Sequence[LLMUse], **overrides: Any) -> LLMAssist:
        """An assist over the mocked gateway that may call the model for ``uses`` only: the
        work the code binds resolves that policy, and the test's own work is bound to it
        until the gateway closes (what a tenant policy naming ``uses`` does)."""
        settings = llm_settings(**overrides)
        policies = StaticPolicies(uses)
        self.bindings.enter_context(
            bind(ModelBinding(ModelIdentity("test", "service:test"), policies.access_value))
        )
        return LLMAssist(BifrostLLM(settings, transport=NO_BACKOFF), settings, policies)

    def prompts(self) -> list[dict[str, Any]]:
        """Request bodies sent to the gateway, in order."""
        return [json.loads(call.request.content) for call in self.route.calls]


@contextmanager
def mocked_gateway(
    replies: Sequence[str | dict[str, Any]] = (), *, failing: bool = False
) -> Iterator[Gateway]:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json=CATALOG))
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
        with ExitStack() as bindings:
            yield Gateway(route=route, bindings=bindings)


def bound_to(
    tenant_id: str,
    principal_id: str,
    *,
    has_key: bool = True,
    uses: Sequence[str] | None = None,
    models: dict[str, str] | None = None,
) -> AbstractContextManager[None]:
    """Bind model work to an identity without resolving it from the database: the default
    policy (or ``uses``/``models``), and (by default) a key that can pay - what
    ``LLMAssist.bound`` yields for an owner that registered one."""
    access = replace(
        DEFAULT_ACCESS,
        has_key=has_key,
        uses=frozenset(uses) if uses is not None else DEFAULT_ACCESS.uses,
        models=dict(models or {}),
    )
    return bind(ModelBinding(ModelIdentity(tenant_id, principal_id), access))
