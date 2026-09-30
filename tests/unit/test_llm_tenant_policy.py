"""Per-tenant model use: the bound tenant's policy, gated on something that can pay; reads
assisted by the policy's read_assist; the model a use calls named by the policy; per-job
accounting and per-call usage recording."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest
import respx

from memory_service.adapters.models.llm import BifrostLLM
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.jobs.registry import _accounted
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.cost import llm_accounting, llm_tokens_used, record_llm_tokens
from memory_service.modules.llm.policy import (
    DEFAULT_ACCESS,
    ModelAccess,
    current_binding,
    model_calls_allowed,
)
from memory_service.observability.metrics import llm_tokens_total
from memory_service.ports.credentials import ModelIdentity
from tests.support_llm import BASE, NO_BACKOFF, bound_to, chat_response, llm_settings

pytestmark = pytest.mark.unit

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")


class _Policies:
    """Stands in for ModelPolicies: one resolved access for every identity."""

    def __init__(self, access: ModelAccess, keyed: list[str] | None = None) -> None:
        self.access_value = access
        self.keyed = keyed or []
        self.resolved: list[ModelIdentity] = []

    async def access(self, identity: ModelIdentity) -> ModelAccess:
        self.resolved.append(identity)
        return self.access_value

    async def tenants_with_keys(self) -> list[str]:
        return self.keyed


def _assist(access: ModelAccess = DEFAULT_ACCESS, **settings: Any) -> tuple[LLMAssist, _Policies]:
    cfg = llm_settings(**settings)
    policies = _Policies(access)
    return LLMAssist(BifrostLLM(cfg, transport=NO_BACKOFF), cfg, policies), policies  # type: ignore[arg-type]


async def test_wants_is_the_bound_policy() -> None:
    access = ModelAccess(frozenset({"summaries", "reflection"}), read_assist=True, has_key=True)
    assist, policies = _assist(access, api_key=None)
    async with assist.bound(ModelIdentity("acme", "user:u1")):
        assert assist.wants("summaries") and assist.wants("reflection")
        assert not assist.wants("query_expansion"), "the policy does not allow it"
    assert policies.resolved == [ModelIdentity("acme", "user:u1")]
    assert current_binding() is None, "the binding ends with the block"


async def test_a_call_needs_a_key_that_can_pay() -> None:
    keyless = ModelAccess(DEFAULT_ACCESS.uses, read_assist=True, has_key=False)
    assist, _ = _assist(keyless, api_key=None)
    assert not assist.wants("summaries"), "unbound and no operator key"
    async with assist.bound(ModelIdentity("acme", "user:u1")):
        assert not assist.wants("summaries"), "no key at any level"
    keyed, _ = _assist(replace(keyless, has_key=True), api_key=None)
    async with keyed.bound(ModelIdentity("acme", "user:u1")):
        assert keyed.wants("summaries")


async def test_the_operator_key_pays_for_unbound_work() -> None:
    assist, _ = _assist()
    assert assist.wants("summaries") and assist.wants("reflection")


async def test_nothing_is_resolved_when_no_model_is_reachable() -> None:
    policies = _Policies(DEFAULT_ACCESS)
    assist = LLMAssist(None, llm_settings(), policies)  # type: ignore[arg-type]
    async with assist.bound(ModelIdentity("acme", "user:u1")) as access:
        assert access == DEFAULT_ACCESS and not assist.wants("summaries")
    assert policies.resolved == [], "a model-free deployment pays no lookup"
    assert await assist.payable_tenants("summaries") == []


@pytest.mark.parametrize("read_assist", [True, False])
async def test_a_read_follows_the_policy_read_assist(read_assist: bool) -> None:
    access = ModelAccess(DEFAULT_ACCESS.uses, read_assist=read_assist, has_key=True)
    assist, policies = _assist(access)
    async with assist.reading(CTX):
        assert model_calls_allowed() is read_assist
        assert assist.wants("query_expansion") is read_assist
    assert policies.resolved == [ModelIdentity("acme", CTX.principal_id)]
    assert model_calls_allowed(), "the read's decision does not outlive it"


async def test_background_work_scans_only_tenants_something_can_pay_for() -> None:
    cfg = llm_settings(api_key=None)
    policies = _Policies(DEFAULT_ACCESS, keyed=["acme", "globex"])
    assist = LLMAssist(BifrostLLM(cfg, transport=NO_BACKOFF), cfg, policies)  # type: ignore[arg-type]
    assert await assist.payable_tenants("reflection") == ["acme", "globex"]
    assert await assist.payable_tenants("reflection", "globex") == ["globex"]
    assert await assist.payable_tenants("reflection", "initech") == []
    operator, _ = _assist()
    assert await operator.payable_tenants("reflection") == [None], "the operator pays: all"


@respx.mock
async def test_the_model_a_use_calls_is_the_one_the_policy_names() -> None:
    respx.get(f"{BASE}/models").mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "gemini/gemini-3.8-pro"}, {"id": "gemini/gemini-3.8-flash"}]}
        )
    )
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("ok"))
    )
    llm = BifrostLLM(llm_settings(), transport=NO_BACKOFF)
    with bound_to("acme", "user:u1", models={"grounding_judge": "openai/gpt-5-mini"}):
        await llm.complete([], use="grounding_judge")
        await llm.complete([], use="summaries")
        await llm.complete([], use="query_expansion")
    sent = [json.loads(call.request.content)["model"] for call in route.calls]
    assert sent == ["openai/gpt-5-mini", "gemini/gemini-3.8-pro", "gemini/gemini-3.8-flash"]
    await llm.close()


async def test_a_job_counts_its_own_tokens_and_leaves_the_request_counter_alone() -> None:
    seen: list[int] = []

    async def job(payload: dict[str, Any]) -> None:
        record_llm_tokens(7, 3)
        seen.append(llm_tokens_used())

    with llm_accounting() as request:
        record_llm_tokens(1, 1)
        await _accounted("memory.reflect", job)({"tenant_id": "acme"})
        assert request.total == 2, "the inline job counted on its own counter"
    assert seen == [10]


@respx.mock
async def test_every_successful_call_is_recorded_against_its_tenant() -> None:
    respx.get(f"{BASE}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gemini/gemini-3.8-pro"}]})
    )
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("ok"))
    )
    recorded: list[tuple[str, str, int]] = []

    class _Usage:
        async def record(self, tenant_id: str, use: str, tokens: int) -> None:
            recorded.append((tenant_id, use, tokens))

    llm = BifrostLLM(llm_settings(), transport=NO_BACKOFF, usage=_Usage())
    before = llm_tokens_total.labels("acme", "summaries", "input")._value.get()
    with bound_to("acme", "user:u1"):
        await llm.complete([], use="summaries")
    await llm.complete([], use="summaries")  # no identity: nothing to charge it to
    assert recorded == [("acme", "summaries", 30)]
    assert llm_tokens_total.labels("acme", "summaries", "input")._value.get() == before + 20
    await llm.close()


@respx.mock
async def test_a_failing_ledger_never_fails_the_call() -> None:
    respx.get(f"{BASE}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gemini/gemini-3.8-pro"}]})
    )
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("ok"))
    )

    class _Broken:
        async def record(self, tenant_id: str, use: str, tokens: int) -> None:
            raise RuntimeError("database down")

    llm = BifrostLLM(llm_settings(), transport=NO_BACKOFF, usage=_Broken())
    with bound_to("acme", "user:u1"):
        assert (await llm.complete([], use="summaries")).text == "ok"
    await llm.close()
