"""Automatic discovery stays credential-scoped, bounded, and model-call-free."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from bifrost_sdk import Options
from pydantic import SecretStr

from memory_service.adapters.models.catalog import ModelCatalog, choose_model
from memory_service.adapters.models.llm import BifrostLLM
from memory_service.config.settings import LLMSettings
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.policy import model_call_policy, model_identity
from memory_service.ports.credentials import ModelIdentity, ResolvedCredential

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    [
        "Qwen/Qwen3",
        "openrouter/deepseek/deepseek-chat",
        "custom/best",
        "openai/gpt-4o-qwen-distill",
        "microsoft/qwen-derivative",
    ],
)
def test_opaque_aliases_and_excluded_families_are_not_auto_selected(name):
    with pytest.raises(ProviderNotConfigured):
        choose_model((name,), fast=True)


def test_compact_and_synthesis_preferences_use_only_eligible_families():
    models = ("openrouter/openai/gpt-4.1-mini", "openrouter/openai/gpt-4.1", "Qwen/Qwen3")
    assert choose_model(models, fast=True).endswith("mini")
    assert choose_model(models, fast=False).endswith("4.1")


async def test_parallel_discovery_reuses_catalog_but_rotation_and_owners_do_not():
    calls = []

    async def response(request):
        calls.append(request.headers["x-bf-vk"])
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={"data": [{"id": "openai/gpt-4.1-mini"}]})

    async with httpx.AsyncClient(
        base_url="http://gateway.test/v1", transport=httpx.MockTransport(response)
    ) as client:
        catalog = ModelCatalog(client)
        alice = ModelIdentity("tenant", "alice")
        bob = ModelIdentity("tenant", "bob")
        try:
            await asyncio.gather(
                *(
                    catalog.resolve((alice, 1), Options(virtual_key="vk-alice"), fast=True)
                    for _ in range(8)
                )
            )
            await catalog.resolve((alice, 1), Options(virtual_key="vk-alice"), fast=False)
            assert calls == ["vk-alice"]
            await catalog.resolve((alice, 2), Options(virtual_key="vk-new"), fast=True)
            await catalog.resolve((bob, 1), Options(virtual_key="vk-bob"), fast=True)
            assert calls == ["vk-alice", "vk-new", "vk-bob"]
        finally:
            await catalog.close()


async def test_auto_without_a_key_or_with_read_denial_sends_no_http():
    transport = AsyncMock(side_effect=AssertionError("No gateway request is authorized"))
    async with httpx.AsyncClient(
        base_url="http://gateway.test/v1", transport=httpx.MockTransport(transport)
    ) as client:
        settings = LLMSettings(base_url="http://gateway.test/v1")
        provider = BifrostLLM(settings, client=client)
        assist = LLMAssist(provider, settings)
        try:
            assert not assist.wants("contextual_extraction")
            with model_identity("tenant", "agent"):
                assert await assist.complete("contextual_extraction", system="s", user="u") is None
                with model_call_policy(False):
                    assert not assist.wants("query_expansion")
            transport.assert_not_called()
        finally:
            await provider.close()


async def test_registered_key_auto_discovers_then_calls_with_no_mcp():
    paths = []

    def response(request):
        paths.append(request.url.path)
        assert request.headers["x-bf-vk"] == "vk-owner"
        assert request.headers["authorization"] == "Bearer vk-owner"
        assert request.headers["x-bf-mcp-include-clients"] == ""
        assert request.headers["x-bf-mcp-include-tools"] == ""
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-4.1-mini"}]})
        import json

        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-4.1-mini"
        assert body["tool_choice"] == "none" and "tools" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "accepted"}}]})

    credentials = AsyncMock()
    resolved = ResolvedCredential(SecretStr("vk-owner"), 3, ModelIdentity("tenant", "agent"))
    credentials.resolve.return_value = resolved
    async with httpx.AsyncClient(
        base_url="http://gateway.test/v1", transport=httpx.MockTransport(response)
    ) as client:
        settings = LLMSettings(base_url="http://gateway.test/v1")
        provider = BifrostLLM(settings, client=client, credentials=credentials)
        assist = LLMAssist(provider, settings)
        try:
            with model_identity("tenant", "agent"):
                for _ in range(2):
                    assert (
                        await assist.complete("contextual_extraction", system="s", user="u")
                        == "accepted"
                    )
            assert paths == ["/v1/models", "/v1/chat/completions", "/v1/chat/completions"]
            # the call is confirmed against the row and revision it ran under
            credentials.confirm.assert_awaited_with(ModelIdentity("tenant", "agent"), resolved)
        finally:
            await provider.close()
