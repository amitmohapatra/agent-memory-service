"""``auto`` chooses by what an endpoint's ``/models`` says, never by a model's name; discovery
stays per endpoint revision, bounded, and model-call-free."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from memory_service.adapters.models.catalog import (
    ListedModel,
    ModelCatalog,
    choose_model,
    listed,
    profile_of,
)
from memory_service.adapters.models.llm import EndpointLLM
from memory_service.config.settings import LLMSettings
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.policy import model_call_policy
from memory_service.ports.credentials import ModelIdentity, ResolvedCredential
from tests.support_llm import bound_to
from trellis.contracts import Endpoint, ModelProfile

pytestmark = pytest.mark.unit

TEXT = {"architecture": {"output_modalities": ["text"]}}


def _rows(*rows: dict) -> list[ListedModel]:
    return [m for m in (listed(r) for r in rows) if m is not None]


@pytest.mark.parametrize(
    "row",
    [
        {"id": "gpt-4o"},  # bare names are fine: nothing is read from the name
        {"id": "llama3.1:8b", "type": "chat"},  # vLLM / Together
        {"id": "my-alias", "mode": "chat"},  # LiteLLM
        {"id": "anything", "architecture": {"modality": "text+image->text"}},
    ],
)
def test_any_name_is_chosen_when_its_listing_says_it_writes_text(row: dict) -> None:
    assert choose_model(_rows(row)).id == row["id"]


def test_a_listing_that_says_nothing_is_usable_only_with_one_model() -> None:
    assert choose_model(_rows({"id": "local-model"})).id == "local-model"  # llama.cpp
    with pytest.raises(ProviderNotConfigured, match="MEMORY__MODELS__LLM__MODEL"):
        choose_model(_rows({"id": "a"}, {"id": "b"}))  # Ollama, OpenAI: which one?


def test_models_that_do_not_write_text_and_excluded_origins_are_never_chosen() -> None:
    with pytest.raises(ProviderNotConfigured, match="writes text"):
        choose_model(
            _rows(
                {"id": "embed-1", "type": "embedding"},
                {"id": "image-1", "architecture": {"output_modalities": ["image"]}},
                {"id": "qwen/qwen3", **TEXT},
            )
        )


def test_among_text_models_the_cheapest_listed_is_chosen_and_unpriced_ones_are_ambiguous() -> None:
    rows = _rows(
        {"id": "big", **TEXT, "pricing": {"completion": "0.00006"}},
        {"id": "small", **TEXT, "pricing": {"completion": "0.0000006"}},
        {"id": "unpriced", **TEXT},
    )
    assert choose_model(rows).id == "small"
    with pytest.raises(ProviderNotConfigured, match="no price"):
        choose_model(_rows({"id": "a", **TEXT}, {"id": "b", **TEXT}))


def test_a_listing_s_profile_facts_shape_the_request() -> None:
    model = listed(
        {
            "id": "o-reasoner",
            **TEXT,
            "supported_parameters": ["reasoning", "max_tokens"],
            "top_provider": {"max_completion_tokens": 32000},
        }
    )
    assert model is not None
    profile = profile_of(model, ModelProfile())
    assert profile == ModelProfile(
        reasoning_output=True, temperature=False, max_output_tokens=32000
    )
    plain = listed({"id": "m", **TEXT})
    assert plain is not None and profile_of(plain, ModelProfile()) == ModelProfile()


async def test_parallel_discovery_is_shared_but_a_rotation_discovers_again() -> None:
    calls = []

    async def response(request):
        calls.append(request.headers["authorization"])
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={"data": [{"id": "gpt-4.1-mini"}]})

    catalog = ModelCatalog()

    def client(key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="http://endpoint.test/v1",
            headers={"Authorization": f"Bearer {key}"},
            transport=httpx.MockTransport(response),
        )

    alice = ModelIdentity("tenant", "agent:alice")
    async with client("k-alice") as a1, client("k-new") as a2, client("k-bob") as b:
        try:
            await asyncio.gather(*(catalog.resolve((alice, 1), a1) for _ in range(8)))
            await catalog.resolve((alice, 1), a1)
            assert calls == ["Bearer k-alice"]
            await catalog.resolve((alice, 2), a2)
            await catalog.resolve((ModelIdentity("tenant", "agent:bob"), 1), b)
            assert calls == ["Bearer k-alice", "Bearer k-new", "Bearer k-bob"]
        finally:
            await catalog.close()


async def test_auto_without_an_endpoint_or_with_read_denial_sends_no_http() -> None:
    transport = AsyncMock(side_effect=AssertionError("No endpoint request is authorized"))
    settings = LLMSettings()
    provider = EndpointLLM(settings, http=httpx.MockTransport(transport))
    assist = LLMAssist(provider, settings)
    try:
        assert not assist.wants("contextual_extraction")
        with bound_to("tenant", "agent", has_key=False):
            assert not assist.wants("contextual_extraction")
        with bound_to("tenant", "agent"), model_call_policy(False):
            assert not assist.wants("query_expansion")
        transport.assert_not_called()
    finally:
        await provider.close()


async def test_a_registered_endpoint_auto_discovers_then_calls_with_its_own_key() -> None:
    paths = []

    def response(request):
        paths.append((request.url.host, request.url.path))
        assert request.headers["authorization"] == "Bearer sk-owner"
        assert "x-bf-vk" not in request.headers  # no Bifrost header on another endpoint
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-4.1-mini"}]})
        body = json.loads(request.content)
        assert body["model"] == "gpt-4.1-mini"
        assert "tool_choice" not in body and "tools" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "accepted"}}]})

    credentials = AsyncMock()
    resolved = ResolvedCredential(
        Endpoint(base_url="https://agent-endpoint.test/v1", api_key="sk-owner"),
        3,
        ModelIdentity("tenant", "agent:agent"),
    )
    credentials.resolve.return_value = resolved
    settings = LLMSettings()
    provider = EndpointLLM(settings, http=httpx.MockTransport(response), credentials=credentials)
    assist = LLMAssist(provider, settings)
    try:
        with bound_to("tenant", "agent:agent"):
            for _ in range(2):
                assert (
                    await assist.complete("contextual_extraction", system="s", user="u")
                    == "accepted"
                )
        host = "agent-endpoint.test"
        assert paths == [
            (host, "/v1/models"),
            (host, "/v1/chat/completions"),
            (host, "/v1/chat/completions"),
        ]
        # the call is confirmed against the row and revision it ran under
        credentials.confirm.assert_awaited_with(ModelIdentity("tenant", "agent:agent"), resolved)
    finally:
        await provider.close()
