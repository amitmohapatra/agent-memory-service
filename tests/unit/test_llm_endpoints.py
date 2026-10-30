"""One resolution order for a model call's endpoint: the agent's registration (else its
tenant's), the service's own, none — and each answer says why."""

from __future__ import annotations

import pytest

from memory_service.config.settings import LLMSettings
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.modules.llm.endpoints import EndpointResolver
from memory_service.ports.credentials import ModelIdentity, ResolvedCredential, tenant_identity
from trellis.contracts import Endpoint

pytestmark = pytest.mark.unit

AGENT = ModelIdentity("acme", "agent:u1/research")
AGENT_ROW = ModelIdentity("acme", "agent:research")
HARNESS = Endpoint(base_url="https://api.openai.com/v1", model="gpt-4.1-mini", api_key="sk-a")
OWN = Endpoint(base_url="http://ollama:11434/v1", model="llama3.1")


class _Registered:
    """What ``ModelCredentials.resolve`` answers: a row, none, or a refusal."""

    def __init__(self, answer: ResolvedCredential | Exception | None) -> None:
        self.answer = answer
        self.asked: list[ModelIdentity] = []

    async def resolve(self, identity: ModelIdentity) -> ResolvedCredential | None:
        self.asked.append(identity)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    async def confirm(self, identity, resolved) -> None:  # pragma: no cover - not called
        return None


def _resolver(answer, *, own: Endpoint | None = OWN, accept: bool = True) -> EndpointResolver:
    return EndpointResolver(
        LLMSettings(endpoint=own, accept_agent_endpoints=accept), _Registered(answer)
    )


async def test_the_agent_s_registered_endpoint_comes_first() -> None:
    row = ResolvedCredential(HARNESS, 4, AGENT_ROW)
    choice = await _resolver(row).resolve(AGENT)
    assert (choice.source, choice.endpoint, choice.credential) == ("agent", HARNESS, row)
    assert choice.reason == "the endpoint the agent registered (revision 4)"


async def test_the_tenant_s_serves_an_agent_without_one() -> None:
    row = ResolvedCredential(HARNESS, 1, tenant_identity("acme"))
    choice = await _resolver(row).resolve(AGENT)
    assert choice.source == "tenant" and choice.endpoint == HARNESS
    assert "the tenant registered" in choice.reason


async def test_then_the_service_s_own_saying_why_the_registration_passed() -> None:
    choice = await _resolver(None).resolve(AGENT)
    assert (choice.source, choice.endpoint, choice.credential) == ("service", OWN, None)
    assert choice.reason == (
        "the service's own endpoint (LLM_BASE_URL): "
        "neither the agent nor its tenant registered an endpoint"
    )


async def test_then_none_and_the_call_is_refused() -> None:
    choice = await _resolver(None, own=None).resolve(AGENT)
    assert choice.source == "none" and choice.endpoint is None
    assert "LLM_BASE_URL is unset" in choice.reason


async def test_a_revoked_registration_refuses_rather_than_falls_through() -> None:
    revoked = ProviderNotConfigured("Model credential is revoked")
    choice = await _resolver(revoked).resolve(AGENT)
    assert choice.source == "none" and choice.endpoint is None
    assert choice.reason.startswith("Model credential is revoked: calls are refused")


async def test_the_operator_switch_skips_registrations_without_reading_them() -> None:
    row = ResolvedCredential(HARNESS, 1, AGENT_ROW)
    resolver = _resolver(row, accept=False)
    choice = await resolver.resolve(AGENT)
    assert choice.source == "service" and choice.endpoint == OWN
    assert "accept_agent_endpoints is false" in choice.reason
    assert resolver.sources[0].credentials.asked == []  # type: ignore[attr-defined]
    off = await _resolver(row, own=None, accept=False).resolve(AGENT)
    assert off.source == "none"


async def test_unbound_work_uses_the_service_s_own() -> None:
    choice = await _resolver(ResolvedCredential(HARNESS, 1, AGENT_ROW)).resolve(None)
    assert choice.source == "service" and "not bound" in choice.reason
