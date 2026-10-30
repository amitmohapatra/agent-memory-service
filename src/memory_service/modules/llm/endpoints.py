"""Which model endpoint a call goes to: one resolver, its sources tried in order.

1. **registered** - the endpoint the acting agent registered (else its tenant's):
   "configure the harness only, memory follows". Skipped when the operator turned
   ``models.llm.accept_agent_endpoints`` off. A revoked row stops the chain: a revoked agent
   is refused, never quietly served by another endpoint.
2. **service** - the service's own endpoint (``LLM_BASE_URL`` and the other ``LLM_*``).
3. **none** - no model: the call raises ``ProviderNotConfigured`` (the ``DisabledLLM`` path),
   and every assisted feature keeps its native result.

The first source that answers wins, and the answer says why (``EndpointChoice.reason``), which
is what the model-key status routes report per agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from memory_service.config.settings import LLMSettings
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.ports.credentials import (
    AGENT_PRINCIPAL_PREFIX,
    CredentialResolver,
    ModelIdentity,
    ResolvedCredential,
)
from trellis.contracts import Endpoint

#: Where a call's endpoint came from: the agent's registration, its tenant's, the service's
#: own settings, or nowhere.
EndpointSource = Literal["agent", "tenant", "service", "none"]


@dataclass(frozen=True)
class EndpointChoice:
    source: EndpointSource
    #: why this source, in a sentence: what the status routes and the refusal say
    reason: str
    #: None when the call is refused (``source == "none"``)
    endpoint: Endpoint | None = field(default=None, repr=False)
    #: the registered row the endpoint came from, which the call is confirmed against
    credential: ResolvedCredential | None = None


class Source(Protocol):
    async def choose(
        self, identity: ModelIdentity | None, passed: Sequence[str]
    ) -> EndpointChoice | str:
        """A choice, or why this source passes (the next one is asked)."""
        ...


class Registered:
    def __init__(self, credentials: CredentialResolver | None, *, accept: bool) -> None:
        self.credentials = credentials
        self.accept = accept

    async def choose(
        self, identity: ModelIdentity | None, passed: Sequence[str]
    ) -> EndpointChoice | str:
        if not self.accept:
            return (
                "models.llm.accept_agent_endpoints is false, so registered endpoints are not used"
            )
        if identity is None or self.credentials is None:
            return "the work is not bound to an agent or tenant"
        try:
            resolved = await self.credentials.resolve(identity)
        except ProviderNotConfigured as exc:
            return EndpointChoice("none", f"{exc.message}: calls are refused, not rerouted")
        if resolved is None:
            return "neither the agent nor its tenant registered an endpoint"
        level = resolved.identity.principal_id
        source: EndpointSource = "agent" if level.startswith(AGENT_PRINCIPAL_PREFIX) else "tenant"
        return EndpointChoice(
            source,
            f"the endpoint {'the agent' if source == 'agent' else 'the tenant'} registered "
            f"(revision {resolved.revision})",
            resolved.endpoint,
            resolved,
        )


class Service:
    def __init__(self, endpoint: Endpoint | None) -> None:
        self.endpoint = endpoint

    async def choose(
        self, identity: ModelIdentity | None, passed: Sequence[str]
    ) -> EndpointChoice | str:
        if self.endpoint is None:
            return "the service has no endpoint of its own (LLM_BASE_URL is unset)"
        return EndpointChoice(
            "service",
            f"the service's own endpoint (LLM_BASE_URL): {'; '.join(passed)}",
            self.endpoint,
        )


class Nothing:
    async def choose(
        self, identity: ModelIdentity | None, passed: Sequence[str]
    ) -> EndpointChoice | str:
        return EndpointChoice("none", f"no model endpoint: {'; '.join(passed)}")


class EndpointResolver:
    """The sources in order; the first that chooses wins."""

    def __init__(self, settings: LLMSettings, credentials: CredentialResolver | None) -> None:
        self.settings = settings
        self.sources: tuple[Source, ...] = (
            Registered(credentials, accept=settings.accept_agent_endpoints),
            Service(settings.endpoint),
            Nothing(),
        )

    async def resolve(self, identity: ModelIdentity | None) -> EndpointChoice:
        passed: list[str] = []
        for source in self.sources:
            answer = await source.choose(identity, passed)
            if isinstance(answer, EndpointChoice):
                return answer
            passed.append(answer)
        raise AssertionError("the last source always chooses")  # pragma: no cover
