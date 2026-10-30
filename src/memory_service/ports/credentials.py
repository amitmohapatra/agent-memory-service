"""Registered model endpoints, stored encrypted; plaintext never belongs to a memory or job.

An endpoint (``trellis.contracts.Endpoint``: where the model is, the route and auth names, the
model, its key and profile) is stored at one of two levels: the agent's (``agent:<agent_id>``,
whichever user it acts for - the harness registers it once, at startup) or the tenant's
(``tenant``). A call resolves the most specific row that exists: the acting agent's endpoint,
else the tenant's.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Protocol

from pydantic import SecretStr

from trellis.contracts import Endpoint

AGENT_PRINCIPAL_PREFIX: Final = "agent:"
TENANT_PRINCIPAL: Final = "tenant"


def agent_level(principal_id: str) -> str | None:
    """The agent level a principal resolves through: ``agent:<agent_id>`` for an agent, bound
    to a user (``agent:u1/research``) or not; None for anyone else."""
    if not principal_id.startswith(AGENT_PRINCIPAL_PREFIX):
        return None
    return (
        AGENT_PRINCIPAL_PREFIX
        + principal_id[len(AGENT_PRINCIPAL_PREFIX) :].rsplit("/", maxsplit=1)[-1]
    )


@dataclass(frozen=True)
class ModelIdentity:
    """Who a model call is for: the tenant and the acting principal."""

    tenant_id: str
    principal_id: str

    def levels(self) -> tuple["ModelIdentity", ...]:
        """The rows a resolution reads, most specific first: the agent's, then the
        tenant's."""
        agent = agent_level(self.principal_id)
        rows = [ModelIdentity(self.tenant_id, agent)] if agent else []
        rows.append(tenant_identity(self.tenant_id))
        return tuple(rows)


def tenant_identity(tenant_id: str) -> ModelIdentity:
    return ModelIdentity(tenant_id, TENANT_PRINCIPAL)


@dataclass(frozen=True)
class StoredCredential:
    identity: ModelIdentity
    key_id: str
    ciphertext: bytes | None = field(repr=False)
    revision: int
    updated_at: datetime


@dataclass(frozen=True)
class ResolvedCredential:
    endpoint: Endpoint = field(repr=False)
    revision: int
    #: The row the endpoint came from: the agent's or the tenant's.
    identity: ModelIdentity


class CredentialRepository(Protocol):
    async def get(self, identity: ModelIdentity) -> StoredCredential | None: ...

    async def first(self, levels: Sequence[ModelIdentity]) -> StoredCredential | None:
        """The row of the first level (most specific first) that has one, revoked or not:
        one indexed read for the whole resolution."""
        ...

    async def tenants_with_keys(self) -> list[str]:
        """Tenants with at least one live (non-revoked) key at any level."""
        ...

    async def put(
        self, identity: ModelIdentity, *, key_id: str, ciphertext: bytes | None
    ) -> StoredCredential: ...


class CredentialCipher(Protocol):
    def encrypt(self, identity: ModelIdentity, value: SecretStr) -> tuple[str, bytes]: ...

    def decrypt(self, record: StoredCredential) -> SecretStr: ...


class EndpointProbe(Protocol):
    async def __call__(self, endpoint: Endpoint) -> None:
        """Return when the service can reach ``endpoint`` (its ``/models`` answers with the
        endpoint's own auth); raise ``ValidationFailed`` saying why it cannot."""
        ...


class CredentialResolver(Protocol):
    async def resolve(self, identity: ModelIdentity) -> ResolvedCredential | None:
        """The registered endpoint of the most specific level that has a row; None when no
        level has one. A revoked row raises ``ProviderNotConfigured``."""
        ...

    async def confirm(self, identity: ModelIdentity, resolved: ResolvedCredential | None) -> None:
        """Refuse a result obtained under a credential that has since changed."""
        ...
