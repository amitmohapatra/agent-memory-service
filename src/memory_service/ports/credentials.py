"""Protected model credentials; plaintext never belongs to a memory or job.

A key is stored per ``(tenant_id, principal_id)``. Since ADR 0023 the principal may also be
a workspace (``workspace:<id>``) or the tenant itself (``tenant``): a call resolves the most
specific row that exists, so a team registers one Bifrost key and every agent of the
workspace calls through it unless it has a key of its own.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Protocol

from pydantic import SecretStr

WORKSPACE_PRINCIPAL_PREFIX: Final = "workspace:"
TENANT_PRINCIPAL: Final = "tenant"


@dataclass(frozen=True)
class ModelIdentity:
    tenant_id: str
    principal_id: str
    workspace_id: str | None = None

    def levels(self) -> tuple["ModelIdentity", ...]:
        """The rows a resolution reads, most specific first: the principal's own, then the
        workspace's (when the call runs in one), then the tenant's."""
        rows = [ModelIdentity(self.tenant_id, self.principal_id)]
        if self.workspace_id:
            rows.append(workspace_identity(self.tenant_id, self.workspace_id))
        rows.append(tenant_identity(self.tenant_id))
        return tuple(dict.fromkeys(rows))


def workspace_identity(tenant_id: str, workspace_id: str) -> ModelIdentity:
    return ModelIdentity(tenant_id, f"{WORKSPACE_PRINCIPAL_PREFIX}{workspace_id}")


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
    key: SecretStr = field(repr=False)
    revision: int
    #: The row the key came from: the principal's own, the workspace's or the tenant's.
    identity: ModelIdentity


class CredentialRepository(Protocol):
    async def get(self, identity: ModelIdentity) -> StoredCredential | None: ...

    async def put(
        self, identity: ModelIdentity, *, key_id: str, ciphertext: bytes | None
    ) -> StoredCredential: ...


class CredentialCipher(Protocol):
    def encrypt(self, identity: ModelIdentity, value: SecretStr) -> tuple[str, bytes]: ...

    def decrypt(self, record: StoredCredential) -> SecretStr: ...


class CredentialResolver(Protocol):
    async def resolve(self, identity: ModelIdentity) -> ResolvedCredential | None: ...

    async def confirm(self, identity: ModelIdentity, resolved: ResolvedCredential | None) -> None:
        """Refuse a result obtained under a credential that has since changed."""
        ...
