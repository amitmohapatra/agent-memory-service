"""Protected agent model credentials; plaintext never belongs to a memory or job."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from pydantic import SecretStr


@dataclass(frozen=True)
class ModelIdentity:
    tenant_id: str
    principal_id: str


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

    async def confirm(self, identity: ModelIdentity, revision: int | None) -> None: ...
