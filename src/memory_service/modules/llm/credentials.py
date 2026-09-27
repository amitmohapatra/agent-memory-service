"""Agent-owned credentials, atomically rotated/revoked and resolved at call time."""

from pydantic import SecretStr

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ProviderNotConfigured, ValidationFailed
from memory_service.domain.revisions import RevisionKind
from memory_service.ports.credentials import (
    CredentialCipher,
    ModelIdentity,
    ResolvedCredential,
    StoredCredential,
)
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory


def agent_identity(ctx: MemoryExecutionContext) -> ModelIdentity:
    if ctx.agent_id is None:
        raise ValidationFailed("An agent_id is required for model credentials")
    return ModelIdentity(ctx.tenant_id, ctx.principal_id)


class AgentCredentials:
    def __init__(self, uow_factory: UnitOfWorkFactory, cipher: CredentialCipher) -> None:
        self.uow_factory = uow_factory
        self.cipher = cipher

    async def set(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, key: SecretStr | None
    ) -> StoredCredential:
        identity = agent_identity(ctx)
        if key is not None:
            value = key.get_secret_value()
            if not 1 <= len(value) <= 4096 or any(not 33 <= ord(c) <= 126 for c in value):
                raise ValidationFailed("A virtual key must contain 1-4096 visible ASCII characters")
        key_id, encrypted = self.cipher.encrypt(identity, key) if key is not None else ("", None)
        result = await uow.credentials.put(identity, key_id=key_id, ciphertext=encrypted)
        # Read caches contain model-assisted output, but never a credential. Rotation and
        # revocation must invalidate that output, including explicit bundle replay.
        await uow.revisions.bump(ctx.tenant_id, RevisionKind.TENANT, "")
        return result

    async def metadata(self, ctx: MemoryExecutionContext) -> StoredCredential | None:
        async with self.uow_factory() as uow:
            return await uow.credentials.get(agent_identity(ctx))

    async def resolve(self, identity: ModelIdentity) -> ResolvedCredential | None:
        async with self.uow_factory() as uow:
            record = await uow.credentials.get(identity)
        if record is None:
            return None
        if record.ciphertext is None:
            # A tombstone must never silently select the operator's key instead.
            raise ProviderNotConfigured("Agent model credential is revoked")
        return ResolvedCredential(self.cipher.decrypt(record), record.revision)

    async def confirm(self, identity: ModelIdentity, revision: int | None) -> None:
        async with self.uow_factory() as uow:
            record = await uow.credentials.get(identity)
        if revision is None and record is None:
            return  # operator fallback remains authorized only while no agent policy exists
        if record is None or record.ciphertext is None or record.revision != revision:
            raise ProviderNotConfigured("Agent model credential changed during the request")
