"""Model credentials at agent, workspace and tenant level: rotated and revoked atomically,
resolved at call time from the most specific row that exists (ADR 0023)."""

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

KEY_MAX_CHARS = 4096


def agent_identity(ctx: MemoryExecutionContext) -> ModelIdentity:
    if ctx.agent_id is None:
        raise ValidationFailed("An agent_id is required for model credentials")
    return ModelIdentity(ctx.tenant_id, ctx.principal_id, ctx.workspace_id)


def _validated(key: SecretStr) -> None:
    value = key.get_secret_value()
    if not 1 <= len(value) <= KEY_MAX_CHARS or any(not 33 <= ord(c) <= 126 for c in value):
        raise ValidationFailed(
            f"A virtual key must contain 1-{KEY_MAX_CHARS} visible ASCII characters"
        )


class ModelCredentials:
    def __init__(self, uow_factory: UnitOfWorkFactory, cipher: CredentialCipher) -> None:
        self.uow_factory = uow_factory
        self.cipher = cipher

    async def set(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, key: SecretStr | None
    ) -> StoredCredential:
        """Register, rotate (a key) or revoke (None) the acting agent's own key."""
        return await self.set_for(uow, agent_identity(ctx), key)

    async def set_for(
        self, uow: UnitOfWork, identity: ModelIdentity, key: SecretStr | None
    ) -> StoredCredential:
        """The same for any level: the row is the identity's own, never a fallback's."""
        own = ModelIdentity(identity.tenant_id, identity.principal_id)
        if key is not None:
            _validated(key)
        key_id, encrypted = self.cipher.encrypt(own, key) if key is not None else ("", None)
        result = await uow.credentials.put(own, key_id=key_id, ciphertext=encrypted)
        # Read caches contain model-assisted output, but never a credential. Rotation and
        # revocation must invalidate that output, including explicit bundle replay.
        await uow.revisions.bump(identity.tenant_id, RevisionKind.TENANT, "")
        return result

    async def metadata(self, ctx: MemoryExecutionContext) -> StoredCredential | None:
        return await self.metadata_for(agent_identity(ctx))

    async def metadata_for(self, identity: ModelIdentity) -> StoredCredential | None:
        async with self.uow_factory() as uow:
            return await uow.credentials.get(
                ModelIdentity(identity.tenant_id, identity.principal_id)
            )

    async def resolve(self, identity: ModelIdentity) -> ResolvedCredential | None:
        """The key of the most specific level that has a row. A revocation tombstone at that
        level refuses: a revoked agent never silently borrows the team's or operator's key."""
        async with self.uow_factory() as uow:
            record = await uow.credentials.first(identity.levels())
        if record is None:
            return None
        if record.ciphertext is None:
            raise ProviderNotConfigured(
                "Model credential is revoked", details={"level": record.identity.principal_id}
            )
        return ResolvedCredential(self.cipher.decrypt(record), record.revision, record.identity)

    async def confirm(self, identity: ModelIdentity, resolved: ResolvedCredential | None) -> None:
        """The call ran under ``resolved``; refuse its result unless that is still the answer.
        With ``None`` the operator fallback stays authorised only while no row exists at any
        level, so a key registered mid-call fails the call closed rather than mixing keys."""
        current = await self.resolve(identity)
        expected = (resolved.identity, resolved.revision) if resolved is not None else None
        actual = (current.identity, current.revision) if current is not None else None
        if expected != actual:
            raise ProviderNotConfigured("Model credential changed during the request")
