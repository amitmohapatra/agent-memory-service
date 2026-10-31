"""Registered model endpoints at agent and tenant level: rotated and revoked atomically,
resolved at call time from the most specific row that exists.

The whole endpoint is one envelope (``trellis.contracts.Endpoint`` as JSON, its key in the
clear inside it) encrypted with the credential cipher: a database dump shows neither the key
nor where the agent's model is.
"""

import json

from pydantic import SecretStr, ValidationError

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ProviderNotConfigured, ValidationFailed
from memory_service.domain.revisions import RevisionKind
from memory_service.ports.credentials import (
    AGENT_PRINCIPAL_PREFIX,
    CredentialCipher,
    ModelIdentity,
    ResolvedCredential,
    StoredCredential,
)
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory
from trellis.contracts import Endpoint

KEY_MAX_CHARS = 4096


def agent_identity(ctx: MemoryExecutionContext) -> ModelIdentity:
    """The agent's own level: ``agent:<agent_id>``, the same whichever user the request
    names, so the key registered at startup is the key every run of the agent resolves."""
    if ctx.agent_id is None:
        raise ValidationFailed("An agent_id is required for model credentials")
    return ModelIdentity(ctx.tenant_id, f"{AGENT_PRINCIPAL_PREFIX}{ctx.agent_id}")


def _validated(endpoint: Endpoint) -> None:
    if endpoint.api_key is None:
        return
    value = endpoint.api_key.get_secret_value()
    if not 1 <= len(value) <= KEY_MAX_CHARS or any(not 33 <= ord(c) <= 126 for c in value):
        raise ValidationFailed(
            f"An endpoint's api_key must contain 1-{KEY_MAX_CHARS} visible ASCII characters"
        )


def sealed(endpoint: Endpoint) -> SecretStr:
    """The endpoint as the plaintext the cipher seals: its JSON, the key unmasked."""
    data = endpoint.model_dump(mode="json")
    data["api_key"] = endpoint.api_key.get_secret_value() if endpoint.api_key else None
    return SecretStr(json.dumps(data, sort_keys=True, separators=(",", ":")))


def unsealed(plain: SecretStr) -> Endpoint:
    try:
        return Endpoint.model_validate_json(plain.get_secret_value())
    except ValidationError as exc:
        raise ProviderNotConfigured(
            "The stored model endpoint is unreadable: register it again"
        ) from exc


class ModelCredentials:
    def __init__(self, uow_factory: UnitOfWorkFactory, cipher: CredentialCipher) -> None:
        self.uow_factory = uow_factory
        self.cipher = cipher

    async def set(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, endpoint: Endpoint | None
    ) -> StoredCredential:
        """Register, rotate (an endpoint) or revoke (None) the acting agent's own."""
        return await self.set_for(uow, agent_identity(ctx), endpoint)

    async def set_for(
        self, uow: UnitOfWork, identity: ModelIdentity, endpoint: Endpoint | None
    ) -> StoredCredential:
        """The same for any level: the row is the identity's own, never a fallback's."""
        own = identity
        if endpoint is not None:
            _validated(endpoint)
        key_id, encrypted = (
            self.cipher.encrypt(own, sealed(endpoint)) if endpoint is not None else ("", None)
        )
        result = await uow.credentials.put(own, key_id=key_id, ciphertext=encrypted)
        # Read caches contain model-assisted output, but never a credential. Rotation and
        # revocation must invalidate that output, including explicit bundle replay.
        await uow.revisions.bump(identity.tenant_id, RevisionKind.TENANT, "")
        return result

    async def metadata(self, ctx: MemoryExecutionContext) -> StoredCredential | None:
        return await self.metadata_for(agent_identity(ctx))

    async def metadata_for(self, identity: ModelIdentity) -> StoredCredential | None:
        async with self.uow_factory() as uow:
            return await uow.credentials.get(identity)

    def endpoint_of(self, record: StoredCredential) -> Endpoint | None:
        """A row's endpoint (None for a revocation)."""
        if record.ciphertext is None:
            return None
        return unsealed(self.cipher.decrypt(record))

    async def resolve(self, identity: ModelIdentity) -> ResolvedCredential | None:
        """The endpoint of the most specific level that has a row. A revocation tombstone at
        that level refuses: a revoked agent never silently borrows the tenant's or the
        service's endpoint."""
        async with self.uow_factory() as uow:
            record = await uow.credentials.first(identity.levels())
        if record is None:
            return None
        if record.ciphertext is None:
            raise ProviderNotConfigured(
                "Model credential is revoked", details={"level": record.identity.principal_id}
            )
        return ResolvedCredential(
            unsealed(self.cipher.decrypt(record)), record.revision, record.identity
        )

    async def confirm(self, identity: ModelIdentity, resolved: ResolvedCredential | None) -> None:
        """The call ran under ``resolved``; refuse its result unless that is still the answer.
        With ``None`` the operator fallback stays authorised only while no row exists at any
        level, so a key registered mid-call fails the call closed rather than mixing keys."""
        current = await self.resolve(identity)
        expected = (resolved.identity, resolved.revision) if resolved is not None else None
        actual = (current.identity, current.revision) if current is not None else None
        if expected != actual:
            raise ProviderNotConfigured("Model credential changed during the request")
