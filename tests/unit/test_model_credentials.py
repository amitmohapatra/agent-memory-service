"""Model keys resolve agent -> tenant, and a call fails closed on any change."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from memory_service.adapters.models.llm import _catalog_key
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ProviderNotConfigured, ValidationFailed
from memory_service.modules.llm.credentials import ModelCredentials, agent_identity
from memory_service.modules.llm.policy import current_model_identity, identity_of
from memory_service.ports.credentials import (
    ModelIdentity,
    ResolvedCredential,
    StoredCredential,
    tenant_identity,
)

CTX = MemoryExecutionContext(tenant_id="acme", workspace_id="fin", user_id="alice", agent_id="ref")


class _Repo:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], StoredCredential] = {}

    async def get(self, identity: ModelIdentity) -> StoredCredential | None:
        return self.rows.get((identity.tenant_id, identity.principal_id))

    async def first(self, levels) -> StoredCredential | None:
        return next(
            (r for level in levels if (r := self.rows.get((level.tenant_id, level.principal_id)))),
            None,
        )

    async def put(self, identity: ModelIdentity, *, key_id: str, ciphertext: bytes | None):
        key = (identity.tenant_id, identity.principal_id)
        revision = self.rows[key].revision + 1 if key in self.rows else 1
        record = StoredCredential(
            ModelIdentity(identity.tenant_id, identity.principal_id),
            key_id,
            ciphertext,
            revision,
            datetime.now(UTC),
        )
        self.rows[key] = record
        return record


class _Revisions:
    async def bump(self, *args, **kwargs) -> None:
        return None


class _Uow:
    def __init__(self, repo: _Repo) -> None:
        self.credentials = repo
        self.revisions = _Revisions()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def commit(self) -> None:
        return None


class _Cipher:
    def encrypt(self, identity: ModelIdentity, value: SecretStr) -> tuple[str, bytes]:
        return "k1", f"{identity.principal_id}\x1f{value.get_secret_value()}".encode()

    def decrypt(self, record: StoredCredential) -> SecretStr:
        assert record.ciphertext is not None
        return SecretStr(record.ciphertext.decode().split("\x1f", 1)[1])


def _service() -> tuple[ModelCredentials, _Repo]:
    repo = _Repo()
    return ModelCredentials(lambda: _Uow(repo), _Cipher()), repo


async def _set(service: ModelCredentials, identity: ModelIdentity, key: str | None):
    async with service.uow_factory() as uow:
        return await service.set_for(uow, identity, SecretStr(key) if key else None)


def test_the_levels_are_the_agent_then_the_tenant() -> None:
    identity = agent_identity(CTX)
    assert identity == ModelIdentity("acme", "agent:ref")
    assert identity.levels() == (ModelIdentity("acme", "agent:ref"), tenant_identity("acme"))
    # the agent acting for a user resolves through the same agent level
    assert identity_of(CTX) == ModelIdentity("acme", "agent:alice/ref")
    assert identity_of(CTX).levels() == identity.levels()
    assert ModelIdentity("acme", "user:bob").levels() == (tenant_identity("acme"),)
    assert tenant_identity("acme").levels() == (tenant_identity("acme"),)
    assert current_model_identity() is None, "nothing is bound outside a request or job"
    with pytest.raises(ValidationFailed):
        agent_identity(CTX.model_copy(update={"agent_id": None}))


def test_the_agent_key_is_the_same_whichever_user_the_request_names() -> None:
    """The harness registers the agent's key at startup, with no user; its runs then read
    and use it with a user in the headers. Keyed by the user-bound principal, the status
    read with a user header looked at a different row than the PUT wrote."""
    unattended = CTX.model_copy(update={"user_id": None})
    assert agent_identity(CTX) == agent_identity(unattended) == ModelIdentity("acme", "agent:ref")


async def test_the_most_specific_key_wins_and_a_tombstone_never_borrows_one() -> None:
    service, _ = _service()
    identity = identity_of(CTX)
    assert await service.resolve(identity) is None
    await _set(service, tenant_identity("acme"), "vk-tenant")
    tenant = await service.resolve(identity)
    assert tenant is not None and tenant.key.get_secret_value() == "vk-tenant"
    assert tenant.identity == tenant_identity("acme") and tenant.revision == 1
    # another agent still gets the tenant key
    other = await service.resolve(ModelIdentity("acme", "agent:bob/other"))
    assert other is not None and other.identity == tenant_identity("acme")
    await _set(service, agent_identity(CTX), "vk-own")
    own = await service.resolve(identity)
    assert own is not None and own.key.get_secret_value() == "vk-own"
    assert own.identity == ModelIdentity("acme", "agent:ref")
    # the same agent acting for another user resolves the same key
    carol = await service.resolve(ModelIdentity("acme", "agent:carol/ref"))
    assert carol is not None and carol.key.get_secret_value() == "vk-own"
    # revoking the agent's key refuses: it must not fall through to the tenant's
    await _set(service, agent_identity(CTX), None)
    with pytest.raises(ProviderNotConfigured, match="revoked"):
        await service.resolve(identity)
    # and a foreign tenant sees nothing
    assert await service.resolve(ModelIdentity("rival", "agent:alice/ref")) is None


async def test_confirm_requires_the_same_row_and_revision_the_call_ran_under() -> None:
    service, _ = _service()
    identity = identity_of(CTX)
    await service.confirm(identity, None)  # operator fallback while nothing exists
    await _set(service, tenant_identity("acme"), "vk-tenant")
    with pytest.raises(ProviderNotConfigured, match="changed"):
        await service.confirm(identity, None)  # a key appeared mid-call
    resolved = await service.resolve(identity)
    await service.confirm(identity, resolved)
    await _set(service, tenant_identity("acme"), "vk-rotated")
    with pytest.raises(ProviderNotConfigured, match="changed"):
        await service.confirm(identity, resolved)  # rotated mid-call
    rotated = await service.resolve(identity)
    await _set(service, agent_identity(CTX), "vk-own")
    with pytest.raises(ProviderNotConfigured, match="changed"):
        await service.confirm(identity, rotated)  # a more specific key appeared mid-call


async def test_keys_are_validated_and_metadata_is_per_level() -> None:
    service, _ = _service()
    with pytest.raises(ValidationFailed):
        await _set(service, tenant_identity("acme"), "has space")
    with pytest.raises(ValidationFailed):
        await _set(service, tenant_identity("acme"), "x" * 5000)
    await _set(service, tenant_identity("acme"), "vk-tenant")
    assert await service.metadata_for(tenant_identity("acme")) is not None
    assert await service.metadata(CTX) is None  # the agent's own row, not the fallback
    await _set(service, agent_identity(CTX), "vk-own")
    unattended = CTX.model_copy(update={"user_id": None})
    assert (await service.metadata(CTX)) == (await service.metadata(unattended))


def test_the_discovery_cache_is_keyed_by_row_and_revision_never_by_key_material() -> None:
    identity = identity_of(CTX)
    assert _catalog_key(None) is None
    assert _catalog_key((identity, None)) == (identity, None)
    resolved = ResolvedCredential(SecretStr("vk"), 3, tenant_identity("acme"))
    assert _catalog_key((identity, resolved)) == (tenant_identity("acme"), 3)
