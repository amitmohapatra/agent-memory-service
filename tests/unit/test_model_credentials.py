"""Model keys resolve agent -> workspace -> tenant, and a call fails closed on any change."""

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
    workspace_identity,
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


def test_the_levels_walk_from_the_principal_to_the_tenant_without_repeats() -> None:
    identity = agent_identity(CTX)
    assert identity == ModelIdentity("acme", "agent:alice/ref", "fin")
    assert identity.levels() == (
        ModelIdentity("acme", "agent:alice/ref"),
        ModelIdentity("acme", "workspace:fin"),
        ModelIdentity("acme", "tenant"),
    )
    assert ModelIdentity("acme", "user:bob").levels() == (
        ModelIdentity("acme", "user:bob"),
        ModelIdentity("acme", "tenant"),
    )
    assert workspace_identity("acme", "fin").levels() == (
        ModelIdentity("acme", "workspace:fin"),
        ModelIdentity("acme", "tenant"),
    )
    assert tenant_identity("acme").levels() == (ModelIdentity("acme", "tenant"),)
    assert identity_of(CTX) == identity
    assert current_model_identity() is None, "nothing is bound outside a request or job"
    with pytest.raises(ValidationFailed):
        agent_identity(CTX.model_copy(update={"agent_id": None}))


async def test_the_most_specific_key_wins_and_a_tombstone_never_borrows_one() -> None:
    service, _ = _service()
    identity = agent_identity(CTX)
    assert await service.resolve(identity) is None
    await _set(service, tenant_identity("acme"), "vk-tenant")
    tenant = await service.resolve(identity)
    assert tenant is not None and tenant.key.get_secret_value() == "vk-tenant"
    assert tenant.identity == tenant_identity("acme") and tenant.revision == 1
    await _set(service, workspace_identity("acme", "fin"), "vk-team")
    team = await service.resolve(identity)
    assert team is not None and team.key.get_secret_value() == "vk-team"
    assert team.identity == workspace_identity("acme", "fin")
    # another workspace's agent still gets the tenant key
    other = await service.resolve(ModelIdentity("acme", "agent:bob/ref", "ops"))
    assert other is not None and other.identity == tenant_identity("acme")
    await _set(service, ModelIdentity("acme", "agent:alice/ref"), "vk-own")
    own = await service.resolve(identity)
    assert own is not None and own.key.get_secret_value() == "vk-own"
    # revoking the agent's key refuses: it must not fall through to the team's
    await _set(service, ModelIdentity("acme", "agent:alice/ref"), None)
    with pytest.raises(ProviderNotConfigured, match="revoked"):
        await service.resolve(identity)
    # a revoked workspace key blocks every agent of the workspace the same way
    await _set(service, workspace_identity("acme", "fin"), None)
    with pytest.raises(ProviderNotConfigured, match="revoked"):
        await service.resolve(ModelIdentity("acme", "agent:carol/ref", "fin"))
    # and a foreign tenant sees nothing
    assert await service.resolve(ModelIdentity("rival", "agent:alice/ref", "fin")) is None


async def test_confirm_requires_the_same_row_and_revision_the_call_ran_under() -> None:
    service, _ = _service()
    identity = agent_identity(CTX)
    await service.confirm(identity, None)  # operator fallback while nothing exists
    await _set(service, workspace_identity("acme", "fin"), "vk-team")
    with pytest.raises(ProviderNotConfigured, match="changed"):
        await service.confirm(identity, None)  # a key appeared mid-call
    resolved = await service.resolve(identity)
    await service.confirm(identity, resolved)
    await _set(service, workspace_identity("acme", "fin"), "vk-rotated")
    with pytest.raises(ProviderNotConfigured, match="changed"):
        await service.confirm(identity, resolved)  # rotated mid-call
    rotated = await service.resolve(identity)
    await _set(service, ModelIdentity("acme", "agent:alice/ref"), "vk-own")
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
    assert await service.metadata_for(workspace_identity("acme", "fin")) is None
    assert await service.metadata(CTX) is None  # the agent's own row, not the fallback


def test_the_discovery_cache_is_keyed_by_row_and_revision_never_by_key_material() -> None:
    identity = agent_identity(CTX)
    assert _catalog_key(None) is None
    assert _catalog_key((identity, None)) == (identity, None)
    resolved = ResolvedCredential(SecretStr("vk"), 3, workspace_identity("acme", "fin"))
    assert _catalog_key((identity, resolved)) == (workspace_identity("acme", "fin"), 3)
