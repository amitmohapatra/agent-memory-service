"""The tenant's model policy, and the usage ledger.

``access`` answers the one question a request or job asks before any model call: which uses
may run under this identity, whether reads are assisted, which model each use calls, and
whether a registered key (the agent's or the tenant's) can pay. One unit of work, two indexed
reads.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime

from memory_service.domain.revisions import RevisionKind
from memory_service.modules.llm.policy import DEFAULT_ACCESS, ModelAccess
from memory_service.ports.credentials import ModelIdentity
from memory_service.ports.llm import StoredPolicy, UsageDay
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory


class ModelPolicies:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self.uow_factory = uow_factory

    async def access(self, identity: ModelIdentity) -> ModelAccess:
        async with self.uow_factory() as uow:
            key = await uow.credentials.first(identity.levels())
            policy = await uow.llm_policies.get(identity.tenant_id)
        has_key = key is not None and key.ciphertext is not None
        if policy is None:
            return ModelAccess(DEFAULT_ACCESS.uses, DEFAULT_ACCESS.read_assist, has_key)
        return ModelAccess(policy.uses, policy.read_assist, has_key, policy.models)

    async def tenants_with_keys(self) -> list[str]:
        async with self.uow_factory() as uow:
            return await uow.credentials.tenants_with_keys()

    async def get(self, tenant_id: str) -> StoredPolicy | None:
        async with self.uow_factory() as uow:
            return await uow.llm_policies.get(tenant_id)

    async def set(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        *,
        uses: Sequence[str],
        read_assist: bool,
        models: Mapping[str, str],
    ) -> StoredPolicy:
        stored = await uow.llm_policies.put(
            tenant_id, uses=uses, read_assist=read_assist, models=models
        )
        # Cached bundles carry model-assisted output; a policy change must invalidate them.
        await uow.revisions.bump(tenant_id, RevisionKind.TENANT, "")
        return stored


class LLMUsage:
    """Daily token and call counts per tenant and use; the gateway adapter's recorder."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self.uow_factory = uow_factory

    async def record(self, tenant_id: str, use: str, tokens: int) -> None:
        async with self.uow_factory() as uow:
            await uow.llm_usage.add(tenant_id, use, datetime.now(UTC).date(), tokens)
            await uow.commit()

    async def between(self, tenant_id: str, since: date, until: date) -> list[UsageDay]:
        async with self.uow_factory() as uow:
            return await uow.llm_usage.between(tenant_id, since, until)
