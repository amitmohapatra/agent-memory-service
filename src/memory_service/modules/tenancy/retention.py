"""Per-tenant retention: canonical memories older than the tenant's policy are forgotten.

Forgetting here is the same soft delete ``DELETE /v1/memories/{id}`` performs - the row is
retracted, its projections are removed by the index job, and every reader's revision moves -
so a retention sweep is verifiable the way a manual forget is. Age is the record's creation
time: retention is a data-minimisation promise about how long something is kept, not about
how recently it was useful. Each run drains a tenant in bounded batches up to a per-run cap,
so a backlog converges over a few runs without one tenant holding a transaction for long,
and one tenant's failure never skips the next. Conversation rows, documents and
observations are not covered yet; that is stated in the plan, not hidden in this docstring.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memory_service.modules.memory.pipeline import TASK_MEMORY_INDEX
from memory_service.modules.memory.revisions import bump_memory_revisions
from memory_service.observability.logging import get_logger
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)


class RetentionService:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, *, batch: int = 500, max_batches: int = 20
    ) -> None:
        self.uow_factory = uow_factory
        self.batch = batch
        self.max_batches = max_batches

    async def sweep(self, *, now: datetime | None = None) -> int:
        now = now or datetime.now(UTC)
        async with self.uow_factory() as uow:
            policies = await uow.tenants.retention_policies()
        forgotten = 0
        for tenant_id, days in sorted(policies.items()):
            try:
                forgotten += await self._sweep_tenant(tenant_id, now - timedelta(days=days))
            except Exception as exc:  # the next tenant still gets its sweep
                log.warning("retention.tenant_failed", tenant_id=tenant_id, error=str(exc))
        if forgotten:
            log.info("retention.swept", forgotten=forgotten)
        return forgotten

    async def _sweep_tenant(self, tenant_id: str, cutoff: datetime) -> int:
        forgotten = 0
        for _ in range(self.max_batches):
            n = await self._sweep_batch(tenant_id, cutoff)
            forgotten += n
            if n < self.batch:
                return forgotten
        log.info("retention.backlog", tenant_id=tenant_id, forgotten=forgotten)
        return forgotten

    async def _sweep_batch(self, tenant_id: str, cutoff: datetime) -> int:
        async with self.uow_factory() as uow:
            memories = await uow.memories.list_older_than(
                tenant_id, before=cutoff, limit=self.batch
            )
            if not memories:
                return 0
            for memory in memories:
                await uow.memories.forget(tenant_id, memory.memory_id)
            await bump_memory_revisions(uow, memories)
            memory_ids = [m.memory_id for m in memories]
            await uow.enqueue(
                JobSpec(
                    task_name=TASK_MEMORY_INDEX,
                    queue=Queue.EMBEDDING,
                    payload={"tenant_id": tenant_id, "memory_ids": memory_ids},
                    tenant_id=tenant_id,
                    idempotency_key=f"memidx:retention:{tenant_id}:{memory_ids[0]}:{memory_ids[-1]}",
                )
            )
            await uow.commit()
        return len(memories)
