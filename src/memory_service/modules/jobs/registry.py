"""Background task registration. Each module contributes handlers; the worker and the
API process both register them so inline/test queues can execute jobs in-process."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from memory_service.observability.logging import get_logger
from memory_service.ports.tasks import Queue

if TYPE_CHECKING:
    from memory_service.application.container import Container

log = get_logger(__name__)

TASK_PROCESS_OBSERVATION = "memory.process_observation"
TASK_ARCHIVE_STAGE = "archive.stage_message"
TASK_OUTBOX_SWEEP = "system.outbox_sweep"
TASK_IDEMPOTENCY_PURGE = "system.idempotency_purge"
TASK_RECONCILE = "system.reconcile"
TASK_ARCHIVE_PURGE = "archive.purge_payloads"
TASK_MEMORY_INDEX = "memory.index"
TASK_MEMORY_EXPIRE = "memory.expire"
TASK_MEMORY_FORGET = "memory.forget"
TASK_MEMORY_REFLECT = "memory.reflect"


def register_handlers(container: Container) -> None:
    queue = container.tasks
    if queue is None:
        return
    uow_factory = container.services["uow_factory"]

    async def process_observation(payload: dict[str, Any]) -> None:
        """M3 baseline: mark the observation processed. M7 replaces this with the memory
        intelligence pipeline (extract -> classify -> dedup -> consolidate -> index)."""
        pipeline = container.services.get("observation_pipeline")
        if pipeline is not None:
            await pipeline.run(payload)
            return
        async with uow_factory() as uow:
            await uow.observations.mark_processed(
                payload["tenant_id"], payload["observation_id"], status="RECORDED"
            )
            await uow.commit()

    async def archive_stage(payload: dict[str, Any]) -> None:
        """M4 replaces this with segment compaction + blob upload + verification."""
        archiver = container.services.get("archive_service")
        if archiver is not None:
            await archiver.archive_thread(payload["tenant_id"], payload["thread_id"])

    async def document_parse(payload: dict[str, Any]) -> None:
        ingestion = container.services.get("ingestion")
        if ingestion is not None:
            await ingestion.parse_document(payload["tenant_id"], payload["document_id"])

    async def document_index(payload: dict[str, Any]) -> None:
        """M6 attaches the indexer; until then the job is a no-op that keeps the chain intact."""
        indexer = container.services.get("indexer")
        if indexer is not None:
            await indexer.index_document(payload["tenant_id"], payload["document_id"])
        graph = container.services.get("graph")
        if graph is not None:
            await graph.enrich_document(payload["tenant_id"], payload["document_id"])

    async def memory_index(payload: dict[str, Any]) -> None:
        indexer = container.services.get("indexer")
        if indexer is not None:
            await indexer.index_memories(payload["tenant_id"], list(payload["memory_ids"]))
        graph = container.services.get("graph")
        if graph is not None:
            await graph.enrich_memories(payload["tenant_id"], list(payload["memory_ids"]))

    async def memory_expire(payload: dict[str, Any]) -> None:
        """Mark SHORT_TERM memories past their TTL as EXPIRED and drop them from the index."""
        async with uow_factory() as uow:
            expired = await uow.memories.expire_due(now=datetime.now(UTC))
            await uow.commit()
        indexer = container.services.get("indexer")
        by_tenant: dict[str, list[str]] = {}
        for tenant_id, memory_id in expired:
            by_tenant.setdefault(tenant_id, []).append(memory_id)
        for tenant_id, ids in by_tenant.items():
            if indexer is not None:
                await indexer.index_memories(tenant_id, ids)
        if expired:
            log.info("memory.expired", count=len(expired))

    async def memory_forget(payload: dict[str, Any]) -> None:
        """Archive idle, low-value memories and evict working memory by the same score.

        ``sweep`` enqueues its own re-index jobs inside the transaction that archives the rows
        and evicts working memory itself, so there is nothing to do here but run it. Canonical
        rows are never deleted; ``ForgettingService.restore`` brings an archived memory back.
        """
        forgetting = container.services.get("forgetting")
        if forgetting is not None:
            await forgetting.sweep()

    async def memory_reflect(payload: dict[str, Any]) -> None:
        """Derive insights over each principal's recent memories (LLM use ``reflection``)."""
        reflection = container.services.get("reflection")
        if reflection is not None:
            await reflection.reflect_all()

    async def outbox_sweep(payload: dict[str, Any]) -> None:
        relay = container.services.get("outbox_relay")
        if relay is not None:
            n = await relay.sweep(older_than_seconds=int(payload.get("older_than_seconds", 30)))
            if n:
                log.info("outbox.swept", dispatched=n)

    async def idempotency_purge(payload: dict[str, Any]) -> None:
        async with uow_factory() as uow:
            n = await uow.idempotency.purge_expired(now=datetime.now(UTC))
            await uow.commit()
        if n:
            log.info("idempotency.purged", count=n)

    async def reconcile(payload: dict[str, Any]) -> None:
        relay = container.services.get("outbox_relay")
        if relay is not None:
            await relay.sweep(older_than_seconds=30)
        archiver = container.services.get("archive_service")
        if archiver is not None:
            report = await archiver.reconcile()
            if any(report.values()):
                log.info("reconcile.report", **report)
        recover = getattr(container.tasks, "recover_stalled", None)
        if recover is not None:  # jobs orphaned by a worker that died mid-run
            await recover(seconds_since_heartbeat=container.settings.tasks.stalled_after_seconds)
        for extra in container.services.get("extra_reconcilers", []):
            await extra()

    async def archive_purge(payload: dict[str, Any]) -> None:
        archiver = container.services.get("archive_service")
        if archiver is not None:
            await archiver.purge_staged_payloads()

    queue.register(TASK_PROCESS_OBSERVATION, Queue.CHAT_FAST, process_observation, retries=5)
    queue.register("document.parse", Queue.DOCUMENT_PARSE, document_parse, retries=3)
    queue.register("document.index", Queue.EMBEDDING, document_index, retries=5)
    queue.register(TASK_MEMORY_INDEX, Queue.EMBEDDING, memory_index, retries=5)
    queue.register(TASK_MEMORY_EXPIRE, Queue.RECONCILE, memory_expire, retries=0)
    queue.register(TASK_MEMORY_FORGET, Queue.RECONCILE, memory_forget, retries=0)
    queue.register(TASK_RECONCILE, Queue.RECONCILE, reconcile, retries=0)
    queue.register(TASK_ARCHIVE_PURGE, Queue.ARCHIVE, archive_purge, retries=0)
    every = max(1, container.settings.tasks.periodic_reconcile_seconds // 60)
    queue.register_periodic(
        "periodic.reconcile", Queue.RECONCILE, reconcile, cron=f"*/{min(every, 59)} * * * *"
    )
    queue.register_periodic(
        "periodic.archive_purge", Queue.ARCHIVE, archive_purge, cron="17 * * * *"
    )
    queue.register_periodic(
        "periodic.idempotency_purge", Queue.RECONCILE, idempotency_purge, cron="43 * * * *"
    )
    queue.register_periodic(
        "periodic.memory_expire", Queue.RECONCILE, memory_expire, cron="29 * * * *"
    )
    queue.register_periodic(
        "periodic.memory_forget", Queue.RECONCILE, memory_forget, cron="11 4 * * *"
    )
    if container.settings.models.llm.wants("reflection"):
        queue.register(TASK_MEMORY_REFLECT, Queue.RECONCILE, memory_reflect, retries=0)
        queue.register_periodic(
            "periodic.memory_reflect", Queue.RECONCILE, memory_reflect, cron="53 */6 * * *"
        )
    queue.register(TASK_ARCHIVE_STAGE, Queue.ARCHIVE, archive_stage, retries=10)
    queue.register(TASK_OUTBOX_SWEEP, Queue.RECONCILE, outbox_sweep, retries=0)
    queue.register(TASK_IDEMPOTENCY_PURGE, Queue.RECONCILE, idempotency_purge, retries=0)
    for extra in container.services.get("extra_task_registrars", []):
        extra(container)
