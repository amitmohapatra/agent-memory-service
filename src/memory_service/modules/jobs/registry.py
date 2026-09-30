"""Background task registration. Each module contributes handlers; the worker and the
API process both register them so inline/test queues can execute jobs in-process."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from memory_service.config.constants import TASKS
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.conversation.summary import TASK_SUMMARY_REFRESH
from memory_service.modules.feedback.service import TASK_FEEDBACK_PROJECT
from memory_service.modules.jobs.names import TASK_MEMORY_INDEX
from memory_service.modules.llm.cost import llm_accounting
from memory_service.modules.memory.connections import TASK_MEMORY_CONNECT
from memory_service.modules.memory.revisions import bump_memory_revisions
from memory_service.modules.profile.service import (
    PROFILE_MEMORY_TYPES,
    TASK_PROFILE_QUERY,
    TASK_PROFILE_REFRESH,
    ProfileService,
)
from memory_service.modules.tools.service import TASK_TOOLS_INDEX, TASK_TOOLS_LEARN
from memory_service.observability.logging import get_logger
from memory_service.ports.tasks import JobSpec, Queue, TaskHandler

if TYPE_CHECKING:
    from memory_service.application.container import Container

log = get_logger(__name__)

TASK_PROCESS_OBSERVATION = "memory.process_observation"
TASK_ARCHIVE_STAGE = "archive.stage_message"
TASK_OUTBOX_SWEEP = "system.outbox_sweep"
TASK_IDEMPOTENCY_PURGE = "system.idempotency_purge"
TASK_RECONCILE = "system.reconcile"
TASK_ARCHIVE_PURGE = "archive.purge_payloads"
TASK_MEMORY_EXPIRE = "memory.expire"
TASK_MEMORY_FORGET = "memory.forget"
TASK_MEMORY_REFLECT = "memory.reflect"


class _AccountedQueue:
    """Registers every handler inside its own LLM accounting scope, so a job's model tokens
    are counted and logged per job (and never leak into the request that ran it inline)."""

    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def register(self, name: str, queue: Queue, handler: TaskHandler, **kwargs: Any) -> None:
        self._queue.register(name, queue, _accounted(name, handler), **kwargs)

    def register_periodic(
        self, name: str, queue: Queue, handler: TaskHandler, *, cron: str
    ) -> None:
        self._queue.register_periodic(name, queue, _accounted(name, handler), cron=cron)


def _accounted(name: str, handler: TaskHandler) -> TaskHandler:
    async def run(payload: dict[str, Any]) -> Any:
        with llm_accounting() as tokens:
            try:
                return await handler(payload)
            finally:
                if tokens.total:
                    log.info(
                        "job.llm_tokens",
                        task=name,
                        tenant_id=payload.get("tenant_id"),
                        input_tokens=tokens.input,
                        output_tokens=tokens.output,
                    )

    return run


def register_handlers(container: Container) -> None:
    if container.tasks is None:
        return
    queue = _AccountedQueue(container.tasks)
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

    async def feedback_project(payload: dict[str, Any]) -> None:
        """Learn from one feedback record (idempotent; see modules.feedback)."""
        feedback = container.services.get("feedback")
        if feedback is not None:
            await feedback.project(payload["tenant_id"], payload["feedback_id"])

    async def memory_index(payload: dict[str, Any]) -> None:
        tenant_id, memory_ids = payload["tenant_id"], list(payload["memory_ids"])
        indexer = container.services.get("indexer")
        if indexer is not None:
            await indexer.index_memories(tenant_id, memory_ids)
        graph = container.services.get("graph")
        if graph is not None:
            await graph.enrich_memories(tenant_id, memory_ids)
        await _bump_for(tenant_id, memory_ids)

    async def _bump_for(tenant_id: str, memory_ids: Sequence[str]) -> None:
        """Move the revisions now that the memory is *findable*, not when it was written.

        The writing transaction already bumps, but it does so while enqueuing this job — so
        a context request arriving in between builds a bundle that cannot see the new memory
        yet and caches it under the new revision. Nothing moved the revision again, so that
        empty bundle stayed addressed for the whole cache TTL: a memory written now was
        invisible to the query that motivated it for the next five minutes.

        Bumping again here closes the window. It costs one extra build per write, which is
        the correct trade: a cache that serves answers known to be stale is not a cache.
        """
        if not memory_ids:
            return
        async with uow_factory() as uow:
            memories = await uow.memories.get_many(tenant_id, memory_ids)
            await bump_memory_revisions(uow, memories)
            # what the user said about themselves keeps their pinned profile block current
            for user_id in sorted(
                {m.scope.user_id for m in memories if m.memory_type in PROFILE_MEMORY_TYPES}
                - {None}
            ):
                await ProfileService.enqueue_refresh(uow, tenant_id, str(user_id))
            # A deletion is absent from get_many. Invalidate after index removal too:
            # a read between the SQL commit and this job could cache the stale index.
            if len(memories) < len(set(memory_ids)):
                await uow.revisions.bump(tenant_id, RevisionKind.TENANT)
            await uow.commit()

    async def memory_expire(payload: dict[str, Any]) -> None:
        """Expire SHORT_TERM memories and remove their search and graph projections."""
        async with uow_factory() as uow:
            expired = await uow.memories.expire_due(now=datetime.now(UTC))
            by_tenant: dict[str, list[str]] = {}
            for tenant_id, memory_id in expired:
                by_tenant.setdefault(tenant_id, []).append(memory_id)
            for tenant_id, ids in by_tenant.items():
                await bump_memory_revisions(uow, await uow.memories.get_many(tenant_id, ids))
                # Projection cleanup must survive a crash after the expiry commit.
                await uow.enqueue(
                    JobSpec(
                        task_name=TASK_MEMORY_INDEX,
                        queue=Queue.EMBEDDING,
                        payload={"tenant_id": tenant_id, "memory_ids": ids},
                        tenant_id=tenant_id,
                    )
                )
            await uow.commit()
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

    async def retention_sweep(payload: dict[str, Any]) -> None:
        """Forget canonical memories past their tenant's retention (modules/tenancy)."""
        await container.services["retention"].sweep()

    async def read_audit_purge(payload: dict[str, Any]) -> None:
        """Drop who-read-what rows past TASKS.read_audit_retention_days, one batch a run."""
        n = await container.services["read_audit"].purge(
            older_than_days=TASKS.read_audit_retention_days
        )
        if n:
            log.info("read_audit.purged", count=n)

    async def memory_reflect(payload: dict[str, Any]) -> None:
        """Derive insights over each principal's recent memories (LLM use ``reflection``)."""
        reflection = container.services.get("reflection")
        if reflection is not None:
            await reflection.reflect_all()

    async def memory_connect(payload: dict[str, Any]) -> None:
        """Connect memories nothing compared at ingest (LLM use ``memory_connections``)."""
        connections = container.services.get("connections")
        if connections is not None:
            await connections.connect_all()

    async def summary_refresh(payload: dict[str, Any]) -> None:
        """Fold a thread's new messages into its durable summary."""
        await container.services["thread_summaries"].refresh(
            payload["tenant_id"],
            payload["thread_id"],
            principal_id=payload.get("principal_id"),
        )

    async def profile_refresh(payload: dict[str, Any]) -> None:
        """Keep a user's pinned ``user`` block from their USER and PREFERENCE memories."""
        await container.services["profile"].refresh_user(payload["tenant_id"], payload["user_id"])

    async def profile_query(payload: dict[str, Any]) -> None:
        """Answer one block's standing question into its text."""
        await container.services["profile"].answer_query(
            payload["tenant_id"], payload["scope_key"], payload["block"]
        )

    async def profile_queries(payload: dict[str, Any]) -> None:
        """Claim the standing questions that are due."""
        await container.services["profile"].schedule_due()

    async def prefetch_learn(payload: dict[str, Any]) -> None:
        """Fold settled agent-tool pulls into what the context pre-includes."""
        await container.services["agent_tools"].learn_prefetch()

    async def tools_index(payload: dict[str, Any]) -> None:
        """Embed changed catalog entries for tool search."""
        await container.services["tool_index"].index(payload["tenant_id"], payload["tool_ids"])

    async def tools_learn(payload: dict[str, Any]) -> None:
        """Fold recorded tool calls into stored procedures and graph edges."""
        await container.services["tool_learning"].learn(payload.get("tenant_id"))

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

    async def outbox_purge(payload: dict[str, Any]) -> None:
        """Delete outbox rows whose job the queue already owns.

        The outbox exists so that a business write and the job it schedules commit or fail
        together. Once the relay has handed the job over, the row is a receipt - and
        nothing deleted them, so the table grew with every write forever. Rows marked dead
        are kept: those are the ones somebody has to look at.
        """
        async with uow_factory() as uow:
            n = await uow.outbox.purge_dispatched(older_than_seconds=TASKS.outbox_retention_seconds)
            await uow.commit()
        if n:
            log.info("outbox.purged", count=n)

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
            await recover(seconds_since_heartbeat=TASKS.stalled_after_seconds)
        for extra in container.services.get("extra_reconcilers", []):
            await extra()

    async def archive_purge(payload: dict[str, Any]) -> None:
        archiver = container.services.get("archive_service")
        if archiver is not None:
            await archiver.purge_staged_payloads()

    queue.register_periodic(
        "periodic.retention", Queue.RECONCILE, retention_sweep, cron="37 3 * * *"
    )
    queue.register_periodic(
        "periodic.read_audit_purge", Queue.RECONCILE, read_audit_purge, cron="7 * * * *"
    )
    queue.register(TASK_PROCESS_OBSERVATION, Queue.CHAT_FAST, process_observation, retries=5)
    queue.register("document.parse", Queue.DOCUMENT_PARSE, document_parse, retries=3)
    queue.register("document.index", Queue.EMBEDDING, document_index, retries=5)
    queue.register(TASK_MEMORY_INDEX, Queue.EMBEDDING, memory_index, retries=5)
    queue.register(TASK_FEEDBACK_PROJECT, Queue.RECONCILE, feedback_project, retries=3)
    queue.register(TASK_MEMORY_EXPIRE, Queue.RECONCILE, memory_expire, retries=0)
    queue.register(TASK_MEMORY_FORGET, Queue.RECONCILE, memory_forget, retries=0)
    queue.register(TASK_RECONCILE, Queue.RECONCILE, reconcile, retries=0)
    queue.register(TASK_ARCHIVE_PURGE, Queue.ARCHIVE, archive_purge, retries=0)
    every = max(1, TASKS.periodic_reconcile_seconds // 60)
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
        "periodic.outbox_purge", Queue.RECONCILE, outbox_purge, cron="53 * * * *"
    )
    queue.register_periodic(
        "periodic.memory_expire", Queue.RECONCILE, memory_expire, cron="29 * * * *"
    )
    queue.register_periodic(
        "periodic.memory_forget", Queue.RECONCILE, memory_forget, cron="11 4 * * *"
    )
    if container.settings.llm.enabled:
        # Registered whenever a model may be reached: whether one tenant may use it is
        # decided per identity inside the job (a tenant key, workspace key or policy can
        # appear at any time), and a job with nothing payable scans nothing. Connections are
        # offset from reflection's hour so the two passes do not contend for one worker.
        queue.register(TASK_MEMORY_REFLECT, Queue.RECONCILE, memory_reflect, retries=0)
        queue.register_periodic(
            "periodic.memory_reflect", Queue.RECONCILE, memory_reflect, cron="53 */6 * * *"
        )
        queue.register(TASK_MEMORY_CONNECT, Queue.RECONCILE, memory_connect, retries=0)
        queue.register_periodic(
            "periodic.memory_connect", Queue.RECONCILE, memory_connect, cron="19 */6 * * *"
        )
    queue.register(TASK_SUMMARY_REFRESH, Queue.SUMMARY, summary_refresh, retries=3)
    queue.register(TASK_PROFILE_REFRESH, Queue.SUMMARY, profile_refresh, retries=3)
    queue.register(TASK_PROFILE_QUERY, Queue.SUMMARY, profile_query, retries=1)
    queue.register_periodic(
        "periodic.profile_queries", Queue.RECONCILE, profile_queries, cron="*/5 * * * *"
    )
    queue.register(TASK_TOOLS_INDEX, Queue.EMBEDDING, tools_index, retries=5)
    queue.register_periodic(
        "periodic.prefetch_learn", Queue.RECONCILE, prefetch_learn, cron="*/5 * * * *"
    )
    queue.register(TASK_TOOLS_LEARN, Queue.RECONCILE, tools_learn, retries=0)
    queue.register_periodic(
        "periodic.tools_learn", Queue.RECONCILE, tools_learn, cron="*/5 * * * *"
    )
    queue.register(TASK_ARCHIVE_STAGE, Queue.ARCHIVE, archive_stage, retries=10)
    queue.register(TASK_OUTBOX_SWEEP, Queue.RECONCILE, outbox_sweep, retries=0)
    # Registered *and scheduled*. It was only registered, so the handler existed and nothing
    # ever called it — and the outbox is not an optimisation, it is the only path from a
    # committed write to the work that turns it into a memory. The fast path after commit is
    # best-effort by design (a crash between COMMIT and dispatch leaves the row behind), and
    # this sweep is the repair. Without it those rows sit at attempts=0 forever: measured on
    # a running service, 441 undispatched rows, 255 of them memory.process_observation, the
    # oldest 42 minutes old. Every one of those writes was answered 202 and never happened.
    #
    # Every minute, not every five: this is the floor on how late a write can become a
    # memory when the fast path misses it.
    queue.register_periodic(
        "periodic.outbox_sweep", Queue.RECONCILE, outbox_sweep, cron="* * * * *"
    )
    queue.register(TASK_IDEMPOTENCY_PURGE, Queue.RECONCILE, idempotency_purge, retries=0)
    for extra in container.services.get("extra_task_registrars", []):
        extra(container)
