"""Tool memory service (design/TOOL_MEMORY.md): the catalog and call records.

The service never runs a tool. The catalog says what each tool is and does (side effects,
the entity types its arguments name); ``record`` stores what an agent called and counts it;
a run's outcome is a projection of its feedback (``modules.feedback``). Everything learned from
the records - procedures, graph edges, approval suggestions - is the learning job's
(``modules.tools.learning``); the advice read back is ``modules.tools.hints``.

Nothing is read that the caller's visibility keys do not cover, and a call record carries the
same audience keys a memory written by the same caller would.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Sequence
from typing import Any, Final

import zstandard

from memory_service.config.constants import ARCHIVE
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, Visibility
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.tools import (
    SubCall,
    ToolDescriptor,
    ToolInvocation,
    ToolStats,
    ToolStatus,
    stable_hash,
)
from memory_service.modules.archive.segments import tenant_shard
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.memory.pipeline import keys_for, scope_for
from memory_service.modules.tenancy.gate import guard_workspace_visibility
from memory_service.modules.tools.patterns import task_pattern
from memory_service.modules.tools.trajectories import flatten
from memory_service.observability.logging import get_logger
from memory_service.ports.blob import BlobAlreadyExists
from memory_service.ports.intelligence import MemoryCandidate
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork

log = get_logger(__name__)

TASK_TOOLS_INDEX: Final = "tools.index"
TASK_TOOLS_LEARN: Final = "tools.learn"
#: Stores a tool output whose upload failed when it was recorded (the outbox retries it).
TASK_TOOLS_OUTPUT: Final = "tools.output"
#: What a call's row keeps of its output (the context reads it); a longer output is stored
#: whole in the blob store, by reference.
SUMMARY_CHARS: Final = 600
#: Entries one catalog listing or upsert handles.
CATALOG_MAX: Final = 500


class ToolMemoryService:
    def __init__(
        self,
        authz: AuthorizationService,
        *,
        blob: Any = None,
        blob_bucket: str = "memory-tool-outputs",
    ) -> None:
        self.authz = authz
        self.blob = blob
        self.blob_bucket = blob_bucket

    # ------------------------------------------------------------------ catalog
    async def put_catalog(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        entries: Sequence[tuple[ToolDescriptor, frozenset[str]]],
    ) -> list[ToolDescriptor]:
        """Upsert entries in the caller's workspace (or tenant-wide without one); an existing
        entry changes only in the fields the caller sent. Changed entries are re-indexed for
        tool search, and bundles that carried tool hints go stale."""
        stored: list[ToolDescriptor] = []
        changed: list[str] = []
        for entry, fields in entries:
            scoped = entry.model_copy(
                update={"tenant_id": ctx.tenant_id, "workspace_id": ctx.workspace_id}
            )
            saved, was_changed = await uow.tools.upsert(scoped, fields=fields)
            stored.append(saved)
            if was_changed:
                changed.append(saved.tool_id)
        if changed:
            await uow.enqueue(
                JobSpec(
                    task_name=TASK_TOOLS_INDEX,
                    queue=Queue.EMBEDDING,
                    payload={"tenant_id": ctx.tenant_id, "tool_ids": changed},
                    tenant_id=ctx.tenant_id,
                )
            )
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.TENANT, "")
        log.info("tools.catalog", upserted=len(stored), changed=len(changed), **ctx.log_fields())
        return stored

    async def catalog(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        names: Sequence[str] | None,
        *,
        after: str = "",
        limit: int = CATALOG_MAX,
    ) -> list[tuple[ToolDescriptor, ToolStats]]:
        """A page of the catalog visible in this scope, by name after ``after``."""
        entries = await uow.tools.catalog(
            ctx.tenant_id, workspace_id=ctx.workspace_id, names=names, limit=limit, after=after
        )
        stats = await uow.tools.stats(ctx.tenant_id, [e.name for e in entries])
        return [(e, stats.get(e.name) or ToolStats(tool_name=e.name)) for e in entries]

    # ------------------------------------------------------------------ recording
    async def record(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        tool: str,
        args: dict[str, Any],
        output: Any = None,
        output_summary: str | None = None,
        status: ToolStatus = "ok",
        error_class: str | None = None,
        latency_ms: float | None = None,
        cost: float | None = None,
        task: str = "",
        step: int | None = None,
        sub_calls: Sequence[dict[str, Any]] | None = None,
        visibility: Visibility = Visibility.PRIVATE,
    ) -> tuple[ToolInvocation, bool]:
        """Persist one call and count it. Idempotent on (run, step, tool, args_hash): a
        replayed graph step re-reads its row instead of inflating the statistics."""
        await guard_workspace_visibility(uow, self.authz, ctx, visibility)
        descriptor = await uow.tools.by_name(
            ctx.tenant_id, tool, workspace_id=ctx.workspace_id
        ) or await uow.tools.ensure(ctx.tenant_id, tool)
        summary, blob_ref, output_digest = await self._store_output(
            uow, ctx, output, output_summary
        )
        invocation = ToolInvocation(
            tenant_id=ctx.tenant_id,
            tool_id=descriptor.tool_id,
            tool_name=descriptor.name,
            tool_version=descriptor.version,
            run_id=ctx.agent_run_id,
            thread_id=ctx.thread_id,
            turn_id=ctx.turn_id,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            agent_id=ctx.agent_id,
            principal_id=ctx.principal_id,
            step=step if step is not None else await self._next_step(uow, ctx),
            args_redacted=descriptor.redacted_args(args),
            # The *full* arguments: only the redacted copy is persisted, but the hash has to
            # tell apart calls that differ solely in a redacted field. It is one-way.
            args_hash=stable_hash(args),
            output_summary=summary,
            output_digest=output_digest,
            output_blob_ref=blob_ref,
            output_fields=flatten(output) if output is not None else {},
            status=status,
            error_class=error_class,
            latency_ms=latency_ms,
            cost=cost,
            task=task,
            task_pattern=task_pattern(task) if task else None,
            sub_calls=[SubCall.model_validate(c) for c in (sub_calls or [])],
            visibility_keys=_visibility_keys(ctx, visibility),
        )
        stored, created = await uow.tools.record(invocation)
        if created:
            await uow.tools.count_call(
                ctx.tenant_id,
                stored.tool_name,
                ok=stored.succeeded,
                latency_ms=stored.latency_ms,
                at=stored.occurred_at,
            )
        return stored, created

    async def _next_step(self, uow: UnitOfWork, ctx: MemoryExecutionContext) -> int:
        if not ctx.agent_run_id:
            return 0
        existing = await uow.tools.invocations_for_run(ctx.tenant_id, ctx.agent_run_id)
        return max((i.step for i in existing), default=-1) + 1

    async def _store_output(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        output: Any,
        provided_summary: str | None,
    ) -> tuple[str, str | None, str | None]:
        """The row's summary, the reference to the whole output, and its digest. An output
        longer than the summary is stored whole (``store_output``); an upload that fails is
        retried by the outbox (``TASK_TOOLS_OUTPUT``, in the call's transaction) under the
        same content address, so the reference holds and nothing is cut."""
        if output is None:
            return (provided_summary or "", None, None)
        text = output if isinstance(output, str) else _render(output)
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        summary = provided_summary or text[:SUMMARY_CHARS]
        if len(text) <= SUMMARY_CHARS:
            return (summary, None, digest)
        key = output_key(ctx.tenant_id, digest)
        try:
            if self.blob is None:
                raise RuntimeError("no blob store")
            await store_output(self.blob, self.blob_bucket, key, text)
        except Exception as exc:  # the store's own errors vary by provider
            log.warning("tools.output_deferred", error=type(exc).__name__, key=key)
            await uow.enqueue(
                JobSpec(
                    task_name=TASK_TOOLS_OUTPUT,
                    queue=Queue.RECONCILE,
                    payload={"bucket": self.blob_bucket, "key": key, "text": text},
                    tenant_id=ctx.tenant_id,
                    idempotency_key=f"tool-output:{key}",
                )
            )
        return (summary, f"{self.blob_bucket}/{key}", digest)


def output_key(tenant_id: str, digest: str) -> str:
    """Where a tool output is stored: by its digest, under the tenant, sharded as the archive's
    segments are (``archive.segments``), compressed with the archive's codec."""
    shard = tenant_shard(tenant_id, ARCHIVE.tenant_shards)
    return f"tenant-shard={shard:03d}/tenant={tenant_id}/tool-output/sha256={digest}.txt.zst"


async def store_output(blob: Any, bucket: str, key: str, text: str) -> None:
    """Store an output once (the same digest is the same object): zstd, as the archive."""
    data = zstandard.ZstdCompressor(level=ARCHIVE.zstd_level).compress(
        text.encode("utf-8", "replace")
    )
    # content-addressed: an earlier call that stored the same output stored this object
    with contextlib.suppress(BlobAlreadyExists):
        await blob.put(bucket, key, data, content_type="application/zstd")


def _visibility_keys(ctx: MemoryExecutionContext, visibility: Visibility) -> list[str]:
    """Tool records are audienced exactly like memories: the same anchor rules and the same
    audience keys, so an agent's tool chatter reaches a user or a group only when it was
    explicitly shared, and a hand-off is visible to the child run and no further. The default
    is PRIVATE, the agent's own durable store: a procedure is learned across its runs."""
    anchor = MemoryCandidate(
        content="", memory_type=MemoryType.TOOL, lifetime=Lifetime.SHORT_TERM, visibility=visibility
    )
    return keys_for(scope_for(anchor, ctx), visibility, ctx)


def _render(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)[:20000]
    except (TypeError, ValueError):
        return str(value)[:20000]
