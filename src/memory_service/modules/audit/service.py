"""The read audit: recorded off the request path, flushed in batches.

A read costs the caller nothing here: ``record`` appends to a bounded queue and returns. A
flusher writes the queue to PostgreSQL every ``flush_every`` seconds or ``batch`` entries,
and once more at shutdown. What this buys is a request path with no audit write on it.

The guarantee, precisely (ADR 0021, restated in ADR 0031). An entry is written unless:

- the process dies without shutting down (SIGKILL, OOM kill, a crash): what was queued and
  not yet committed is lost - at most one ``flush_every`` (1 s) of reads, plus a batch in
  flight;
- the queue is full (``max_pending``, 10 000 entries - the store has stalled for seconds):
  the newest entry is dropped;
- a row cannot be stored even on its own after its batch failed: that row is dropped.

Every drop of the last two kinds is counted (``memory_read_audit_dropped_total``); the first
cannot be counted by the process that died. A graceful stop - SIGTERM to uvicorn, which
runs the app's shutdown and ``Container.close`` - flushes the queue and waits for batches in
flight before the pool closes. It is therefore an operational record with a stated loss
window, not a compliance ledger: a read is never refused or slowed because its audit entry
could not be written. A deployment that needs every read on the record before the caller is
answered needs a transactional write per read (a WAL flush on every recall), which this
module deliberately does not do.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from memory_service.domain.audit import ReadAuditEntry, ReadKind
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.ids import content_hash
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import read_audit_dropped_total
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)
#: ``memory_reads.credential`` is String(512)
CREDENTIAL_MAX = 512


class ReadAudit:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        *,
        max_pending: int = 10_000,
        flush_every: float = 1.0,
        batch: int = 500,
    ) -> None:
        self.uow_factory = uow_factory
        self.queue: asyncio.Queue[ReadAuditEntry] = asyncio.Queue(maxsize=max_pending)
        self.flush_every = flush_every
        self.batch = batch
        self._task: asyncio.Task[None] | None = None
        #: writes taken off the queue and not yet committed; close() waits for them
        self._inflight: set[asyncio.Future[int]] = set()

    def record(
        self,
        ctx: MemoryExecutionContext,
        kind: ReadKind,
        query: str,
        record_ids: Iterable[str],
        *,
        scope_fingerprint: str,
        credential: str,
    ) -> None:
        entry = ReadAuditEntry(
            tenant_id=ctx.tenant_id,
            credential=credential[:CREDENTIAL_MAX],  # a JWT subject can be long; the column is not
            principal=ctx.principal_id,
            kind=kind,
            query_hash=content_hash(query),
            scope_fingerprint=scope_fingerprint,
            record_ids=list(record_ids),
        )
        try:
            self.queue.put_nowait(entry)
        except asyncio.QueueFull:
            read_audit_dropped_total.inc()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="read-audit-flush")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._inflight:
            # a batch taken off the queue is written before the process lets go of it
            await asyncio.gather(*self._inflight, return_exceptions=True)
        await self.flush()

    async def flush(self) -> int:
        """Write what is queued now, in batches. Returns the number of entries written.

        Drains a snapshot of the queue rather than "until empty": under steady traffic the
        queue is never empty, and ``GET /v1/reads`` flushes inline before listing.
        """
        written = 0
        pending = self.queue.qsize()
        while pending > 0:
            entries: list[ReadAuditEntry] = []
            while len(entries) < self.batch and pending > 0:
                try:
                    entries.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
                pending -= 1
            if not entries:
                break
            inflight = asyncio.ensure_future(self._write(entries))
            self._inflight.add(inflight)
            inflight.add_done_callback(self._inflight.discard)
            # shielded: a cancelled flusher does not abandon a batch already off the queue
            written += await asyncio.shield(inflight)
        # A listing that flushed first must also see what the loop took off the queue a
        # moment ago and is still writing (futures of this loop; a test harness may run the
        # loop elsewhere, and a foreign future cannot be awaited).
        loop = asyncio.get_running_loop()
        pending = [f for f in self._inflight if not f.done() and f.get_loop() is loop]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return written

    async def _write(self, entries: list[ReadAuditEntry]) -> int:
        try:
            await self._store(entries)
            return len(entries)
        except Exception as exc:
            log.warning("read_audit.batch_failed", count=len(entries), error=str(exc))
        # One bad row must not sink the other tenants' entries in the batch: write the rows
        # one at a time and count only what is actually lost.
        written = 0
        for entry in entries:
            try:
                await self._store([entry])
                written += 1
            except Exception as exc:
                read_audit_dropped_total.inc()
                log.warning("read_audit.entry_dropped", tenant_id=entry.tenant_id, error=str(exc))
        return written

    async def _store(self, entries: list[ReadAuditEntry]) -> None:
        async with self.uow_factory() as uow:
            await uow.read_audit.add_many(entries)
            await uow.commit()

    async def purge(self, *, older_than_days: int, limit: int = 5000, max_batches: int = 50) -> int:
        """Drop audit rows past the retention horizon: bounded batches, each its own
        transaction, up to ``max_batches`` per call so a backlog drains without one long
        delete and the rest waits for the next run."""
        before = datetime.now(UTC) - timedelta(days=older_than_days)
        purged = 0
        for _ in range(max_batches):
            async with self.uow_factory() as uow:
                n = await uow.read_audit.purge_before(before, limit=limit)
                await uow.commit()
            purged += n
            if n < limit:
                break
        return purged

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.flush_every)
            try:
                await self.flush()
            except Exception as exc:  # _write swallows store errors; this is for the rest
                log.warning("read_audit.flush_loop_failed", error=str(exc))
