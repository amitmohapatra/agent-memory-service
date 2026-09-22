"""ArchiveService: staged source -> immutable verified blob -> manifest -> purge after grace.

Invariants
----------
* A staged payload is removed from the hot database only after the blob's generation and
  checksum have been verified AND the manifest row is committed AND the grace period passed.
* Every step is idempotent: re-running ``archive_thread`` after a crash never duplicates a
  segment (object keys are unique; messages already ARCHIVED are skipped).
* A blob outage fails the job (retried by the queue); nothing is marked archived.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from memory_service.config.constants import BLOB_LIFECYCLE, ArchiveSettings, BlobLifecycle
from memory_service.config.settings import BlobSettings
from memory_service.domain.conversation import Message
from memory_service.domain.enums import ArchiveStatus
from memory_service.domain.errors import CorruptSource, DependencyUnavailable, NotFound
from memory_service.modules.archive.segments import build_segment, plan_segments, read_segment
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import (
    archive_bytes_total,
    reconciler_repairs_total,
    stage_seconds,
)
from memory_service.observability.tracing import span
from memory_service.ports.blob import BlobStore
from memory_service.ports.repositories import ArchiveSegment
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)


class ArchiveService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        blob: BlobStore,
        *,
        archive: ArchiveSettings,
        blob_settings: BlobSettings,
        lifecycle: BlobLifecycle = BLOB_LIFECYCLE,
    ) -> None:
        self.uow_factory = uow_factory
        self.blob = blob
        self.cfg = archive
        self.buckets = blob_settings
        self.lifecycle = lifecycle

    # -- write path -------------------------------------------------------------
    async def archive_thread(self, tenant_id: str, thread_id: str) -> list[str]:
        """Archive every STAGED message of the thread. Returns verified segment ids."""
        async with self.uow_factory() as uow:
            staged = await uow.messages.list_staged(
                tenant_id=tenant_id, thread_id=thread_id, limit=self.cfg.segment_max_messages * 4
            )
        if not staged:
            return []
        staged.sort(key=lambda m: m.sequence)
        segment_ids: list[str] = []
        for plan in plan_segments(
            staged,
            target_compressed_bytes=self.cfg.segment_target_bytes,
            max_messages=self.cfg.segment_max_messages,
        ):
            segment_ids.append(await self._archive_messages(tenant_id, thread_id, plan.messages))
        return segment_ids

    async def _archive_messages(
        self, tenant_id: str, thread_id: str, messages: list[Message]
    ) -> str:
        with (
            span("archive.segment", tenant_id=tenant_id),
            stage_seconds.labels("archive.segment").time(),
        ):
            built = build_segment(
                messages,
                tenant_id=tenant_id,
                thread_id=thread_id,
                shards=self.cfg.tenant_shards,
                zstd_level=self.cfg.zstd_level,
            )
            bucket = self.buckets.chat_bucket
            # 1. manifest row in UPLOADING state (so a crash after upload is reconcilable)
            async with self.uow_factory() as uow:
                await uow.archive.add(
                    ArchiveSegment(
                        segment_id=built.segment_id,
                        tenant_id=tenant_id,
                        kind="chat",
                        thread_id=thread_id,
                        bucket=bucket,
                        key=built.key,
                        size_bytes=len(built.data),
                        raw_bytes=built.raw_bytes,
                        checksum_sha256=built.checksum_sha256,
                        message_count=len(messages),
                        first_sequence=built.first_sequence,
                        last_sequence=built.last_sequence,
                        first_at=built.first_at,
                        last_at=built.last_at,
                        status="UPLOADING",
                        manifest=built.manifest,
                    )
                )
                await uow.commit()
            # 2. immutable upload + verification
            try:
                ref = await self.blob.put(
                    bucket,
                    built.key,
                    built.data,
                    content_type="application/zstd",
                    checksum_sha256=built.checksum_sha256,
                    if_generation_match=0,
                    metadata={
                        "tenant": tenant_id,
                        "thread": thread_id,
                        "segment": built.segment_id,
                    },
                )
                verified = await self.blob.verify(ref)
            except Exception as exc:
                async with self.uow_factory() as uow:
                    await uow.archive.mark_failed(
                        built.segment_id, error=f"{type(exc).__name__}: {exc}"
                    )
                    await uow.commit()
                raise DependencyUnavailable(
                    f"blob store unavailable: {type(exc).__name__}"
                ) from exc
            if not verified:
                async with self.uow_factory() as uow:
                    await uow.archive.mark_failed(built.segment_id, error="verification failed")
                    await uow.commit()
                raise CorruptSource(
                    f"segment {built.segment_id} failed verification; messages stay STAGED"
                )
            archive_bytes_total.labels("chat", "raw").inc(built.raw_bytes)
            archive_bytes_total.labels("chat", "compressed").inc(len(built.data))
            # 3. manifest VERIFIED + messages ARCHIVED, atomically
            now = datetime.now(UTC)
            async with self.uow_factory() as uow:
                await uow.archive.mark_verified(
                    built.segment_id, generation=ref.generation, verified_at=now
                )
                await uow.messages.mark_archived(
                    built.message_ids, segment_id=built.segment_id, archived_at=now
                )
                await uow.commit()
            log.info(
                "archive.segment_verified",
                tenant_id=tenant_id,
                thread_id=thread_id,
                segment_id=built.segment_id,
                messages=len(messages),
                raw_bytes=built.raw_bytes,
                compressed_bytes=len(built.data),
            )
            return built.segment_id

    async def purge_staged_payloads(self, *, now: datetime | None = None, limit: int = 1000) -> int:
        """Remove large staged payloads whose segment was verified before the grace period."""
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.cfg.purge_grace_seconds)
        async with self.uow_factory() as uow:
            n = await uow.messages.purge_payloads(
                archived_before=cutoff, min_bytes=self.cfg.purge_min_payload_bytes, limit=limit
            )
            await uow.commit()
        if n:
            log.info("archive.purged_payloads", count=n)
        return n

    # -- read path --------------------------------------------------------------
    async def load_message_content(self, message: Message) -> str:
        """Return the message text, reading the archive segment when the hot payload was purged."""
        if message.archive_status is not ArchiveStatus.PURGED:
            return message.content
        async with self.uow_factory() as uow:
            segments = await uow.archive.list_for_thread(message.tenant_id, message.thread_id)
        for seg in segments:
            if seg.status != "VERIFIED" or not (seg.first_sequence or 0) <= message.sequence <= (
                seg.last_sequence or 0
            ):
                continue
            data = await self.blob.get(seg.bucket, seg.key, generation=seg.generation)
            for record in read_segment(data):
                if record["message_id"] == message.message_id:
                    if record["content_hash"] != message.content_hash:
                        raise CorruptSource(
                            "archived content hash does not match the canonical record"
                        )
                    return str(record["content"])
        raise NotFound(f"archived content for {message.message_id} not found")

    # -- reconciliation ---------------------------------------------------------
    async def reconcile(
        self, *, stale_after_seconds: int = 900, verify_sample: int = 50
    ) -> dict[str, int]:
        """Periodic repair. Safe to run concurrently with archiving (idempotent steps)."""
        now = datetime.now(UTC)
        stale = now - timedelta(seconds=stale_after_seconds)
        report: dict[str, int] = {
            "requeued_threads": 0,
            "uploading_repaired": 0,
            "failed_requeued": 0,
            "verify_mismatch": 0,
        }

        async with self.uow_factory() as uow:
            staged = await uow.messages.list_staged(older_than=stale, limit=2000)
            uploading = await uow.archive.list_by_status("UPLOADING", older_than=stale)
            failed = await uow.archive.list_by_status("FAILED", older_than=stale)
            verified = await uow.archive.list_verified(limit=verify_sample)

        # ACKed but archive incomplete -> requeue the thread
        from memory_service.modules.jobs.registry import TASK_ARCHIVE_STAGE
        from memory_service.ports.tasks import JobSpec, Queue

        threads = sorted({(m.tenant_id, m.thread_id) for m in staged})
        if threads:
            async with self.uow_factory() as uow:
                for tenant_id, thread_id in threads:
                    if await uow.enqueue(
                        JobSpec(
                            task_name=TASK_ARCHIVE_STAGE,
                            queue=Queue.ARCHIVE,
                            payload={"tenant_id": tenant_id, "thread_id": thread_id},
                            idempotency_key=f"archive:thread:{tenant_id}:{thread_id}",
                            lock=f"archive:{tenant_id}:{thread_id}",
                            tenant_id=tenant_id,
                        )
                    ):
                        report["requeued_threads"] += 1
                        reconciler_repairs_total.labels("acked_but_unarchived").inc()
                await uow.commit()

        # upload happened but the verified-commit did not (crash between steps 2 and 3)
        for seg in uploading + failed:
            head = None
            try:
                head = await self.blob.head(seg.bucket, seg.key)
                ok = head.checksum_sha256 == seg.checksum_sha256 and await self.blob.verify(head)
            except Exception:
                ok = False
            async with self.uow_factory() as uow:
                if ok and head is not None:
                    await uow.archive.mark_verified(
                        seg.segment_id, generation=head.generation, verified_at=now
                    )
                    await uow.messages.mark_archived(
                        list(seg.manifest.get("message_ids", [])),
                        segment_id=seg.segment_id,
                        archived_at=now,
                    )
                    report["uploading_repaired"] += 1
                    reconciler_repairs_total.labels("manifest_missing").inc()
                else:
                    # object absent/corrupt: the messages are still STAGED; drop the dead manifest
                    await uow.archive.mark_failed(
                        seg.segment_id, error="object missing or unverifiable at reconcile"
                    )
                    report["failed_requeued"] += 1
                await uow.commit()

        # archive checksum mismatch on a sample of verified segments
        #
        # A mismatch used to be logged and nothing else, so every reconcile pass rediscovered
        # the same segment and logged it again — observed firing every five minutes,
        # indefinitely, on one segment. That is the worst of both: a real corruption is never
        # acted on, and the noise hides the next one. A durable mismatch is corruption and is
        # quarantined exactly like a missing object (the source messages stay STAGED and are
        # requeued); a read that *failed* is transient and is left for the next pass.
        for seg in verified:
            try:
                ref = await self.blob.head(seg.bucket, seg.key)
                mismatch = ref.checksum_sha256 != seg.checksum_sha256
                if not mismatch:
                    mismatch = not await self.blob.verify(ref)
            except Exception as exc:
                report["verify_unreadable"] = report.get("verify_unreadable", 0) + 1
                log.warning(
                    "archive.verify_unreadable",
                    segment_id=seg.segment_id,
                    key=seg.key,
                    error=type(exc).__name__,
                )
                continue
            if mismatch:
                report["verify_mismatch"] += 1
                reconciler_repairs_total.labels("checksum_mismatch").inc()
                log.error(
                    "archive.verified_segment_mismatch",
                    segment_id=seg.segment_id,
                    key=seg.key,
                    action="quarantined",
                )
                async with self.uow_factory() as uow:
                    await uow.archive.mark_failed(
                        seg.segment_id, error="checksum mismatch at reconcile"
                    )
                    await uow.commit()
        return report

    def lifecycle_policy(self) -> dict[str, Any]:
        """GCS lifecycle configuration (``constants.BLOB_LIFECYCLE``: autoclass or explicit)."""
        if self.lifecycle.policy == "autoclass":
            return {"autoclass": {"enabled": True, "terminalStorageClass": "ARCHIVE"}}
        rules = [
            {"action": {"type": "SetStorageClass", "storageClass": cls}, "condition": {"age": days}}
            for cls, days in (
                ("NEARLINE", self.lifecycle.nearline_days),
                ("COLDLINE", self.lifecycle.coldline_days),
                ("ARCHIVE", self.lifecycle.archive_days),
            )
        ]
        return {"lifecycle": {"rule": rules}}
