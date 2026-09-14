"""Archive protocol against real PostgreSQL + filesystem blob store, with failure injection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from memory_service.domain.conversation import Message, Session, Thread, Turn
from memory_service.domain.enums import ArchiveStatus, MessageRole
from memory_service.domain.errors import CorruptSource, DependencyUnavailable
from memory_service.domain.ids import content_hash
from memory_service.modules.archive.service import ArchiveService

pytestmark = [pytest.mark.integration, pytest.mark.failure]

TENANT = "acme"


async def _seed(uow_factory, thread: Thread, n: int, size: int = 5000) -> list[Message]:
    msgs: list[Message] = []
    async with uow_factory() as uow:
        await uow.threads.add(thread)
        session = Session(thread_id=thread.thread_id, tenant_id=TENANT, user_id="u1")
        await uow.sessions.add(session)
        turn = Turn(
            session_id=session.session_id, thread_id=thread.thread_id, tenant_id=TENANT, sequence=1
        )
        await uow.turns.add(turn)
        for i in range(n):
            content = f"message {i} " + ("lorem ipsum " * (size // 12))
            m = Message(
                thread_id=thread.thread_id,
                session_id=session.session_id,
                turn_id=turn.turn_id,
                tenant_id=TENANT,
                role=MessageRole.USER,
                sequence=i + 1,
                content=content,
                content_hash=content_hash(content),
                author_principal="user:u1",
            )
            await uow.messages.add(m)
            msgs.append(m)
        await uow.commit()
    return msgs


async def _statuses(uow_factory, msgs: list[Message]) -> set[ArchiveStatus]:
    async with uow_factory() as uow:
        out = set()
        for m in msgs:
            stored = await uow.messages.get(TENANT, m.message_id)
            assert stored is not None
            out.add(stored.archive_status)
        return out


@pytest.fixture
def archive(container) -> ArchiveService:
    return container.services["archive_service"]


async def test_archive_thread_verifies_then_marks_and_purges_after_grace(
    container, uow_factory, archive
) -> None:
    thread = Thread(tenant_id=TENANT, owner_user_id="u1")
    msgs = await _seed(uow_factory, thread, 6)
    segment_ids = await archive.archive_thread(TENANT, thread.thread_id)
    assert len(segment_ids) == 1
    async with uow_factory() as uow:
        seg = await uow.archive.get(segment_ids[0])
        assert (
            seg is not None
            and seg.status == "VERIFIED"
            and seg.generation == "1"
            and seg.message_count == 6
        )
        assert await container.blob.verify(await container.blob.head(seg.bucket, seg.key))
        for m in msgs:
            stored = await uow.messages.get(TENANT, m.message_id)
            assert (
                stored is not None
                and stored.archive_status is ArchiveStatus.ARCHIVED
                and stored.content == m.content
            )
    # re-running is idempotent: nothing staged, no new segment
    assert await archive.archive_thread(TENANT, thread.thread_id) == []
    # purge respects the grace period
    assert await archive.purge_staged_payloads() == 0
    future = datetime.now(UTC) + timedelta(
        seconds=container.settings.archive.purge_grace_seconds + 1
    )
    assert await archive.purge_staged_payloads(now=future) == 6
    async with uow_factory() as uow:
        purged = await uow.messages.get(TENANT, msgs[2].message_id)
        assert (
            purged is not None
            and purged.archive_status is ArchiveStatus.PURGED
            and purged.content == ""
        )
    # content is still readable from the verified segment, hash-checked
    assert await archive.load_message_content(purged) == msgs[2].content


async def test_blob_outage_leaves_messages_staged(container, uow_factory, archive) -> None:
    thread = Thread(tenant_id=TENANT, owner_user_id="u1")
    msgs = await _seed(uow_factory, thread, 3)
    container.blob.available = False
    with pytest.raises(DependencyUnavailable):
        await archive.archive_thread(TENANT, thread.thread_id)
    assert await _statuses(uow_factory, msgs) == {ArchiveStatus.STAGED}
    async with uow_factory() as uow:
        failed = await uow.archive.list_by_status("FAILED")
        assert len(failed) == 1 and "ConnectionError" in (failed[0].last_error or "")
    container.blob.available = True
    assert len(await archive.archive_thread(TENANT, thread.thread_id)) == 1


async def test_failed_verification_never_marks_archived(container, uow_factory, archive) -> None:
    thread = Thread(tenant_id=TENANT, owner_user_id="u1")
    msgs = await _seed(uow_factory, thread, 2)
    container.blob.corrupt_next_verify = True
    with pytest.raises(CorruptSource):
        await archive.archive_thread(TENANT, thread.thread_id)
    assert await _statuses(uow_factory, msgs) == {ArchiveStatus.STAGED}


async def test_reconciler_repairs_crash_between_upload_and_commit(
    container, uow_factory, archive
) -> None:
    """Simulate: object uploaded + verified, process died before the VERIFIED commit."""
    thread = Thread(tenant_id=TENANT, owner_user_id="u1")
    msgs = await _seed(uow_factory, thread, 2)
    from memory_service.modules.archive.segments import build_segment
    from memory_service.ports.repositories import ArchiveSegment

    built = build_segment(msgs, tenant_id=TENANT, thread_id=thread.thread_id, shards=4)
    async with uow_factory() as uow:
        await uow.archive.add(
            ArchiveSegment(
                segment_id=built.segment_id,
                tenant_id=TENANT,
                thread_id=thread.thread_id,
                bucket=container.settings.blob.chat_bucket,
                key=built.key,
                size_bytes=len(built.data),
                raw_bytes=built.raw_bytes,
                checksum_sha256=built.checksum_sha256,
                message_count=2,
                first_sequence=1,
                last_sequence=2,
                status="UPLOADING",
                manifest=built.manifest,
            )
        )
        await uow.commit()
    await container.blob.put(container.settings.blob.chat_bucket, built.key, built.data)
    async with container.database.engine.begin() as conn:  # make it look old
        await conn.execute(
            text("UPDATE archive_segments SET created_at = now() - interval '1 hour'")
        )
    report = await archive.reconcile(stale_after_seconds=60)
    assert report["uploading_repaired"] == 1
    async with uow_factory() as uow:
        assert (await uow.archive.get(built.segment_id)).status == "VERIFIED"
    assert await _statuses(uow_factory, msgs) == {ArchiveStatus.ARCHIVED}


async def test_reconciler_requeues_acked_but_unarchived_messages(
    container, uow_factory, archive
) -> None:
    thread = Thread(tenant_id=TENANT, owner_user_id="u1")
    await _seed(uow_factory, thread, 1)
    async with container.database.engine.begin() as conn:
        await conn.execute(text("UPDATE messages SET created_at = now() - interval '1 hour'"))
    from memory_service.modules.jobs.registry import register_handlers

    register_handlers(container)
    report = await archive.reconcile(stale_after_seconds=60)
    assert report["requeued_threads"] == 1
    assert any(j.task_name == "archive.stage_message" for j in container.tasks.jobs.values())
    # the same thread is not requeued twice while the job is pending
    assert (await archive.reconcile(stale_after_seconds=60))["requeued_threads"] == 0


async def test_reconciler_flags_checksum_mismatch_on_verified_segment(
    container, uow_factory, archive, tmp_path
) -> None:
    thread = Thread(tenant_id=TENANT, owner_user_id="u1")
    await _seed(uow_factory, thread, 2)
    (seg_id,) = await archive.archive_thread(TENANT, thread.thread_id)
    async with uow_factory() as uow:
        seg = await uow.archive.get(seg_id)
    path = container.blob.root / seg.bucket / seg.key
    path.write_bytes(b"corrupted")
    report = await archive.reconcile(stale_after_seconds=60)
    assert report["verify_mismatch"] == 1


def test_lifecycle_policy_from_settings(archive) -> None:
    assert archive.lifecycle_policy() == {
        "autoclass": {"enabled": True, "terminalStorageClass": "ARCHIVE"}
    }
    archive.buckets = archive.buckets.model_copy(update={"lifecycle_policy": "explicit"})
    rules = archive.lifecycle_policy()["lifecycle"]["rule"]
    assert [r["action"]["storageClass"] for r in rules] == ["NEARLINE", "COLDLINE", "ARCHIVE"]
