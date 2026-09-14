"""Durability protocol: source row + job commit atomically; nothing is acknowledged otherwise."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from memory_service.domain.conversation import Message, Session, Thread, Turn
from memory_service.domain.enums import JobStatus, MessageRole, ObservationKind
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.domain.revisions import RevisionKind
from memory_service.ports.tasks import JobSpec, Queue

pytestmark = [pytest.mark.integration]

TENANT = "acme"


async def _noop(payload):
    return payload


def _thread(tenant: str = TENANT) -> Thread:
    return Thread(tenant_id=tenant, owner_user_id="u1", title="hello")


async def _seed_thread(uow, thread: Thread) -> tuple[Session, Turn]:
    await uow.threads.add(thread)
    session = Session(
        thread_id=thread.thread_id, tenant_id=thread.tenant_id, user_id="u1", client="test"
    )
    await uow.sessions.add(session)
    seq = await uow.turns.next_sequence(thread.tenant_id, thread.thread_id)
    turn = Turn(
        session_id=session.session_id,
        thread_id=thread.thread_id,
        tenant_id=thread.tenant_id,
        sequence=seq,
    )
    await uow.turns.add(turn)
    return session, turn


async def test_commit_persists_source_and_outbox_atomically(container, uow_factory) -> None:
    container.tasks.register("noop", Queue.CHAT_FAST, _noop)
    thread = _thread()
    async with uow_factory() as uow:
        session, turn = await _seed_thread(uow, thread)
        seq = await uow.messages.next_sequence(TENANT, thread.thread_id)
        msg = Message(
            thread_id=thread.thread_id,
            session_id=session.session_id,
            turn_id=turn.turn_id,
            tenant_id=TENANT,
            role=MessageRole.USER,
            sequence=seq,
            content="hi",
            content_hash=content_hash("hi"),
            author_principal="user:u1",
        )
        await uow.messages.add(msg)
        outbox_id = await uow.enqueue(
            JobSpec(
                task_name="noop",
                queue=Queue.CHAT_FAST,
                payload={"message_id": msg.message_id},
                tenant_id=TENANT,
            )
        )
        assert outbox_id is not None
        await uow.revisions.bump(TENANT, RevisionKind.THREAD, thread.thread_id)
        await uow.commit()
        assert uow.committed
        assert len(uow.dispatched_job_ids) == 1

    async with uow_factory() as uow:
        stored = await uow.messages.get(TENANT, msg.message_id)
        assert stored is not None and stored.content == "hi" and stored.sequence == 1
        assert await uow.threads.get(TENANT, thread.thread_id) is not None
        # dispatched by the relay -> outbox row marked, job visible in the queue
        pending = await uow.outbox.pending()
        assert pending == []
    job = next(iter(container.tasks.jobs.values()))
    assert job.task_name == "noop" and job.status is JobStatus.PENDING


async def test_exception_rolls_back_everything_including_outbox(container, uow_factory) -> None:
    container.tasks.register("noop", Queue.CHAT_FAST, _noop)
    thread = _thread()
    with pytest.raises(RuntimeError):
        async with uow_factory() as uow:
            await _seed_thread(uow, thread)
            await uow.enqueue(
                JobSpec(task_name="noop", queue=Queue.CHAT_FAST, payload={}, tenant_id=TENANT)
            )
            raise RuntimeError("boom before commit")
    async with uow_factory() as uow:
        assert await uow.threads.get(TENANT, thread.thread_id) is None
        assert await uow.outbox.pending() == []
    assert container.tasks.jobs == {}


async def test_no_commit_means_no_ack_and_no_job(container, uow_factory) -> None:
    container.tasks.register("noop", Queue.CHAT_FAST, _noop)
    thread = _thread()
    async with uow_factory() as uow:
        await _seed_thread(uow, thread)
        await uow.enqueue(
            JobSpec(task_name="noop", queue=Queue.CHAT_FAST, payload={}, tenant_id=TENANT)
        )
        # forgot to commit
    async with uow_factory() as uow:
        assert await uow.threads.get(TENANT, thread.thread_id) is None
    assert container.tasks.jobs == {}


async def test_tenant_scoping_of_reads(container, uow_factory) -> None:
    thread = _thread("tenant-a")
    async with uow_factory() as uow:
        await _seed_thread(uow, thread)
        await uow.commit()
    async with uow_factory() as uow:
        assert await uow.threads.get("tenant-a", thread.thread_id) is not None
        assert await uow.threads.get("tenant-b", thread.thread_id) is None


async def test_message_sequences_are_unique_under_concurrency(container, uow_factory) -> None:
    thread = _thread()
    async with uow_factory() as uow:
        session, turn = await _seed_thread(uow, thread)
        await uow.commit()

    async def write(i: int) -> int:
        async with uow_factory() as uow:
            seq = await uow.messages.next_sequence(TENANT, thread.thread_id)
            await uow.messages.add(
                Message(
                    thread_id=thread.thread_id,
                    session_id=session.session_id,
                    turn_id=turn.turn_id,
                    tenant_id=TENANT,
                    role=MessageRole.USER,
                    sequence=seq,
                    content=f"m{i}",
                    content_hash=content_hash(f"m{i}"),
                    author_principal="user:u1",
                )
            )
            await uow.commit()
            return seq

    seqs = await asyncio.gather(*(write(i) for i in range(12)))
    assert sorted(seqs) == list(range(1, 13))
    async with uow_factory() as uow:
        msgs = await uow.messages.list_thread(TENANT, thread.thread_id, limit=100)
        assert [m.sequence for m in msgs] == list(range(1, 13))


async def test_outbox_sweep_recovers_when_dispatch_failed(container, uow_factory) -> None:
    """Crash/outage between COMMIT and defer: the row stays pending and the sweep dispatches it."""
    container.tasks.register("noop", Queue.CHAT_FAST, _noop)
    container.tasks.fail_enqueue = True
    thread = _thread()
    async with uow_factory() as uow:
        await _seed_thread(uow, thread)
        await uow.enqueue(
            JobSpec(task_name="noop", queue=Queue.CHAT_FAST, payload={"x": 1}, tenant_id=TENANT)
        )
        await uow.commit()
        assert uow.committed and uow.dispatched_job_ids == []  # acknowledged, not yet dispatched
    async with uow_factory() as uow:
        pending = await uow.outbox.pending()
        assert len(pending) == 1 and pending[0].attempts == 1
    container.tasks.fail_enqueue = False
    relay = container.services["outbox_relay"]
    assert await relay.sweep(older_than_seconds=0) == 1
    async with uow_factory() as uow:
        assert await uow.outbox.pending() == []
    assert len(container.tasks.jobs) == 1


async def test_outbox_idempotency_key_deduplicates_jobs(container, uow_factory) -> None:
    container.tasks.register("noop", Queue.ARCHIVE, _noop)
    async with uow_factory() as uow:
        first = await uow.enqueue(
            JobSpec(task_name="noop", queue=Queue.ARCHIVE, idempotency_key="archive:seg1")
        )
        second = await uow.enqueue(
            JobSpec(task_name="noop", queue=Queue.ARCHIVE, idempotency_key="archive:seg1")
        )
        assert first is not None and second is None
        await uow.commit()
    assert len(container.tasks.jobs) == 1


async def test_revisions_bump_and_read(container, uow_factory) -> None:
    async with uow_factory() as uow:
        assert await uow.revisions.bump(TENANT, RevisionKind.THREAD, "thr_x") == 1
        assert await uow.revisions.bump(TENANT, RevisionKind.THREAD, "thr_x") == 2
        assert await uow.revisions.bump(TENANT, RevisionKind.TENANT) == 1
        values = await uow.revisions.get_many(
            TENANT, [(RevisionKind.THREAD, "thr_x"), (RevisionKind.USER, "u1")]
        )
        assert values == {"thread:thr_x": 2, "user:u1": 0}
        await uow.commit()


async def test_observation_roundtrip(container, uow_factory) -> None:
    obs = Observation(
        tenant_id=TENANT,
        kind=ObservationKind.EVENT,
        content="user prefers dark mode",
        content_hash=content_hash("x"),
        principal_id="user:u1",
        user_id="u1",
    )
    async with uow_factory() as uow:
        await uow.observations.add(obs)
        await uow.commit()
    async with uow_factory() as uow:
        got = await uow.observations.get(TENANT, obs.observation_id)
        assert got is not None and got.content == obs.content and got.kind is ObservationKind.EVENT
        stale = await uow.observations.list_unprocessed(
            older_than=datetime.now(UTC) + timedelta(seconds=1)
        )
        assert [o.observation_id for o in stale] == [obs.observation_id]
        await uow.observations.mark_processed(TENANT, obs.observation_id, status="DONE")
        await uow.commit()
    async with uow_factory() as uow:
        assert (
            await uow.observations.list_unprocessed(
                older_than=datetime.now(UTC) + timedelta(seconds=1)
            )
            == []
        )


async def test_readiness_reports_postgres(container) -> None:
    results = await container.readiness()
    assert results["postgres"] == {"ok": True, "mandatory": True}


async def test_statement_timeout_is_applied(container) -> None:
    async with container.database.engine.connect() as conn:
        value = (await conn.execute(text("SHOW statement_timeout"))).scalar_one()
    assert value == "15s"
