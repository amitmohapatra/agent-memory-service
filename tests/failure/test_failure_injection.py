"""Failure-injection scenarios behind the release gate (``failure_injection.json``):

    worker_kill     a worker process dies mid-job (SIGKILL) -> the job is re-queued and the
                    acknowledged observation is processed exactly once
    cache_flush     the cache is flushed / unavailable mid-flow -> identical results, no
                    duplicate side effects, idempotency still honoured
    blob_outage     the archive store is down -> acknowledged messages stay readable and
                    STAGED; archiving resumes when it returns
    search_rebuild  the search index is lost -> rebuilt from PostgreSQL with the same hits
    authz_denial    the authorization provider is down or says no -> fail closed, no data

Every scenario runs against real PostgreSQL (and real Procrastinate for the kill). The
scenario name is the test-name prefix; ``benchmark/failure_injection.py`` records the
outcome per scenario, never a hand-written "pass".
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ArchiveStatus, JobStatus, MessageRole, ObservationKind
from memory_service.domain.errors import DependencyUnavailable, ScopeDenied
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers
from memory_service.ports.tasks import Queue
from memory_service.tools.reindex import rebuild_search_index
from tests.integration.conftest import DB_URL, TABLES, integration_settings, requires_pg

pytestmark = [pytest.mark.failure, requires_pg]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")


def _ctx(**extra) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme",
        user_id="u1",
        workspace_id="ws1",
        thread_id=new_id("thread"),
        session_id=new_id("session"),
        turn_id=new_id("turn"),
        **extra,
    )


async def _say(container, ctx, content: str, role=MessageRole.USER):
    async with container.services["uow_factory"]() as uow:
        res = await container.services["conversation"].append_message(
            uow, ctx, role=role, content=content
        )
        await uow.commit()
    return res


async def _memories(container, ctx):
    async with container.services["uow_factory"]() as uow:
        return await container.services["memory"].list_memories(uow, ctx)


# -- worker_kill ------------------------------------------------------------------


async def test_worker_kill_requeues_the_job_and_processes_once(make_settings, tmp_path) -> None:
    settings = integration_settings(
        make_settings,
        tasks={"provider": "procrastinate"},
        blob={"provider": "filesystem", "filesystem_root": str(tmp_path / "blob")},
    )
    container = await build_container(settings, __version__)
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
            await conn.execute(
                text("TRUNCATE procrastinate_jobs, procrastinate_events RESTART IDENTITY CASCADE")
            )
        register_handlers(container)
        ctx = _ctx()
        async with container.services["uow_factory"]() as uow:
            ack = await container.services["memory"].submit_observation(
                uow, ctx, kind=ObservationKind.EVENT, content="My timezone is Europe/Berlin."
            )
            await uow.commit()
        assert ack.job_ids  # durably queued (outbox -> procrastinate) before the ack
        from memory_service.adapters.db.orm import OutboxRow

        async with container.database.session() as session:
            row = await session.get(OutboxRow, int(ack.job_ids[0][4:]))
        assert row is not None and row.job_id is not None
        job_id = row.job_id
        # a worker takes the job and is killed -9 while it is `doing`
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        proc = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                str(Path(__file__).with_name("_slow_worker.py")),
                DB_URL.replace("postgresql+psycopg://", "postgresql://"),
            ],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            line = await asyncio.wait_for(asyncio.to_thread(proc.stdout.readline), timeout=60)  # type: ignore[union-attr]
            assert line.startswith("TOOK")
        finally:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)
        # the job is orphaned in `doing`; nothing has been processed
        job = await container.tasks.get(job_id)
        assert job is not None and job.status is JobStatus.RUNNING
        assert await _memories(container, ctx) == []
        # the periodic reconcile re-queues stalled jobs; a healthy worker finishes it
        await container.tasks.recover_stalled(seconds_since_heartbeat=0)
        job = await container.tasks.get(job_id)
        assert job is not None and job.status is JobStatus.PENDING
        # only the work queues: periodic maintenance (reconcile/archive) is not under test
        work = [Queue.CHAT_FAST, Queue.EMBEDDING, Queue.GRAPH, Queue.MEMORY_EXTRACT]
        await asyncio.wait_for(container.tasks.run_until_idle(work, concurrency=2), 60)
        await asyncio.wait_for(container.tasks.run_until_idle(work, concurrency=2), 60)
        mems = await _memories(container, ctx)
        assert [m.predicate for m in mems] == ["timezone"]
        # a second replay (e.g. two recoveries racing) is idempotent: processed_at guards it
        await container.tasks.recover_stalled(seconds_since_heartbeat=0)
        await asyncio.wait_for(container.tasks.run_until_idle(work, concurrency=2), 60)
        assert len(await _memories(container, ctx)) == 1
        async with container.services["uow_factory"]() as uow:
            obs = await uow.observations.get("acme", ack.observation_id)
        assert obs is not None and obs.processed_at is not None
    finally:
        await container.close()


# -- cache_flush ------------------------------------------------------------------


async def test_cache_flush_mid_flow_changes_nothing(container, uow_factory) -> None:
    register_handlers(container)
    ctx = _ctx()
    await _say(container, ctx, "I prefer concise answers and my timezone is Europe/Berlin.")
    await container.tasks.drain()
    await container.tasks.drain()
    builder = container.services["context_builder"]
    first = await builder.build(ctx, "what is my timezone?")
    cached = await builder.build(ctx, "what is my timezone?")
    assert cached.cache_hit and cached.render() == first.render()
    # flush: every cached scope, bundle and idempotency reservation is gone
    container.cache._data.clear()
    container.cache._lists.clear()
    after = await builder.build(ctx, "what is my timezone?")
    assert not after.cache_hit and after.render() == first.render()
    assert [m.item_id for m in after.memories] == [m.item_id for m in first.memories]
    # outage (not just flush) while appending: the write still lands exactly once
    container.cache.available = False
    try:
        res = await _say(container, ctx, "Also, I work at ACME Corp.")
        assert res.ack.message_id
        async with uow_factory() as uow:
            msgs = await container.services["conversation"].list_messages(uow, ctx, ctx.thread_id)
        assert len(msgs) == 2
        # the same idempotent request replays from PostgreSQL, not from the cache
        again = await _say(container, ctx, "Also, I work at ACME Corp.")
        assert again.ack.message_id  # accepted, no exception
    finally:
        container.cache.available = True
    async with uow_factory() as uow:
        msgs = await container.services["conversation"].list_messages(uow, ctx, ctx.thread_id)
    assert [m.content for m in msgs][:2] == [
        "I prefer concise answers and my timezone is Europe/Berlin.",
        "Also, I work at ACME Corp.",
    ]
    await container.tasks.drain()
    await container.tasks.drain()
    assert {m.predicate for m in await _memories(container, ctx)} >= {"timezone", "works_at"}


# -- blob_outage ------------------------------------------------------------------


async def test_blob_outage_keeps_acknowledged_messages_and_recovers(container, uow_factory) -> None:
    register_handlers(container)
    ctx = _ctx()
    archive = container.services["archive_service"]
    container.blob.available = False
    try:
        acks = [
            await _say(container, ctx, f"message {i} " + "lorem ipsum " * 400) for i in range(3)
        ]
        assert all(a.ack.message_id for a in acks)  # acknowledged: durable in PostgreSQL
        with pytest.raises(DependencyUnavailable):
            await archive.archive_thread("acme", ctx.thread_id)
        async with uow_factory() as uow:
            msgs = await container.services["conversation"].list_messages(uow, ctx, ctx.thread_id)
        assert len(msgs) == 3 and all(m.archive_status is ArchiveStatus.STAGED for m in msgs)
        assert all(m.content.startswith("message") for m in msgs)  # readable during the outage
    finally:
        container.blob.available = True
    segments = await archive.archive_thread("acme", ctx.thread_id)
    assert len(segments) == 1
    async with uow_factory() as uow:
        msgs = await container.services["conversation"].list_messages(uow, ctx, ctx.thread_id)
        stored = [await uow.messages.get("acme", m.message_id) for m in msgs]
    assert {m.archive_status for m in stored if m} == {ArchiveStatus.ARCHIVED}
    assert (await archive.reconcile())["verify_mismatch"] == 0


# -- search_rebuild ---------------------------------------------------------------


async def test_search_rebuild_from_postgres_restores_identical_hits(container, uow_factory) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["authz"].grant_membership(
            "acme", "u1", workspaces=["ws1"], revisions=uow.revisions
        )
        await container.services["ingestion"].accept_file(
            uow,
            U1,
            filename="acme_fy26_annual_report.md",
            media_type="text/markdown",
            data=FIXTURE.read_bytes(),
            title="ACME FY26",
        )
        await uow.commit()
    ctx = _ctx()
    await _say(container, ctx, "My favourite editor is neovim.")
    await container.tasks.drain()
    await container.tasks.drain()
    engine = container.services["retrieval"]
    q = "Why did Adjusted EBITDA increase despite lower revenue?"
    before = await engine.retrieve(U1, q)
    before_ids = [c.record_id for c in before.candidates]
    mem_before = {
        c.record_id
        for c in (await engine.retrieve(ctx, "favourite editor", kinds=("memory",))).candidates
    }
    assert before_ids and mem_before
    # the index is lost
    indexer = container.services["indexer"]
    for base in ("knowledge", "memories"):
        assert await container.search.drop_collection(indexer.collection(base))
    engine._ensured.clear()
    degraded = await engine.retrieve(U1, q)  # degraded (graph evidence only), not failing
    assert not [c for c in degraded.candidates if c.kind == "chunk" and c.expansion_edge is None]
    # rebuild from the canonical store
    report = await rebuild_search_index(container, drop=True)
    assert report.ok and report.documents == 1 and report.chunks > 0 and report.memories >= 1
    after = await engine.retrieve(U1, q)
    assert [c.record_id for c in after.candidates] == before_ids
    assert after.diagnostics["evidence"]["status"] == before.diagnostics["evidence"]["status"]
    assert {
        c.record_id
        for c in (await engine.retrieve(ctx, "favourite editor", kinds=("memory",))).candidates
    } == mem_before


# -- authz_denial -----------------------------------------------------------------


async def test_authz_denial_fails_closed(container, uow_factory) -> None:
    register_handlers(container)
    owner = _ctx()
    await _say(container, owner, "My timezone is Europe/Berlin.")
    await container.tasks.drain()
    await container.tasks.drain()
    engine = container.services["retrieval"]
    assert (await engine.retrieve(owner, "timezone", kinds=("memory",))).candidates
    # 1. a plain denial: another user is refused the thread and sees nothing
    stranger = owner.model_copy(update={"user_id": "u2"})
    with pytest.raises(ScopeDenied):
        async with uow_factory() as uow:
            await container.services["conversation"].list_messages(uow, stranger, owner.thread_id)
    assert (await engine.retrieve(stranger, "timezone", kinds=("memory",))).candidates == []
    # 2. the authorization provider is down: no cached scope -> the request fails, never
    #    falls back to "allow"
    container.cache._data.clear()
    container.services["authz"].provider.available = False
    try:
        fresh = owner.model_copy(update={"user_id": "u3"})
        with pytest.raises(DependencyUnavailable):
            await engine.retrieve(fresh, "timezone", kinds=("memory",))
        with pytest.raises((DependencyUnavailable, ScopeDenied)):
            async with uow_factory() as uow:
                await container.services["conversation"].list_messages(uow, fresh, owner.thread_id)
    finally:
        container.services["authz"].provider.available = True
    # 3. back: the owner reads again, the stranger still cannot
    assert (await engine.retrieve(owner, "timezone", kinds=("memory",))).candidates
    assert (await engine.retrieve(stranger, "timezone", kinds=("memory",))).candidates == []


async def test_cache_flush_over_the_api_never_hides_acknowledged_messages(app, client) -> None:
    """Regression: a message acknowledged while the cache was down must appear in the
    listing once the cache is back (the hot list is validated against the thread revision),
    and concurrent first messages of a new thread never race on its creation."""
    import asyncio

    import httpx

    H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
    container = app.state.container
    scope = {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:

        async def say(i: int) -> int:
            r = await c.post(
                "/v1/messages", headers=H, json={"scope": scope, "role": "USER", "content": f"m{i}"}
            )
            return r.status_code

        # 1. concurrent creation of the same new thread
        assert set(await asyncio.gather(*[say(i) for i in range(8)])) == {202}
        listing = await c.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H)
        assert listing.status_code == 200 and len(listing.json()["messages"]) == 8
        # 2. cache outage in the middle of the conversation, then flush
        container.cache.available = False
        try:
            assert await say(8) == 202
        finally:
            container.cache.available = True
        assert await say(9) == 202
        got = [
            m["content"]
            for m in (await c.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H)).json()[
                "messages"
            ]
        ]
        assert got[-2:] == ["m8", "m9"] and set(got[:8]) == {f"m{i}" for i in range(8)}
        container.cache._data.clear()
        container.cache._lists.clear()
        got = [
            m["content"]
            for m in (
                await c.get(
                    f"/v1/threads/{scope['thread_id']}/messages", headers=H, params={"limit": 4}
                )
            ).json()["messages"]
        ]
        assert len(got) == 4 and got[-2:] == ["m8", "m9"]
        # 3. a message acknowledged during an outage that hit the *first* message of a thread
        scope2 = {
            "thread_id": new_id("thread"),
            "session_id": new_id("session"),
            "turn_id": new_id("turn"),
        }
        container.cache.available = False
        try:
            r = await c.post(
                "/v1/messages",
                headers=H,
                json={"scope": scope2, "role": "USER", "content": "first"},
            )
            assert r.status_code == 202
        finally:
            container.cache.available = True
        r = await c.post(
            "/v1/messages",
            headers=H,
            json={"scope": scope2, "role": "ASSISTANT", "content": "second"},
        )
        assert r.status_code == 202
        got = [
            m["content"]
            for m in (await c.get(f"/v1/threads/{scope2['thread_id']}/messages", headers=H)).json()[
                "messages"
            ]
        ]
        assert got == ["first", "second"]
