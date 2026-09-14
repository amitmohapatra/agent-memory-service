import pytest

from memory_service.adapters.cache.memory_cache import MemoryCache
from memory_service.adapters.tasks.inline_queue import InlineTaskQueue, RecordingTaskQueue
from memory_service.domain.enums import JobStatus
from memory_service.ports.cache import CacheUnavailable
from memory_service.ports.tasks import JobSpec, Queue


async def test_memory_cache_basic_ops_and_outage() -> None:
    c = MemoryCache()
    await c.set("a", b"1", ttl_seconds=60)
    assert await c.get("a") == b"1"
    assert await c.set_if_absent("a", b"2") is False
    assert await c.set_if_absent("b", b"2") is True
    assert await c.incr("n") == 1 and await c.incr("n", amount=5) == 6
    assert await c.list_push("l", b"x", b"y", b"z", max_len=2) == 2
    assert await c.list_range("l") == [b"y", b"z"]
    assert await c.mget(["a", "missing"]) == [b"1", None]
    assert await c.delete("a", "b") == 2
    c.available = False
    with pytest.raises(CacheUnavailable):
        await c.get("a")
    assert await c.ping() is False


async def test_recording_queue_dedups_and_drains() -> None:
    q = RecordingTaskQueue()
    ran: list[dict] = []

    async def h(payload):
        ran.append(payload)

    q.register("t", Queue.CHAT_FAST, h)
    a = await q.enqueue(
        JobSpec(task_name="t", queue=Queue.CHAT_FAST, payload={"i": 1}, idempotency_key="k")
    )
    b = await q.enqueue(
        JobSpec(task_name="t", queue=Queue.CHAT_FAST, payload={"i": 2}, idempotency_key="k")
    )
    assert b == "dedup:k"
    assert (await q.get(a)).status is JobStatus.PENDING
    assert await q.drain() == 1
    assert ran == [{"i": 1}] and (await q.get(a)).status is JobStatus.SUCCEEDED
    with pytest.raises(KeyError):
        await q.enqueue(JobSpec(task_name="unknown", queue=Queue.CHAT_FAST))


async def test_inline_queue_runs_immediately_and_records_failures() -> None:
    q = InlineTaskQueue()

    async def bad(payload):
        raise ValueError("nope")

    q.register("bad", Queue.GRAPH, bad)
    job_id = await q.enqueue(JobSpec(task_name="bad", queue=Queue.GRAPH))
    info = await q.get(job_id)
    assert info.status is JobStatus.FAILED and "ValueError" in (info.last_error or "")
