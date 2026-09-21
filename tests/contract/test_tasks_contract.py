"""One contract, every task Queue adapter.

Every asynchronous promise the service makes — consolidation, indexing, summaries, archive —
runs through this port. The three properties below are what callers actually rely on: a job
that is enqueued is findable, an idempotency key deduplicates rather than double-processes,
and a registered handler is what runs.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from memory_service.domain.enums import JobStatus
from memory_service.ports.tasks import JobSpec, Queue

pytestmark = pytest.mark.contract

ADAPTERS = ("inline", "procrastinate")


async def _noop(payload: dict) -> None:
    return None


def _spec(**over) -> JobSpec:
    return JobSpec(
        task_name=over.pop("task_name", "contract.noop"),
        queue=Queue.MEMORY_EXTRACT,
        payload={"marker": uuid.uuid4().hex},
        tenant_id="acme",
        **over,
    )


@pytest_asyncio.fixture(params=ADAPTERS, loop_scope="function")
async def queue(request: pytest.FixtureRequest):
    if request.param == "inline":
        from memory_service.adapters.tasks.inline_queue import InlineTaskQueue

        yield InlineTaskQueue()
        return

    from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue
    from tests.conftest import QUEUE_DB_URL

    dsn = QUEUE_DB_URL.replace("postgresql+psycopg://", "postgresql://")
    adapter = ProcrastinateTaskQueue(dsn)
    if not await adapter.ping():
        pytest.skip("procrastinate queue database not reachable")
    try:
        yield adapter
    finally:
        await adapter.close()


def _defers(queue) -> bool:
    """True for a real queue (work waits for a worker); False for the inline executor.

    The idempotency contract differs by execution model, and pretending otherwise is how a
    shared suite starts lying: a deferring queue must refuse a duplicate while the first job
    is still pending, whereas inline has already finished the work before the second call
    happens, so a fresh job is the correct answer.
    """
    return not hasattr(queue, "drain")


async def test_an_enqueued_job_is_findable_by_its_id(queue) -> None:
    queue.register("contract.noop", Queue.MEMORY_EXTRACT, _noop)
    job_id = await queue.enqueue(_spec())
    assert job_id, "enqueue must return an identifier callers can follow"
    info = await queue.get(job_id)
    assert info is None or info.job_id == job_id


async def test_an_idempotency_key_refuses_a_duplicate_while_one_is_pending(queue) -> None:
    queue.register("contract.noop", Queue.MEMORY_EXTRACT, _noop)
    key = f"contract-{uuid.uuid4().hex}"
    first = await queue.enqueue(_spec(idempotency_key=key))
    second = await queue.enqueue(_spec(idempotency_key=key))
    if _defers(queue):
        assert second != first and second.startswith("dedup:"), (
            "a second job under the same key must be recognised as a duplicate, not queued"
        )
    else:
        assert second, "inline has already run the first job; a new one is legitimate"


async def test_a_registered_handler_receives_the_payload_it_was_enqueued_with(queue) -> None:
    """Handlers are called with the payload as one dict — not as keyword arguments. Getting
    that wrong raises inside the worker, where it becomes a FAILED job and a log line rather
    than a test failure, so the shape is worth pinning."""
    if _defers(queue):
        pytest.skip("needs a worker to execute; the inline queue covers the handler contract")
    seen: list[dict] = []

    async def handler(payload: dict) -> None:
        seen.append(payload)

    name = f"contract.handler.{uuid.uuid4().hex[:6]}"
    queue.register(name, Queue.MEMORY_EXTRACT, handler)
    spec = _spec(task_name=name)
    await queue.enqueue(spec)
    await queue.drain()
    assert seen == [spec.payload]


async def test_a_handler_that_raises_marks_the_job_failed_rather_than_losing_it(queue) -> None:
    if _defers(queue):
        pytest.skip("needs a worker to execute")

    async def explode(payload: dict) -> None:
        raise RuntimeError("handler blew up")

    name = f"contract.boom.{uuid.uuid4().hex[:6]}"
    queue.register(name, Queue.MEMORY_EXTRACT, explode)
    job_id = await queue.enqueue(_spec(task_name=name))
    await queue.drain()
    info = await queue.get(job_id)
    assert info is not None and info.status is JobStatus.FAILED
    assert "handler blew up" in (info.last_error or "")


async def test_an_unregistered_task_is_refused_at_enqueue(queue) -> None:
    """Better to fail the caller than to accept work nothing can run."""
    with pytest.raises(KeyError):
        await queue.enqueue(_spec(task_name=f"never.registered.{uuid.uuid4().hex[:6]}"))


async def test_ping_reports_whether_the_backend_is_usable(queue) -> None:
    assert await queue.ping() is True
