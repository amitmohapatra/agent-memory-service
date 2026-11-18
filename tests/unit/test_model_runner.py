"""The gate every in-process model is entered through.

What is under test is not speed but exclusion: one thread, one caller inside at a time, and
a gate that is released whatever the model does.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from memory_service.adapters.models._runner import SerialRunner

pytestmark = pytest.mark.unit


class Occupancy:
    """Counts how many callers are inside the body at the same moment."""

    def __init__(self) -> None:
        self.peak = 0
        self.inside = 0
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, seconds: float = 0.02) -> int:
        with self._lock:
            self.inside += 1
            self.calls += 1
            self.peak = max(self.peak, self.inside)
        time.sleep(seconds)
        with self._lock:
            self.inside -= 1
        return self.calls


async def test_callers_are_never_inside_the_model_at_once() -> None:
    runner = SerialRunner("test")
    body = Occupancy()
    try:
        await asyncio.gather(*(runner.run(body) for _ in range(6)))
    finally:
        runner.close()
    assert body.calls == 6
    assert body.peak == 1, "the default executor's twelve threads are what this replaces"


async def test_the_gate_is_released_when_the_model_raises() -> None:
    runner = SerialRunner("test")

    def boom() -> None:
        raise ValueError("no")

    try:
        for _ in range(3):
            with pytest.raises(ValueError, match="no"):
                await runner.run(boom)
        assert await runner.run(lambda: "after") == "after"
    finally:
        runner.close()


async def test_arguments_reach_the_model_unchanged() -> None:
    runner = SerialRunner("test")
    try:
        assert await runner.run(lambda a, b=0: a + b, 2, b=3) == 5
    finally:
        runner.close()


def test_a_second_event_loop_gets_a_second_semaphore() -> None:
    """A benchmark harness calls ``asyncio.run`` once per arm around one container. A
    semaphore built in the first loop and awaited in the second raises, which would be a
    fact about the primitive rather than about the encoder."""
    runner = SerialRunner("test")
    try:
        assert asyncio.run(runner.run(lambda: 1)) == 1
        assert asyncio.run(runner.run(lambda: 2)) == 2
    finally:
        runner.close()


async def test_a_cancelled_caller_keeps_the_gate_until_the_model_is_out() -> None:
    """A client disconnect cancels the awaiting coroutine, but the executor thread is
    already inside the model and cannot be recalled. If the permit went back on
    cancellation, the next caller would enter while the first encode was still running."""
    runner = SerialRunner("test")
    body = Occupancy()
    try:
        first = asyncio.ensure_future(runner.run(body, 0.2))
        await asyncio.sleep(0.05)  # long enough to be inside the body
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await runner.run(body, 0.0) == 2
    finally:
        runner.close()
    assert body.peak == 1


async def test_a_cancelled_caller_is_answered_at_once_while_the_model_stays_exclusive() -> None:
    """A request past its deadline is cancelled while its encode is inside the model. It used
    to wait there for the thread to come out (a 15 s verify deadline answered at 55 s under
    load); the 504 goes out now, and the permit still goes back only when the thread is out,
    so the next caller never enters beside the abandoned job. The model is held until the
    test lets it go, so nothing here depends on how long a sleep takes."""
    runner = SerialRunner("test")
    inside, release = threading.Event(), threading.Event()
    entered: list[str] = []

    def held() -> None:
        entered.append("held")
        inside.set()
        release.wait(10)

    try:
        first = asyncio.ensure_future(runner.run(held))
        await asyncio.to_thread(inside.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            # gstack-timing-ok: a ceiling for a hang; the model is still held here
            await asyncio.wait_for(first, 5)
        assert not release.is_set(), "the caller left while the model was still busy"
        second = asyncio.ensure_future(runner.run(entered.append, "second"))
        await asyncio.sleep(0.05)
        assert entered == ["held"], "the next caller entered beside the abandoned job"
        release.set()
        await second
        assert entered == ["held", "second"]
    finally:
        release.set()
        runner.close()


async def test_batches_run_under_one_permit_in_order() -> None:
    runner = SerialRunner("test")
    try:
        assert await runner.run_each(lambda batch: [x * 2 for x in batch], [[1, 2], [3]]) == [
            [2, 4],
            [6],
        ]
        assert await runner.run_each(lambda batch: batch, []) == []
    finally:
        runner.close()


async def test_a_cancelled_batched_caller_stops_at_the_next_batch() -> None:
    """A verify of forty claims is many model batches. Cancelled, it stops after the batch
    inside the model instead of scoring the rest for a caller that is gone, so the requests
    queued behind it get the model back."""
    runner = SerialRunner("test")
    inside, release = threading.Event(), threading.Event()
    scored: list[int] = []

    def batch(n: int) -> int:
        scored.append(n)
        inside.set()
        release.wait(10)
        return n

    try:
        many = asyncio.ensure_future(runner.run_each(batch, list(range(20))))
        await asyncio.to_thread(inside.wait, 5)
        many.cancel()
        with pytest.raises(asyncio.CancelledError):
            await many
        release.set()
        assert await runner.run(lambda: "after") == "after"
        assert scored == [0], "only the batch that was inside the model"
    finally:
        release.set()
        runner.close()


async def test_a_request_waiting_for_the_model_goes_before_background_jobs() -> None:
    """The verify path and the statement labeller share one NLI model (and a query's encode
    and the indexing jobs one encoder). FIFO put a request behind every message job queued
    before it, so under a chat's write load a verify waited out its deadline behind work no
    caller was waiting for. A request now enters before any background job still waiting;
    the job behind it keeps its turn among the jobs."""
    from memory_service.domain.work import background

    runner = SerialRunner("test")
    release = threading.Event()
    order: list[str] = []

    async def job(name: str) -> None:
        with background():
            await runner.run(order.append, name)

    try:
        inside = asyncio.ensure_future(runner.run(release.wait, 5))
        await asyncio.sleep(0.05)
        jobs = [asyncio.ensure_future(job(f"job{n}")) for n in range(3)]
        await asyncio.sleep(0.02)
        request = asyncio.ensure_future(runner.run(order.append, "request"))
        await asyncio.sleep(0.02)
        release.set()
        await asyncio.gather(inside, request, *jobs)
    finally:
        release.set()
        runner.close()
    assert order == ["request", "job0", "job1", "job2"]


async def test_a_cancelled_waiter_leaves_the_queue_and_the_gate_moves_on() -> None:
    from memory_service.domain.work import background

    runner = SerialRunner("test")
    release = threading.Event()
    try:
        inside = asyncio.ensure_future(runner.run(release.wait, 5))
        await asyncio.sleep(0.05)

        async def job() -> str:
            with background():
                return await runner.run(lambda: "job")

        gone = asyncio.ensure_future(runner.run(lambda: "gone"))
        waiting = asyncio.ensure_future(job())
        await asyncio.sleep(0.02)
        assert runner.waiting == 2
        gone.cancel()
        await asyncio.sleep(0)
        assert runner.waiting == 1
        release.set()
        assert await waiting == "job"
        assert await inside is True
        assert await runner.run(lambda: "after") == "after"
        assert runner.waiting == 0
    finally:
        release.set()
        runner.close()


async def test_a_waiter_cancelled_as_the_permit_is_released_leaves_cleanly() -> None:
    """A request's deadline fires while it waits, and the holder releases before the
    cancelled waiter runs again: the release skips it, and its own clean-up must not trip
    over the queue it is no longer in (it raised ``ValueError`` in place of the
    cancellation: a 500 where the deadline's 504 belonged)."""
    from memory_service.adapters.models._runner import _Gate

    gate = _Gate()
    await gate.acquire(job=False)
    waiting = asyncio.ensure_future(gate.acquire(job=False))
    await asyncio.sleep(0)
    waiting.cancel()
    gate.release()  # before the cancelled waiter has run
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not gate.locked()
    await gate.acquire(job=True)
    assert gate.locked()
