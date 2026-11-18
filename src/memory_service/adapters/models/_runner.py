"""How an in-process model is entered: one thread, one caller at a time.

``asyncio.to_thread`` hands work to the event loop's *default* executor, which is sized
``min(32, cpu_count + 4)`` — twelve threads on an eight-core box. Every concurrent request
therefore puts another encode inside a model that is itself fanning its GEMMs over every
core, and the box spends its time in the scheduler instead of in the matrix multiply. That
is the oversubscription this repository has been measuring around for months — measuring
*around*, because nothing has yet timed two callers at once. The case for this class is the
arithmetic plus a queue that is visible and bounded; the number that would settle it is a
concurrent measurement on the 8 vCPU VM, and it does not exist yet.

So each model owns a ``ThreadPoolExecutor(max_workers=1)`` and is entered through a
one-permit gate. The executor is what makes "two encodes are never inside the model at
once" true; the gate is what makes the waiting *visible* — a caller queues in asyncio,
where it can be cancelled and counted, rather than in the executor's unbounded backlog.

The gate is rebuilt when the running loop changes. A benchmark harness that calls
``asyncio.run`` twice around one container would otherwise meet "is bound to a different
event loop" on its second run, which is a fact about the primitive rather than about the
model.

The queue in front of the gate is bounded (``constants.OVERLOAD.model_queue_max_waiters``,
installed by ``configure_runners``). Unbounded, an overloaded process accepted every request
and made each one wait behind all the others: latency grew without limit and every caller
timed out together. Past the bound a caller fails at once with ``DependencyUnavailable``
(a retryable 503), and ``memory_model_queue_waiters`` / ``memory_model_queue_rejected_total``
show the queue.

A cancelled caller's permit goes back only when the model is out. ``run_in_executor``
cannot take back a job the thread has already started, so releasing the gate on cancellation
— a client disconnect, a request timeout — would let the next caller in while the previous
encode is still inside the model, which is the one thing this class exists to prevent. The
*caller* does not wait for it, though: it is cancelled at once and the job's completion
releases the permit. It used to wait, and a request deadline (``api/deadline.py``) was then
answered only when the model was done: a 15 s verify deadline answered at 55 s under load.

Requests go first. One model serves the requests and the jobs of a process (the verify
path and the statement labeller share the NLI head; a query's encode and the indexing jobs
the encoder), and a FIFO queue put a verify behind every message job queued before it: under
a chat's write load it waited out its 15 s deadline behind work nobody was waiting for. The
gate keeps two FIFO queues and hands the model to a waiting request before a waiting job
(``domain.work.background`` marks a job). A job still waits behind requests only while some
are waiting, and the bounded queue keeps that short.

``run_each`` runs a caller's batches one after another under one permit, and stops at the
first batch boundary after a cancellation: a forty-claim verify that timed out no longer
scores its remaining batches for a caller that is gone while the requests behind it wait.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, ClassVar, ParamSpec, TypeVar

from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.work import in_background
from memory_service.observability.metrics import model_queue_rejected_total, model_queue_waiters

P = ParamSpec("P")
R = TypeVar("R")


class _Gate:
    """One permit, two FIFO queues: a waiting request is handed the permit before a waiting
    background job. A cancelled waiter leaves its queue; one handed the permit as it was
    cancelled passes it on."""

    def __init__(self) -> None:
        self._held = False
        self._requests: deque[asyncio.Future[None]] = deque()
        self._jobs: deque[asyncio.Future[None]] = deque()

    def locked(self) -> bool:
        return self._held

    async def acquire(self, *, job: bool) -> None:
        if not self._held and not self._requests and not self._jobs:
            self._held = True
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        queue = self._jobs if job else self._requests
        queue.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                self.release()  # handed the permit as it was cancelled: pass it on
            elif waiter in queue:  # a release may have skipped it already
                queue.remove(waiter)
            raise

    def release(self) -> None:
        for queue in (self._requests, self._jobs):
            while queue:
                waiter = queue.popleft()
                if not waiter.done():
                    waiter.set_result(None)  # the permit moves to it; still held
                    return
        self._held = False


def configure_runners(*, max_waiters: int | None) -> None:
    """Bound the queue in front of every model of this process (``None``: unbounded)."""
    SerialRunner.max_waiters = max_waiters


class SerialRunner:
    """A single-thread executor plus a one-permit gate, shared by every model adapter."""

    #: callers allowed to queue for the gate, process-wide (``configure_runners``)
    max_waiters: ClassVar[int | None] = None

    def __init__(self, name: str) -> None:
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._gate: _Gate | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._waiting = 0

    @property
    def waiting(self) -> int:
        """Callers queued for the gate right now."""
        return self._waiting

    def _gate_for(self, loop: asyncio.AbstractEventLoop) -> _Gate:
        if self._gate is None or self._loop is not loop:
            self._gate = _Gate()
            self._loop = loop
        return self._gate

    async def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        gate = await self._enter()
        released = False
        try:
            running = self._start(partial(fn, *args, **kwargs))
            released = self._release_after(running, gate)
            return await asyncio.shield(running)
        finally:
            if not released:
                gate.release()

    async def run_each[T, U](self, fn: Callable[[T], U], batches: Sequence[T]) -> list[U]:
        """``fn`` over each of ``batches`` in order, under one permit: the caller's batches
        are not interleaved with another caller's, and a cancellation stops them at the next
        batch (the one inside the model finishes; its completion releases the permit)."""
        if not batches:
            return []
        gate = await self._enter()
        released = False
        out: list[U] = []
        try:
            for batch in batches:
                running = self._start(partial(fn, batch))
                try:
                    out.append(await asyncio.shield(running))
                except asyncio.CancelledError:
                    released = self._release_after(running, gate)
                    raise
            return out
        finally:
            if not released:
                gate.release()

    async def _enter(self) -> _Gate:
        """The permit, after the bounded queue (``max_waiters``) in front of it."""
        loop = asyncio.get_running_loop()
        gate = self._gate_for(loop)
        limit = self.max_waiters
        if gate.locked() and limit is not None and self._waiting >= limit:
            model_queue_rejected_total.labels(self.name).inc()
            raise DependencyUnavailable(
                f"the {self.name} model is saturated ({self._waiting} callers queued); "
                "retry shortly"
            )
        self._waiting += 1
        model_queue_waiters.labels(self.name).inc()
        try:
            await gate.acquire(job=in_background())
        finally:
            self._waiting -= 1
            model_queue_waiters.labels(self.name).dec()
        return gate

    def _start[T](self, job: Callable[[], T]) -> asyncio.Future[T]:
        return asyncio.get_running_loop().run_in_executor(self._executor, job)

    @staticmethod
    def _release_after(running: asyncio.Future[Any], gate: _Gate) -> bool:
        """When ``running`` is still inside the model, hand the permit to its completion and
        say so (True): a cancelled caller leaves at once, the next one enters only when the
        thread is out. A finished job leaves the permit with the caller (False)."""
        if running.done():
            return False

        def done(future: asyncio.Future[Any]) -> None:
            if not future.cancelled():
                future.exception()  # retrieved: nobody awaits an abandoned job
            gate.release()

        running.add_done_callback(done)
        return True

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
