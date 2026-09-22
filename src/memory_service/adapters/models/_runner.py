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
semaphore of one. The executor is what makes "two encodes are never inside the model at
once" true; the semaphore is what makes the waiting *visible* — a caller queues in asyncio,
where it can be cancelled and counted, rather than in the executor's unbounded backlog.

The semaphore is rebuilt when the running loop changes. A benchmark harness that calls
``asyncio.run`` twice around one container would otherwise meet "is bound to a different
event loop" on its second run, which is a fact about the primitive rather than about the
model.

A cancelled caller keeps its permit until the model is out. ``run_in_executor`` cannot take
back a job the thread has already started, so releasing the gate on cancellation — a client
disconnect, a request timeout — would let the next caller in while the previous encode is
still inside the model, which is the one thing this class exists to prevent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class SerialRunner:
    """A single-thread executor plus a one-permit gate, shared by every model adapter."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._gate: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _semaphore(self, loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
        if self._gate is None or self._loop is not loop:
            self._gate = asyncio.Semaphore(1)
            self._loop = loop
        return self._gate

    async def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        loop = asyncio.get_running_loop()
        async with self._semaphore(loop):
            running = loop.run_in_executor(self._executor, partial(fn, *args, **kwargs))
            try:
                return await asyncio.shield(running)
            finally:
                # Cancelling the caller does not cancel a started job. Hold the permit
                # until the thread is out, or "one caller at a time" ends at the first
                # disconnect.
                if not running.done():
                    await asyncio.gather(running, return_exceptions=True)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
