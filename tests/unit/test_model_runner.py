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
