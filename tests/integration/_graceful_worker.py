"""A Procrastinate worker that runs one slow job and is stopped with SIGTERM mid-job.

Spawned by tests/integration/test_procrastinate_queue.py: the production worker installs
the signal handlers and a shutdown grace (``memory_service.worker``), and this is that call
around a job whose length the test chooses."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue
from memory_service.ports.tasks import Queue


async def main(dsn: str, seconds: float, grace: float, marker: str) -> None:
    queue = ProcrastinateTaskQueue(dsn, default_retries=2, job_timeout_seconds=600)

    async def slow(payload: dict) -> None:
        print("TOOK", flush=True)
        await asyncio.sleep(seconds)
        Path(marker).write_text("finished")

    queue.register("test.slow", Queue.CHAT_FAST, slow)
    await queue.run_worker(
        [Queue.CHAT_FAST],
        concurrency=1,
        wait=True,
        install_signal_handlers=True,
        shutdown_grace_seconds=grace,
    )
    print("STOPPED", flush=True)
    await queue.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]))
