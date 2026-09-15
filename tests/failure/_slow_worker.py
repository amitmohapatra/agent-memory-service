"""A Procrastinate worker that takes the next chat-fast job and never finishes it.

Spawned by the failure-injection suite and killed with SIGKILL while the job is ``doing``
— the closest thing to a worker process dying mid-job (OOM kill, node loss)."""

from __future__ import annotations

import asyncio
import sys

from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue
from memory_service.modules.jobs.registry import TASK_PROCESS_OBSERVATION
from memory_service.ports.tasks import Queue


async def main(dsn: str) -> None:
    queue = ProcrastinateTaskQueue(dsn, default_retries=5, job_timeout_seconds=600)

    async def hang(payload: dict) -> None:
        print("TOOK", payload.get("observation_id"), flush=True)
        await asyncio.sleep(3600)

    queue.register(TASK_PROCESS_OBSERVATION, Queue.CHAT_FAST, hang)
    await queue.run_worker([Queue.CHAT_FAST], concurrency=1, wait=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
