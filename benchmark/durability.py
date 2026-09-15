"""Acknowledged-data-loss gate: ``durability.json``.

Drives the real HTTP API (in-process ASGI, real PostgreSQL, filesystem blob) with
concurrent chat messages, observations and file uploads while injecting faults in the
middle of the run — cache outage, blob-store outage, task-queue outage (enqueue fails,
outbox holds the job), worker crashes (every job fails on its first attempt) — then, after
recovery (outbox sweep, job retries, archive), checks every acknowledgement:

    acknowledged message     -> row in PostgreSQL, content readable, archived + verified
    acknowledged observation -> processed exactly once (no duplicate memories)
    acknowledged upload      -> document READY (or FAILED with a recorded reason), never lost

``acknowledged_data_loss`` is the number of acknowledgements the service cannot honour
after recovery. Requests refused during an outage (503/429) are not acknowledgements and
are counted separately — the contract is "what we acked, we keep", not "we never refuse".

    uv run python -m benchmark.durability --threads 20 --messages 10 --files 6
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import text

from benchmark.common import provenance, write_result
from benchmark.harness import (
    FACTS,
    FIXTURE_REPORT,
    H,
    count_duplicate_memories,
    drive_stream,
    new_scope,
    verify_documents,
    verify_messages,
)
from benchmark.retrieval import TABLES, _settings
from memory_service.__about__ import __version__
from memory_service.api.app import create_app
from memory_service.application.container import build_container
from memory_service.domain.enums import JobStatus
from memory_service.modules.jobs.registry import register_handlers

__all__ = ["FACTS", "Chaos", "H", "main", "run"]


class Chaos:
    """Fault injection on the in-process container, toggled from the driver."""

    def __init__(self, container: Any) -> None:
        self.container = container
        self.events: list[dict[str, Any]] = []
        self.crash_next: set[str] = set()
        queue = container.tasks
        original = queue._run

        async def crashing_run(job_id: str) -> None:
            spec = queue.payloads[job_id]
            if self.crash_all_first_attempts and job_id not in self.seen_jobs:
                self.seen_jobs.add(job_id)
                info = queue.jobs[job_id]
                queue.jobs[job_id] = info.model_copy(
                    update={"status": JobStatus.FAILED, "attempts": info.attempts + 1}
                )
                self.crashed += 1
                self.crash_next.add(spec.task_name)
                return
            await original(job_id)

        queue._run = crashing_run
        self.crash_all_first_attempts = False
        self.seen_jobs: set[str] = set()
        self.crashed = 0

    def mark(self, what: str, on: bool) -> None:
        self.events.append({"t": round(time.time(), 3), "fault": what, "on": on})

    def cache(self, on: bool) -> None:
        self.container.cache.available = not on
        self.mark("cache_outage", on)

    def blob(self, on: bool) -> None:
        self.container.blob.available = not on
        self.mark("blob_outage", on)

    def queue(self, on: bool) -> None:
        self.container.tasks.fail_enqueue = on
        self.mark("queue_outage", on)

    def workers(self, on: bool) -> None:
        self.crash_all_first_attempts = on
        self.mark("worker_crashes", on)

    async def recover(self) -> dict[str, int]:
        """What operations does after an incident: sweep the outbox, retry failed jobs,
        drain the queue, run the archive."""
        c = self.container
        self.cache(False)
        self.blob(False)
        self.queue(False)
        swept = await c.services["outbox_relay"].sweep(older_than_seconds=0, limit=10_000)
        # the first pass runs with workers crashing on every first attempt
        self.workers(True)
        await c.tasks.drain(max_rounds=20)
        self.workers(False)
        retried = 0
        for _ in range(6):
            failed = [j for j, info in c.tasks.jobs.items() if info.status is JobStatus.FAILED]
            for job_id in failed:
                c.tasks.jobs[job_id] = c.tasks.jobs[job_id].model_copy(
                    update={"status": JobStatus.PENDING}
                )
                retried += 1
            await c.tasks.drain(max_rounds=20)
            if not failed:
                break
        return {"outbox_swept": swept, "jobs_retried": retried}


async def run(threads: int, messages: int, files: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    settings = _settings()
    root = Path(".bench_blob")
    data = settings.model_dump()
    data["blob"] = {"provider": "filesystem", "filesystem_root": str(root)}
    settings = type(settings)(**data)
    container = await build_container(settings, __version__)
    register_handlers(container)
    chaos = Chaos(container)
    app = create_app(settings, container=container)
    t0 = time.perf_counter()
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://bench", timeout=30
            ) as client,
        ):
            scopes = [new_scope() for _ in range(threads)]
            schedule = {  # fault windows as fractions of the message stream
                "cache": (0.15, 0.35),
                "queue": (0.30, 0.45),
                "workers": (0.40, 0.70),
                "blob": (0.55, 0.85),
            }

            async def step(sent: int, total: int) -> None:
                frac = sent / total
                for fault, (start, end) in schedule.items():
                    on = start <= frac < end
                    current = {
                        "cache": not container.cache.available,
                        "queue": container.tasks.fail_enqueue,
                        "workers": chaos.crash_all_first_attempts,
                        "blob": not container.blob.available,
                    }[fault]
                    if on != current:
                        getattr(chaos, fault)(on)

            acked = await drive_stream(
                client,
                H,
                scopes,
                messages=messages,
                files=files,
                fixture=FIXTURE_REPORT.read_bytes(),
                rng=rng,
                on_step=step,
            )
            drive_seconds = round(time.perf_counter() - t0, 2)
            recovery = await chaos.recover()
            # archive everything and purge staged payloads: content must remain readable
            archive = container.services["archive_service"]
            archived = 0
            for scope in scopes:
                archived += len(await archive.archive_thread("acme", scope["thread_id"]))
            report = await archive.reconcile()
            # ---- verification ----------------------------------------------------
            lost = await verify_messages(client, H, scopes, acked.messages)
            uow_factory = container.services["uow_factory"]
            async with uow_factory() as uow:
                for oid in acked.observations:
                    obs = await uow.observations.get("acme", oid)
                    if obs is None:
                        lost.append(f"observation {oid}")
                    elif obs.processed_at is None:
                        lost.append(f"observation {oid} never processed")
            lost += await verify_documents(client, H, acked.uploads)
            duplicates = await count_duplicate_memories(client, H, scopes)
            pending = [
                j
                for j, info in container.tasks.jobs.items()
                if info.status is not JobStatus.SUCCEEDED
            ]
            return {
                "acknowledged": acked.counts(),
                "refused_during_outage": acked.refused,
                "unexpected_errors": acked.errors[:20],
                "faults": chaos.events,
                "worker_crashes_injected": chaos.crashed,
                "recovery": recovery,
                "archive": {"segments": archived, **report},
                "jobs_not_succeeded_after_recovery": len(pending),
                "duplicate_memories": duplicates,
                "lost": lost[:50],
                "acknowledged_data_loss": len(lost),
                "drive_seconds": drive_seconds,
                "total_seconds": round(time.perf_counter() - t0, 2),
            }
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--messages", type=int, default=10)
    parser.add_argument("--files", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    payload = asyncio.run(run(args.threads, args.messages, args.files, args.seed))
    payload["provenance"] = provenance()
    path = write_result("durability.json", payload)
    print(f"wrote {path}")
    print(
        f"acked={payload['acknowledged']} refused={payload['refused_during_outage']} "
        f"crashes={payload['worker_crashes_injected']} loss={payload['acknowledged_data_loss']} "
        f"duplicates={payload['duplicate_memories']} errors={len(payload['unexpected_errors'])}"
    )


if __name__ == "__main__":
    main()
