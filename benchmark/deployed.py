"""Network-hop gate evidence against a deployed instance: ``durability_network.json`` and
``performance_network.json``.

The same acknowledged stream as ``benchmark.durability`` and the same five budgeted
operations as ``benchmark.performance``, but over TCP against a running API with real
Procrastinate worker processes (real PostgreSQL, Qdrant, cache, blob store), and with real
crashes: every ``--kill-every`` seconds one worker is SIGKILLed mid-run and restarted.
Recovery is the deployed service's own — outbox sweep, job retries, the periodic reconcile
re-queueing the killed worker's jobs, the scheduled archive — the tool only waits for it
(bounded by ``--recovery-timeout``; a timeout is reported, never passed silently) and then
verifies every acknowledgement over the API:

    acknowledged message     -> in the thread listing with its content
    acknowledged observation -> processed (``processed_at`` in PostgreSQL), no duplicate memory
    acknowledged upload      -> document READY and ARCHIVED

    uv run python -m benchmark.deployed --base-url http://memory-api:8080 --api-key dev-key
    uv run python -m benchmark.deployed --api-cmd memory-api      # starts the API itself

``MEMORY__DATABASE__URL`` is required (state reset before each phase, observation and job
status reads). The worker processes are started from this process's environment, which
must point at the same PostgreSQL, Qdrant, cache and blob root as the API. A job orphaned
by a kill is re-queued by the reconcile after ``MEMORY__TASKS__STALLED_AFTER_SECONDS``
(the worker heartbeat is 10 s) at the next ``MEMORY__TASKS__PERIODIC_RECONCILE_SECONDS``
tick, so those two settings bound the recovery time.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from benchmark.common import provenance, write_result
from benchmark.env import BENCH, bench_overrides
from benchmark.evaluation.golden import GoldenSet
from benchmark.harness import (
    FIXTURE_REPORT,
    Acked,
    budgets_ms,
    build_corpus,
    count_duplicate_memories,
    drive_stream,
    headers,
    latency_report,
    measure_budgeted,
    new_scope,
    print_budget_table,
    verify_documents,
    verify_messages,
)
from benchmark.retrieval import GOLDEN, TABLES
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.settings import Settings
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES

ENTRYPOINTS = {"memory-worker": "run_worker", "memory-api": "run_api"}
WORK_JOBS = "task_name NOT LIKE 'periodic.%'"
RECOVERY_CONDITIONS = (
    "jobs_open",
    "observations_unprocessed",
    "documents_pending",
    "messages_unarchived",
)


# ------------------------------------------------------------------------ pure helpers


def resolve_cmd(cmd: str, python: str = sys.executable) -> list[str]:
    """``memory-worker`` / ``memory-api`` resolve to the console script when it is on PATH
    and to the same entrypoint run by ``python`` otherwise (uv-managed environments)."""
    argv = shlex.split(cmd)
    entry = ENTRYPOINTS.get(argv[0]) if argv else None
    if entry is not None and shutil.which(argv[0]) is None:
        code = f"from memory_service.__main__ import {entry}; {entry}()"
        return [python, "-c", code, *argv[1:]]
    return argv


def network_providers(version: dict[str, Any]) -> dict[str, Any]:
    """The ``providers`` block of ``performance_network.json`` from ``GET /version``:
    representative unless the embedding is the hash stand-in or the index is in-memory."""
    providers = dict(version.get("providers") or {})
    embedding = str(providers.get("embedding") or "")
    providers["representative"] = (
        bool(embedding)
        and not embedding.startswith("hash")
        and providers.get("search") not in (None, "memory")
    )
    return providers


def recovery_pending(state: dict[str, int]) -> bool:
    return any(state.get(k, 1) for k in RECOVERY_CONDITIONS)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------- managed processes


class Managed:
    """A subprocess the tool owns: started, SIGKILLed, restarted, stopped."""

    def __init__(self, name: str, argv: list[str], env: dict[str, str], log_dir: Path) -> None:
        self.name = name
        self.argv = argv
        self.env = env
        self.log = log_dir / f"{name}.log"
        self.proc: subprocess.Popen[bytes] | None = None
        self.generation = 0

    def start(self) -> int:
        self.generation += 1
        with self.log.open("ab") as fh:
            fh.write(f"\n--- start generation {self.generation} ---\n".encode())
            self.proc = subprocess.Popen(  # noqa: S603 - operator-supplied command
                self.argv, env=self.env, stdout=fh, stderr=subprocess.STDOUT
            )
        return self.proc.pid

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def kill(self, sig: signal.Signals = signal.SIGKILL, timeout: float = 10) -> int | None:
        if self.proc is None:
            return None
        pid = self.proc.pid
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=timeout)
        return pid

    def stop(self) -> None:
        if self.alive():
            self.kill(signal.SIGTERM, timeout=5)


class WorkerPool:
    def __init__(self, argv: list[str], env: dict[str, str], count: int, log_dir: Path) -> None:
        self.workers = [Managed(f"worker-{i}", argv, env, log_dir) for i in range(count)]
        self.kills: list[dict[str, Any]] = []
        self.unexpected_exits: list[dict[str, Any]] = []

    def start(self) -> list[int]:
        return [w.start() for w in self.workers]

    def check(self) -> None:
        for w in self.workers:
            code = w.exit_code()
            if code is not None:
                self.unexpected_exits.append(
                    {"t": round(time.time(), 3), "worker": w.name, "pid": w.pid, "exit": code}
                )
                w.start()

    def kill_and_restart(self, index: int, in_flight: int) -> dict[str, Any]:
        w = self.workers[index % len(self.workers)]
        pid = w.kill(signal.SIGKILL)
        event = {
            "t": round(time.time(), 3),
            "fault": "worker_kill",
            "worker": w.name,
            "pid": pid,
            "signal": "SIGKILL",
            "jobs_in_flight": in_flight,
            "restarted_pid": w.start(),
        }
        self.kills.append(event)
        return event

    def alive(self) -> int:
        return sum(1 for w in self.workers if w.alive())

    def stop(self) -> None:
        for w in self.workers:
            w.stop()


# --------------------------------------------------------------------------- PostgreSQL


class Db:
    """Direct reads of the canonical store: state reset, recovery progress, job status."""

    def __init__(self, url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(url, pool_size=2, max_overflow=2)

    async def close(self) -> None:
        await self.engine.dispose()

    async def reset(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
            await conn.execute(
                text("TRUNCATE procrastinate_jobs, procrastinate_events RESTART IDENTITY CASCADE")
            )

    async def _count(self, sql: str, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        stmt = text(sql).bindparams(bindparam("ids", expanding=True))
        async with self.engine.connect() as conn:
            return int((await conn.execute(stmt, {"ids": list(ids)})).scalar_one())

    async def _ids(self, sql: str, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        stmt = text(sql).bindparams(bindparam("ids", expanding=True))
        async with self.engine.connect() as conn:
            return {row[0] for row in await conn.execute(stmt, {"ids": list(ids)})}

    async def job_counts(self) -> dict[str, int]:
        sql = f"""
            SELECT
              count(*) FILTER (WHERE status = 'doing') AS doing,
              count(*) FILTER (WHERE status = 'todo'
                               AND (scheduled_at IS NULL OR scheduled_at <= now())) AS runnable,
              count(*) FILTER (WHERE status = 'todo' AND scheduled_at > now()) AS scheduled,
              count(*) FILTER (WHERE status = 'failed') AS failed,
              count(*) FILTER (WHERE status = 'succeeded') AS succeeded,
              coalesce(max(attempts), 0) AS max_attempts
            FROM procrastinate_jobs WHERE {WORK_JOBS}
        """
        async with self.engine.connect() as conn:
            row = (await conn.execute(text(sql))).one()
        return {k: int(v) for k, v in row._mapping.items()}

    async def retried_events(self) -> int:
        sql = f"""
            SELECT count(*) FROM procrastinate_events e JOIN procrastinate_jobs j ON j.id = e.job_id
            WHERE e.type = 'deferred_for_retry' AND j.{WORK_JOBS}
        """
        async with self.engine.connect() as conn:
            return int((await conn.execute(text(sql))).scalar_one())

    async def unprocessed_observations(self, ids: Sequence[str]) -> int:
        return await self._count(
            "SELECT count(*) FROM observations WHERE observation_id IN :ids AND processed_at IS NULL",
            ids,
        )

    async def processed_observation_ids(self, ids: Sequence[str]) -> set[str]:
        return await self._ids(
            "SELECT observation_id FROM observations WHERE observation_id IN :ids AND processed_at IS NOT NULL",
            ids,
        )

    async def documents_pending(self, ids: Sequence[str]) -> int:
        return await self._count(
            "SELECT count(*) FROM documents WHERE document_id IN :ids "
            "AND NOT (status IN ('READY', 'FAILED') AND archive_status = 'ARCHIVED')",
            ids,
        )

    async def documents_ready(self, ids: Sequence[str]) -> int:
        return await self._count(
            "SELECT count(*) FROM documents WHERE document_id IN :ids AND status IN ('READY', 'FAILED')",
            ids,
        )

    async def unarchived_messages(self, ids: Sequence[str]) -> int:
        return await self._count(
            "SELECT count(*) FROM messages WHERE message_id IN :ids AND archive_status <> 'ARCHIVED'",
            ids,
        )


async def reset_backends(settings: Settings) -> dict[str, Any]:
    """Drop the search collections and flush the cache the way the real-component test
    fixtures do; PostgreSQL is truncated separately."""
    container = await build_container(settings, __version__, overrides=bench_overrides())
    dropped: list[str] = []
    flushed = 0
    try:
        if BENCH.search == "qdrant":
            indexer = container.services["indexer"]
            for base in (KNOWLEDGE, MEMORIES):
                name = indexer.collection(base)
                if await container.search.drop_collection(name):
                    dropped.append(name)
            await indexer.ensure_collections()
        if container.cache is not None and bench_overrides().cache is None:
            keys = [key async for key in container.cache.scan("*")]
            if keys:
                flushed = int(await container.cache.delete(*keys))
    finally:
        await container.close()
    return {"search_collections_dropped": dropped, "cache_keys_flushed": flushed}


# ------------------------------------------------------------------------------ waiting


async def wait_ready(client: httpx.AsyncClient, max_seconds: float) -> dict[str, Any]:
    deadline = time.perf_counter() + max_seconds
    last: Any = None
    while time.perf_counter() < deadline:
        try:
            r = await client.get("/health/ready", timeout=5)
            last = r.status_code, r.text[:200]
            if r.status_code == 200:
                return r.json()
        except httpx.TransportError as exc:
            last = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(0.5)
    raise RuntimeError(f"API not ready after {max_seconds}s: {last}")


async def wait_idle(db: Db, max_seconds: float, poll: float = 0.5) -> dict[str, int]:
    """No work job running or runnable (scheduled ones — the 60 s archive — may wait)."""
    deadline = time.perf_counter() + max_seconds
    while True:
        counts = await db.job_counts()
        if counts["doing"] == 0 and counts["runnable"] == 0:
            return counts
        if time.perf_counter() > deadline:
            raise TimeoutError(f"queue not idle after {max_seconds}s: {counts}")
        await asyncio.sleep(poll)


async def wait_documents(db: Db, ids: Sequence[str], max_seconds: float, poll: float = 0.5) -> None:
    deadline = time.perf_counter() + max_seconds
    while await db.documents_ready(ids) < len(ids):
        if time.perf_counter() > deadline:
            raise TimeoutError(f"corpus not parsed after {max_seconds}s")
        await asyncio.sleep(poll)


async def recovery_state(db: Db, acked: Acked) -> dict[str, int]:
    counts = await db.job_counts()
    return {
        "jobs_open": counts["doing"] + counts["runnable"] + counts["scheduled"],
        "jobs_doing": counts["doing"],
        "jobs_scheduled": counts["scheduled"],
        "jobs_failed": counts["failed"],
        "observations_unprocessed": await db.unprocessed_observations(list(acked.observations)),
        "documents_pending": await db.documents_pending(list(acked.uploads)),
        "messages_unarchived": await db.unarchived_messages(list(acked.messages)),
    }


async def wait_recovery(
    db: Db, acked: Acked, max_seconds: float, poll: float = 2.0, idle_polls: int = 15
) -> dict[str, Any]:
    """Wait for the deployed service to honour every acknowledgement on its own. Stops
    early when the queue has been idle for ``idle_polls`` polls while acknowledgements are
    still pending: nothing queued can change that any more (a failed job is a finding)."""
    t0 = time.perf_counter()
    reached: dict[str, float] = {}
    idle = 0
    timed_out = False
    while True:
        state = await recovery_state(db, acked)
        elapsed = round(time.perf_counter() - t0, 2)
        for key in RECOVERY_CONDITIONS:
            if state[key] == 0 and key not in reached:
                reached[key] = elapsed
        if not recovery_pending(state):
            break
        idle = idle + 1 if state["jobs_open"] == 0 else 0
        if idle >= idle_polls:
            break
        if elapsed > max_seconds:
            timed_out = True
            break
        await asyncio.sleep(poll)
    counts = await db.job_counts()
    return {
        "seconds": round(time.perf_counter() - t0, 2),
        "timed_out": timed_out,
        "queue_idle_with_pending": (not timed_out) and recovery_pending(state),
        "reached_after_seconds": reached,
        "final_state": state,
        "jobs": counts,
        "jobs_retried": await db.retried_events(),
    }


# --------------------------------------------------------------------------------- phases


async def in_flight(db: Db, patience: float, poll: float = 0.1) -> int:
    """Jobs in ``doing``; waits up to ``patience`` seconds for at least one so a kill lands
    mid-job whenever the stream keeps the workers busy."""
    deadline = time.perf_counter() + patience
    while True:
        doing = (await db.job_counts())["doing"]
        if doing or time.perf_counter() > deadline:
            return doing
        await asyncio.sleep(poll)


async def chaos_loop(pool: WorkerPool, db: Db, every: float, stop: asyncio.Event) -> None:
    i = 0
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=every)
            return
        except TimeoutError:
            pass
        pool.check()
        pool.kill_and_restart(i, await in_flight(db, every))
        i += 1


async def durability_phase(
    client: httpx.AsyncClient,
    hdrs: dict[str, str],
    db: Db,
    pool: WorkerPool,
    *,
    threads: int,
    messages: int,
    files: int,
    seed: int,
    kill_every: float,
    recovery_timeout: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    scopes = [new_scope() for _ in range(threads)]
    t0 = time.perf_counter()
    stop = asyncio.Event()
    chaos = asyncio.create_task(chaos_loop(pool, db, kill_every, stop))
    try:
        acked = await drive_stream(
            client,
            hdrs,
            scopes,
            messages=messages,
            files=files,
            fixture=FIXTURE_REPORT.read_bytes(),
            rng=rng,
        )
    finally:
        stop.set()
        await chaos
    if not pool.kills:  # a stream shorter than --kill-every still sees one real crash
        pool.kill_and_restart(0, await in_flight(db, kill_every))
    drive_seconds = round(time.perf_counter() - t0, 2)
    recovery = await wait_recovery(db, acked, recovery_timeout)
    pool.check()
    # ---- verification ------------------------------------------------------------
    lost = await verify_messages(client, hdrs, scopes, acked.messages)
    processed = await db.processed_observation_ids(list(acked.observations))
    lost += [
        f"observation {oid} never processed" for oid in acked.observations if oid not in processed
    ]
    lost += await verify_documents(client, hdrs, acked.uploads)
    duplicates = await count_duplicate_memories(client, hdrs, scopes)
    final = recovery["final_state"]
    jobs = recovery["jobs"]
    return {
        "transport": "tcp",
        "acknowledged": acked.counts(),
        "refused_during_outage": acked.refused,
        "unexpected_errors": acked.errors[:20],
        "faults": pool.kills,
        "worker_kills_injected": len(pool.kills),
        "worker_crashes_injected": len(pool.kills),
        "workers": {
            "count": len(pool.workers),
            "alive_at_end": pool.alive(),
            "unexpected_exits": pool.unexpected_exits,
        },
        "recovery": recovery,
        "archive": {
            "messages_unarchived": final["messages_unarchived"],
            "documents_pending": final["documents_pending"],
        },
        "jobs_not_succeeded_after_recovery": jobs["doing"]
        + jobs["runnable"]
        + jobs["scheduled"]
        + jobs["failed"],
        "duplicate_memories": duplicates,
        "lost": lost[:50],
        "acknowledged_data_loss": len(lost),
        "drive_seconds": drive_seconds,
        "recovery_seconds": recovery["seconds"],
        "total_seconds": round(time.perf_counter() - t0, 2),
    }


async def performance_phase(
    client: httpx.AsyncClient,
    hdrs: dict[str, str],
    db: Db,
    settings: Settings,
    *,
    copies: int,
    requests: int,
    settle_timeout: float,
) -> dict[str, Any]:
    golden = GoldenSet.load(GOLDEN)
    queries = [q.query for q in golden.questions]
    scope = new_scope()
    t0 = time.perf_counter()
    document_ids = await build_corpus(client, hdrs, scope, golden, copies)
    await wait_documents(db, document_ids, settle_timeout)
    await wait_idle(db, settle_timeout)
    index_seconds = round(time.perf_counter() - t0, 2)

    async def settle() -> None:
        await wait_idle(db, settle_timeout)

    lat, statuses = await measure_budgeted(
        client, hdrs, scope, queries, FIXTURE_REPORT.read_bytes(), requests, settle=settle
    )
    out = latency_report(lat, statuses, budgets_ms(settings))
    out.update(
        {
            "corpus": {
                "copies": copies,
                "documents": copies * len(golden.documents),
                "index_seconds": index_seconds,
            },
            "transport": "tcp",
        }
    )
    return out


# ------------------------------------------------------------------------------------ run


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    log_dir = (
        Path(args.log_dir) if args.log_dir else Path(tempfile.mkdtemp(prefix="memory-deployed-"))
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    hdrs = headers(args.api_key, args.tenant, args.user)
    base_url = args.base_url
    api: Managed | None = None
    if args.api_cmd:
        port = args.api_port or free_port()
        base_url = f"http://127.0.0.1:{port}"
        api_env = {**env, "MEMORY__SERVICE__HOST": "127.0.0.1", "MEMORY__SERVICE__PORT": str(port)}
        api = Managed("api", resolve_cmd(args.api_cmd), api_env, log_dir)
    pool = WorkerPool(resolve_cmd(args.worker_cmd), env, args.workers, log_dir)
    db = Db(settings.database.url)
    code = 1
    try:
        reset: dict[str, Any] = {}
        if not args.keep_state:
            await db.reset()
            reset = await reset_backends(settings)
        if api is not None:
            api.start()
        pool.start()
        async with httpx.AsyncClient(
            base_url=base_url, timeout=60, limits=httpx.Limits(max_connections=32)
        ) as client:
            ready = await wait_ready(client, args.ready_timeout)
            version = (await client.get("/version")).json()
            server = {
                "base_url": base_url,
                "service": version.get("service"),
                "version": version.get("version"),
                "environment": version.get("environment"),
                "readiness": ready,
                "api_started_by_tool": api is not None,
                "logs": str(log_dir),
            }
            print(
                f"target {base_url} ({version.get('service')} {version.get('version')}), workers={args.workers}"
            )
            durability = await durability_phase(
                client,
                hdrs,
                db,
                pool,
                threads=args.threads,
                messages=args.messages,
                files=args.files,
                seed=args.seed,
                kill_every=args.kill_every,
                recovery_timeout=args.recovery_timeout,
            )
            durability.update(
                {
                    "server": server,
                    "workers": {**durability["workers"], "cmd": args.worker_cmd},
                    "state_reset": reset,
                    "provenance": provenance(),
                }
            )
            path = write_result("durability_network.json", durability)
            print(f"wrote {path}")
            print(
                f"acked={durability['acknowledged']} refused={durability['refused_during_outage']} "
                f"kills={durability['worker_kills_injected']} "
                f"recovery={durability['recovery_seconds']}s"
                f"{' TIMED OUT' if durability['recovery']['timed_out'] else ''} "
                f"loss={durability['acknowledged_data_loss']} "
                f"duplicates={durability['duplicate_memories']} "
                f"errors={len(durability['unexpected_errors'])}"
            )
            if not args.keep_state:
                await db.reset()
                reset = await reset_backends(settings)
            performance = await performance_phase(
                client,
                hdrs,
                db,
                settings,
                copies=args.copies,
                requests=args.requests,
                settle_timeout=args.settle_timeout,
            )
            performance.update(
                {
                    "server": server,
                    "providers": network_providers(version),
                    "state_reset": reset,
                    "provenance": provenance(),
                }
            )
            path = write_result("performance_network.json", performance)
            print(f"wrote {path}")
            print_budget_table(performance)
            code = (
                0
                if durability["acknowledged_data_loss"] == 0
                and not durability["recovery"]["timed_out"]
                and pool.alive() == args.workers
                else 1
            )
    finally:
        pool.stop()
        if api is not None:
            api.stop()
        await db.close()
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=os.environ.get("MEMORY_API_KEY", "dev-key"))
    parser.add_argument("--tenant", default="acme")
    parser.add_argument("--user", default="u1")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--messages", type=int, default=10)
    parser.add_argument("--files", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--copies", type=int, default=5)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2, help="worker processes to start")
    parser.add_argument("--worker-cmd", default="memory-worker")
    parser.add_argument("--kill-every", type=float, default=5.0, help="seconds between SIGKILLs")
    parser.add_argument(
        "--api-cmd", default=None, help="start the API with this command on a free port"
    )
    parser.add_argument("--api-port", type=int, default=0)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--recovery-timeout", type=float, default=900.0)
    parser.add_argument("--settle-timeout", type=float, default=600.0)
    parser.add_argument("--keep-state", action="store_true", help="do not truncate/drop first")
    parser.add_argument("--log-dir", default=None, help="worker/API logs (default: a temp dir)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1 (kills need a worker to kill)")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
