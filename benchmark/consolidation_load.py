"""Consolidation under load, with a stub model (no paid call): job lag, the context's p95
before and after beliefs, database growth and worker throughput (design note §8).

Two tenants receive the same synthetic corpus - ``--users`` users in pairs, each pair one
conversation (a workspace, TENANT visibility, as LoCoMo is written) of ``--chats`` chats of
``--turns`` turns, alternating speakers who name each other now and then - through the
product path (``submit_observation`` -> ``memory.process_observation`` on the real
Procrastinate worker, its pools sized as the deployment sizes them). One tenant's policy
allows no consolidation (before), the other's allows ``reflection`` (after): every new fact
queues ``memory.consolidate``, debounced as in production, and the stub answers it after
``--model-seconds`` (keep for a repeated topic, add otherwise; every claim SUPPORTED).

Reported: fact written -> held by a belief (p50/p95), the jobs and the facts per job (the
coalescing ratio), consolidation jobs per second, ``/v1/context`` build p50/p95 for the same
readers and questions in both tenants (in-process, no network), and the size of the belief
tables per 1k users. Stand-ins: the hash encoder and the in-memory search store, so the
latencies compare the two tenants, not production.

    uv run python -m benchmark.consolidation_load --users 40 --chats 3 --turns 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import respx
from sqlalchemy import text

from benchmark.common import dedicated_database, submit_observation
from memory_service.__about__ import __version__
from memory_service.application.container import Overrides, build_container, worker_pools_for
from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.memory import consolidation

BASE = "http://stub-model.invalid/v1"
TOPICS = ["hiking", "painting", "cooking", "chess", "running", "guitar", "gardening", "travel"]
QUESTIONS = ["What does {} like to do?", "Where has {} been lately?", "What is {} planning?"]


class Stub:
    """The model: consolidation keeps a fact whose topic a belief already has, else adds it;
    the citation check calls every claim SUPPORTED. Answers after ``seconds``."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        await asyncio.sleep(self.seconds)
        body = json.loads(request.content)
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        lines = [ln for ln in user.splitlines() if ln[:1] in "BFL" and ln[1:2].isdigit()]
        if system.startswith(consolidation.SYSTEM[:40]):
            beliefs = {
                t: ln.split(" ", 1)[0] for ln in lines if ln[0] == "B" for t in TOPICS if t in ln
            }
            out: dict[str, Any] = {
                "entries": [self._entry(ln, beliefs) for ln in lines if ln[0] == "F"]
            }
        elif system.startswith(consolidation.CHECK_SYSTEM[:40]):
            out = {"verdicts": [{"line": ln.split(":")[0], "label": "SUPPORTED"} for ln in lines]}
        else:
            out = {"rewrites": []}
        content = json.dumps(out)
        usage = {"prompt_tokens": len(user) // 4, "completion_tokens": len(content) // 4}
        return httpx.Response(
            200,
            json={"model": "stub", "choices": [{"message": {"content": content}}], "usage": usage},
        )

    @staticmethod
    def _entry(line: str, beliefs: dict[str, str]) -> dict[str, Any]:
        fid, rest = line.split(" ", 1)
        topic = next((t for t in TOPICS if t in rest), "")
        if topic in beliefs:
            return {
                "op": "keep",
                "facts": [fid],
                "belief": beliefs[topic],
                "text": "",
                "facet": "",
                "items": [],
            }
        beliefs[topic] = "B0"  # one add per topic per call
        said = rest.split(": ", 1)[-1][:200]
        add = {"op": "add", "facts": [fid], "belief": "", "text": said, "facet": "hobbies"}
        return {**add, "items": []}


def corpus(
    users: int, chats: int, turns: int, seed: int = 7
) -> list[tuple[str, str, str, datetime]]:
    """(workspace, speaker, text, said at) for every turn, in order."""
    rnd = random.Random(seed)
    out = []
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for pair in range(users // 2):
        a, b = f"u{pair}a", f"u{pair}b"
        for chat in range(chats):
            day = start + timedelta(days=7 * chat)
            for turn in range(turns):
                speaker, other = (a, b) if turn % 2 == 0 else (b, a)
                topic = rnd.choice(TOPICS)
                said = f"I went {topic} with my friends on day {chat * turns + turn}."
                if rnd.random() < 0.3:
                    said = f"{other.capitalize()} told me about {topic} yesterday."
                out.append((f"ws{pair}", speaker, said, day + timedelta(minutes=turn)))
    return out


async def ingest(container: Any, tenant: str, turns: list[tuple[str, str, str, datetime]]) -> float:
    uow_factory = container.services["uow_factory"]
    started = time.perf_counter()
    for workspace, speaker, said, when in turns:
        ctx = MemoryExecutionContext(tenant_id=tenant, workspace_id=workspace, user_id=speaker)
        async with uow_factory() as uow:
            await submit_observation(
                uow,
                ctx,
                content=said,
                hints=ProcessingHints(visibility=Visibility.TENANT),
                occurred_at=when,
            )
            await uow.commit()
    return time.perf_counter() - started


async def settled(container: Any, *, poll: float = 2.0, limit: float = 3600.0) -> float:
    """Wait until no job is waiting or running and no person waits for consolidation."""
    started = time.perf_counter()
    engine = container.database.engine
    while time.perf_counter() - started < limit:
        async with engine.connect() as conn:
            busy = await conn.scalar(
                text("SELECT count(*) FROM procrastinate_jobs WHERE status IN ('todo', 'doing')")
            )
            waiting = await conn.scalar(
                text("SELECT count(*) FROM belief_progress WHERE pending_since IS NOT NULL")
            )
            outbox = await conn.scalar(
                text("SELECT count(*) FROM job_outbox WHERE dispatched_at IS NULL AND NOT dead")
            )
        if not (busy or waiting or outbox):
            return time.perf_counter() - started
        await asyncio.sleep(poll)
    raise TimeoutError("the queue did not settle")


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1) if ordered else 0.0


async def context_latency(container: Any, tenant: str, users: int, n: int) -> list[float]:
    builder = container.services["context_builder"]
    rnd = random.Random(11)
    out = []
    for i in range(n):
        pair = rnd.randrange(users // 2)
        reader = MemoryExecutionContext(
            tenant_id=tenant, workspace_id=f"ws{pair}", user_id="reader"
        )
        question = rnd.choice(QUESTIONS).format(f"U{pair}a") + f" ({i})"  # unique: no cache hit
        started = time.perf_counter()
        await builder.build(reader, question)
        out.append((time.perf_counter() - started) * 1000)
    return out


async def measure(args: argparse.Namespace) -> dict[str, Any]:
    # the stub endpoint, as the service's own (``LLM_*``): every call it makes lands on ``Stub``
    os.environ.update({"LLM_BASE_URL": BASE, "LLM_API_KEY": "stub"})
    settings = Settings()
    dedicated_database(settings.database.url.get_secret_value())  # it truncates the tables
    overrides = Overrides(
        cache="memory",
        search="memory",
        authorization="memory",
        blob="memory",
        embedding="hash",
        embedding_dimension=64,
        nli="lexical",
        document_parser="builtin",
        endpoint_probe="accept",
    )
    stub = Stub(args.model_seconds)
    turns = corpus(args.users, args.chats, args.turns)
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{BASE}/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "stub"}]})
        )
        mock.post(f"{BASE}/chat/completions").mock(side_effect=stub)
        container = await build_container(settings, __version__, overrides=overrides)
        try:
            return await _run(container, args, stub, turns, settings)
        finally:
            await container.close()


async def _ingested(container: Any, stub: Stub, turns: list, settings: Any) -> dict[str, Any]:
    """Both tenants written through the real worker, and its timings."""
    engine = container.database.engine
    async with engine.begin() as conn:
        await conn.execute(text("SET LOCAL statement_timeout = 0"))
        await conn.execute(
            text(
                "TRUNCATE belief_sources, beliefs, belief_progress, memories, observations, "
                "job_outbox, revisions, llm_policies CASCADE"
            )
        )
        await conn.execute(
            text("TRUNCATE procrastinate_jobs, procrastinate_events RESTART IDENTITY CASCADE")
        )
    policies = container.services["model_policies"]
    async with container.services["uow_factory"]() as uow:
        await policies.set(uow, "load_off", uses=[], read_assist=False, models={})
        await policies.set(uow, "load_on", uses=["reflection"], read_assist=False, models={})
        await uow.commit()
    worker = asyncio.create_task(container.tasks.run_pools(worker_pools_for(settings)))
    try:
        off_ingest = await ingest(container, "load_off", turns)
        await settled(container)
        calls_before = stub.calls
        on_started = time.perf_counter()
        on_ingest = await ingest(container, "load_on", turns)
        drained = await settled(container)
        on_wall = time.perf_counter() - on_started
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    return {
        "ingest_seconds": {"off": round(off_ingest, 1), "on": round(on_ingest, 1)},
        "model_calls": stub.calls - calls_before,
        "drain_after_ingest_seconds": round(drained, 1),
        "on_wall_seconds": round(on_wall, 1),
    }


async def _run(
    container: Any, args: argparse.Namespace, stub: Stub, turns: list, settings: Any
) -> dict[str, Any]:
    register_handlers(container)
    engine = container.database.engine
    phase = {} if args.measure_only else await _ingested(container, stub, turns, settings)
    async with engine.connect() as conn:
        lag = [
            float(r[0])
            for r in await conn.execute(
                text(
                    "SELECT extract(epoch FROM min(b.created_at) - m.created_at) FROM belief_sources s "
                    "JOIN beliefs b ON b.belief_id = s.belief_id JOIN memories m ON m.memory_id = s.memory_id "
                    "WHERE b.tenant_id = 'load_on' GROUP BY m.memory_id, m.created_at"
                )
            )
        ]
        jobs = await conn.scalar(
            text(
                "SELECT count(*) FROM procrastinate_jobs WHERE task_name = 'memory.consolidate' "
                "AND status = 'succeeded'"
            )
        )
        facts = await conn.scalar(
            text("SELECT count(*) FROM memories WHERE tenant_id = 'load_on' AND deleted_at IS NULL")
        )
        beliefs = await conn.scalar(
            text(
                "SELECT count(*) FROM beliefs WHERE tenant_id = 'load_on' AND deleted_at IS NULL "
                "AND superseded_by IS NULL AND valid_to IS NULL"
            )
        )
        # each consolidation job's run time (started -> succeeded), for the worker's capacity
        took = [
            float(r[0])
            for r in await conn.execute(
                text(
                    "SELECT extract(epoch FROM max(e.at) FILTER (WHERE e.type = 'succeeded') "
                    "- max(e.at) FILTER (WHERE e.type = 'started')) FROM procrastinate_events e "
                    "JOIN procrastinate_jobs j ON j.id = e.job_id "
                    "WHERE j.task_name = 'memory.consolidate' GROUP BY e.job_id"
                )
            )
            if r[0] is not None
        ]
        sizes = {
            t: int(await conn.scalar(text(f"SELECT pg_total_relation_size('{t}')")))
            for t in ("beliefs", "belief_sources", "belief_progress", "memories")
        }
    await context_latency(container, "load_off", args.users, 20)  # warm-up, discarded
    off = await context_latency(container, "load_off", args.users, args.queries)
    on = await context_latency(container, "load_on", args.users, args.queries)
    per_k = 1000 / args.users
    return {
        "users": args.users,
        "chats": args.chats,
        "turns": args.turns,
        "turns_total": len(turns),
        "model_seconds": args.model_seconds,
        "debounce_seconds": consolidation.DEBOUNCE_SECONDS,
        "worker_pools": [[[q.value for q in qs], n] for qs, n in worker_pools_for(settings)],
        **phase,
        "fact_to_belief_seconds": {
            "p50": pct(lag, 0.5),
            "p95": pct(lag, 0.95),
            "max": pct(lag, 1.0),
            "n": len(lag),
        },
        "consolidation_jobs": jobs,
        "facts": facts,
        "facts_per_job": round(facts / max(jobs, 1), 1),
        "current_beliefs": beliefs,
        "job_seconds": {"p50": pct(took, 0.5), "p95": pct(took, 0.95)},
        # what one worker sustains: its model-waiting (graph) slots over a job's run time
        "jobs_per_second_per_worker_capacity": round(
            max(n for qs, n in worker_pools_for(settings) if [q.value for q in qs] == ["graph"])
            / max(statistics.fmean(took), 1e-6),
            2,
        )
        if took
        else 0.0,
        "context_ms": {
            "before": {
                "p50": pct(off, 0.5),
                "p95": pct(off, 0.95),
                "mean": round(statistics.fmean(off), 1),
            },
            "after": {
                "p50": pct(on, 0.5),
                "p95": pct(on, 0.95),
                "mean": round(statistics.fmean(on), 1),
            },
        },
        "belief_tables_mb_per_1k_users": round(
            sum(sizes[t] for t in ("beliefs", "belief_sources", "belief_progress")) * per_k / 2**20,
            2,
        ),
        "memories_mb_per_1k_users_both_tenants": round(sizes["memories"] * per_k / 2**20, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--chats", type=int, default=3)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--model-seconds", type=float, default=1.0)
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="measure an earlier run's database (the same --users) without writing again",
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(measure(args)), indent=1))  # noqa: T201


if __name__ == "__main__":
    main()
