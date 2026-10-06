"""Message-search latency: this conversation only vs this conversation plus the user's earlier
ones, for users with 1..300 conversations (docs/guide/04-retrieval.md#past-conversations).

In-process, so it measures what the service adds, without HTTP. It seeds through the service's
own append path into a database of its own, created and migrated first::

    createdb mem_bench_threads
    MEMORY__DATABASE__URL=postgresql+psycopg://memory:memory@localhost:5432/mem_bench_threads \
        uv run alembic upgrade head
    BENCH_THREADS_DB=postgresql+psycopg://memory:memory@localhost:5432/mem_bench_threads \
        uv run python -m benchmark.bench_message_search

The query's words appear in every message: the worst case for the database filter.
"""

import asyncio
import os
import statistics
import time
from datetime import UTC, datetime, timedelta

from memory_service import __version__
from memory_service.application.container import Overrides, build_container
from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole

DB = os.environ.get(
    "BENCH_THREADS_DB", "postgresql+psycopg://memory:memory@localhost:5432/mem_bench_threads"
)
WORDS = "order invoice refund shipment supplier price quote stock warehouse delivery".split()


def settings() -> Settings:
    return Settings(
        _env_file=None,
        service={"environment": "test", "log_level": "WARNING", "log_json": False},
        authentication={"trusted_dev_api_keys": ["k"]},
        database={"url": DB},
        blob={"provider": "filesystem", "filesystem_root": "/dev/shm/bench_blob"},
    )


def ctx(user, thread):
    return MemoryExecutionContext(
        tenant_id="bench", user_id=user, agent_id="a", agent_run_id="r", thread_id=thread
    )


async def seed(c, user, threads, per_thread):
    conv = c.services["conversation"]
    start = datetime.now(UTC) - timedelta(days=threads)
    for t in range(threads):
        async with c.services["uow_factory"]() as uow:
            for m in range(per_thread):
                w = WORDS[(t + m) % len(WORDS)]
                await conv.append_message(
                    uow,
                    ctx(user, f"{user}-t{t}"),
                    role=MessageRole.USER,
                    content=f"Message {m} about the {w} number {t * 1000 + m} for {user}.",
                    occurred_at=start + timedelta(days=t, minutes=m),
                )
            await uow.commit()


async def timed(fn, n=40):
    for _ in range(5):
        await fn()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        await fn()
        xs.append((time.perf_counter() - t0) * 1000)
    xs.sort()
    return statistics.median(xs), xs[int(len(xs) * 0.95) - 1]


async def _measure(search, user: str, here: str):
    """p50/p95 of a message search in ``here``: without, then with the earlier ones."""

    async def query():
        return await search.search(ctx(user, here), "refund number", kinds=["message"], limit=10)

    owned = search._owned

    async def none(*_a, **_k):
        return []

    search._owned = none  # this conversation only
    only = await timed(query)
    search._owned = owned
    return only, await timed(query)


async def main():
    c = await build_container(
        settings(),
        __version__,
        overrides=Overrides(
            cache="memory",
            search="memory",
            tasks="memory",
            authorization="memory",
            blob="memory",
            embedding="hash",
            embedding_dimension=64,
            nli="lexical",
            document_parser="builtin",
        ),
    )
    search = c.services["search"]
    shapes = [("u1", 1, 50), ("u10", 10, 50), ("u100", 100, 50), ("u300", 300, 50)]
    print(
        f"{'user':6} {'threads':>7} {'msgs':>6} | {'this only p50/p95 ms':>22} | {'this+earlier p50/p95 ms':>25}"
    )
    for user, threads, per in shapes:
        await seed(c, user, threads, per)
        here = f"{user}-t{threads - 1}"
        only, both = await _measure(search, user, here)
        print(
            f"{user:6} {threads:>7} {threads * per:>6} | {only[0]:>9.1f} / {only[1]:>9.1f} | {both[0]:>11.1f} / {both[1]:>9.1f}"
        )
    await c.close()


if __name__ == "__main__":
    asyncio.run(main())
