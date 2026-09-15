"""p95 latency gate evidence: ``performance.json``.

Measures the five budgeted operations through the real HTTP stack (in-process ASGI —
no network hop — real PostgreSQL, the configured cache/search/blob providers) on a
corpus of ``--copies`` fixture documents plus a chat history:

    chat_accept_p95_ms      POST /v1/messages       (durable ack; jobs run asynchronously)
    cached_context_p95_ms   POST /v1/context        (same query again -> bundle cache hit)
    recall_p95_ms           POST /v1/recall         (hybrid + graph + verification)
    context_bundle_p95_ms   POST /v1/context        (uncached: retrieval + assembly)
    file_accept_p95_ms      POST /v1/files          (stage + checksum + durable ack)

Every number carries the environment it was measured in (``providers`` and
``representative``): with the hash embedding and local Qdrant the figures bound the
service's own overhead, not a production deployment. ``benchmark/load/locustfile.py`` is
the network-level load test for a deployed instance.

    uv run python -m benchmark.performance --copies 5 --requests 60
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

import httpx
from sqlalchemy import text

from benchmark.common import provenance, write_result
from benchmark.harness import (
    FIXTURE_REPORT,
    H,
    budgets_ms,
    build_corpus,
    latency_report,
    measure_budgeted,
    new_scope,
    print_budget_table,
)
from benchmark.retrieval import GOLDEN, TABLES, _settings
from memory_service.__about__ import __version__
from memory_service.api.app import create_app
from memory_service.application.container import build_container
from memory_service.modules.evaluation.golden import GoldenSet
from memory_service.modules.jobs.registry import register_handlers

__all__ = ["H", "main", "run"]


async def run(copies: int, requests: int) -> dict[str, Any]:
    settings = _settings()
    container = await build_container(settings, __version__)
    register_handlers(container)
    app = create_app(settings, container=container)
    golden = GoldenSet.load(GOLDEN)
    queries = [q.query for q in golden.questions]
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://bench", timeout=60
            ) as client,
        ):
            scope = new_scope()
            # corpus: `copies` salted copies of each golden document, indexed synchronously
            t0 = time.perf_counter()
            await build_corpus(client, H, scope, golden, copies)
            await container.tasks.drain()
            await container.tasks.drain()
            index_seconds = round(time.perf_counter() - t0, 2)
            lat, statuses = await measure_budgeted(
                client,
                H,
                scope,
                queries,
                FIXTURE_REPORT.read_bytes(),
                requests,
                settle=container.tasks.drain,
            )
        out = latency_report(lat, statuses, budgets_ms(settings))
        out.update(
            {
                "corpus": {
                    "copies": copies,
                    "documents": copies * len(golden.documents),
                    "index_seconds": index_seconds,
                },
                "transport": "in-process ASGI (no network hop)",
                "providers": {
                    "embedding": container.embedding.fingerprint(),
                    "reranker": type(container.reranker).__name__ if container.reranker else None,
                    "search": settings.search.provider,
                    "cache": settings.cache.provider,
                    "blob": settings.blob.provider,
                    "tasks": settings.tasks.provider,
                    "representative": not container.embedding.fingerprint().startswith("hash-")
                    and settings.search.provider != "memory",
                },
            }
        )
        return out
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=5)
    parser.add_argument("--requests", type=int, default=60)
    args = parser.parse_args()
    payload = asyncio.run(run(args.copies, args.requests))
    payload["provenance"] = provenance()
    path = write_result("performance.json", payload)
    print(f"wrote {path}")
    print_budget_table(payload)


if __name__ == "__main__":
    main()
