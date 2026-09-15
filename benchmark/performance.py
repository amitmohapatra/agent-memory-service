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
import json
import time
from typing import Any

import httpx
from sqlalchemy import text

from benchmark.common import provenance, write_result
from benchmark.retrieval import FIXTURES, GOLDEN, TABLES, _pct, _settings
from memory_service.__about__ import __version__
from memory_service.api.app import create_app
from memory_service.application.container import build_container
from memory_service.domain.ids import new_id
from memory_service.modules.evaluation.golden import GoldenSet
from memory_service.modules.jobs.registry import register_handlers

H = {"X-API-Key": "bench", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}


def _stats(xs: list[float]) -> dict[str, Any]:
    return {
        "p50": _pct(xs, 50),
        "p95": _pct(xs, 95),
        "p99": _pct(xs, 99),
        "max": round(max(xs), 2) if xs else 0.0,
        "samples": len(xs),
    }


async def _timed(client: httpx.AsyncClient, method: str, path: str, **kw: Any) -> tuple[float, int]:
    t = time.perf_counter()
    r = await client.request(method, path, headers=H, **kw)
    return (time.perf_counter() - t) * 1000, r.status_code


async def run(copies: int, requests: int) -> dict[str, Any]:
    settings = _settings()
    container = await build_container(settings, __version__)
    register_handlers(container)
    app = create_app(settings, container=container)
    golden = GoldenSet.load(GOLDEN)
    queries = [q.query for q in golden.questions]
    fixture = (FIXTURES / "acme_fy26_annual_report.md").read_bytes()
    lat: dict[str, list[float]] = {k: [] for k in ("chat", "cached", "recall", "context", "file")}
    statuses: dict[str, dict[int, int]] = {k: {} for k in lat}
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://bench", timeout=60
            ) as client,
        ):
            scope = {
                "thread_id": new_id("thread"),
                "session_id": new_id("session"),
                "turn_id": new_id("turn"),
            }
            # corpus: `copies` salted copies of each golden document, indexed synchronously
            t0 = time.perf_counter()
            for alias, filename in golden.documents.items():
                data = (FIXTURES / filename).read_bytes()
                for n in range(copies):
                    salt = b"" if n == 0 else f"\n\n<!-- perf copy {n} -->\n".encode()
                    r = await client.post(
                        "/v1/files",
                        headers=H,
                        files={"file": (filename, data + salt, "text/markdown")},
                        data={"scope": json.dumps(scope), "title": f"{alias}-{n}"},
                    )
                    assert r.status_code == 202, r.text
            await container.tasks.drain()
            await container.tasks.drain()
            index_seconds = round(time.perf_counter() - t0, 2)
            # 1. chat accept
            for i in range(requests):
                ms, code = await _timed(
                    client,
                    "POST",
                    "/v1/messages",
                    json={
                        "scope": scope,
                        "role": "USER",
                        "content": f"Turn {i}: {queries[i % len(queries)]}",
                    },
                )
                lat["chat"].append(ms)
                statuses["chat"][code] = statuses["chat"].get(code, 0) + 1
                if i % 10 == 9:
                    await container.tasks.drain()
            await container.tasks.drain()
            await container.tasks.drain()
            # 2. recall
            for i in range(requests):
                ms, code = await _timed(
                    client,
                    "POST",
                    "/v1/recall",
                    json={"scope": scope, "query": queries[i % len(queries)]},
                )
                lat["recall"].append(ms)
                statuses["recall"][code] = statuses["recall"].get(code, 0) + 1
            # 3. context bundle, uncached (a unique suffix defeats the bundle cache)
            for i in range(requests):
                ms, code = await _timed(
                    client,
                    "POST",
                    "/v1/context",
                    json={"scope": scope, "query": f"{queries[i % len(queries)]} (variant {i})"},
                )
                lat["context"].append(ms)
                statuses["context"][code] = statuses["context"].get(code, 0) + 1
            # 4. cached context: the same query twice, the second one is measured
            for i in range(requests):
                body = {"scope": scope, "query": queries[i % len(queries)]}
                await client.post("/v1/context", headers=H, json=body)
                ms, code = await _timed(client, "POST", "/v1/context", json=body)
                lat["cached"].append(ms)
                statuses["cached"][code] = statuses["cached"].get(code, 0) + 1
            # 5. file accept (distinct bytes each time so nothing is deduplicated)
            for i in range(requests):
                salt = f"\n\n<!-- accept {i} -->\n".encode()
                ms, code = await _timed(
                    client,
                    "POST",
                    "/v1/files",
                    files={"file": (f"accept_{i}.md", fixture + salt, "text/markdown")},
                    data={"scope": json.dumps(scope), "title": f"accept {i}"},
                )
                lat["file"].append(ms)
                statuses["file"][code] = statuses["file"].get(code, 0) + 1
            await container.tasks.drain()
        budgets = settings.budgets
        out: dict[str, Any] = {
            "chat_accept_p95_ms": _pct(lat["chat"], 95),
            "cached_context_p95_ms": _pct(lat["cached"], 95),
            "recall_p95_ms": _pct(lat["recall"], 95),
            "context_bundle_p95_ms": _pct(lat["context"], 95),
            "file_accept_p95_ms": _pct(lat["file"], 95),
            "budgets_ms": {
                "chat_accept_p95_ms": budgets.chat_accept_p95_ms,
                "cached_context_p95_ms": budgets.cached_context_p95_ms,
                "recall_p95_ms": budgets.recall_p95_ms,
                "context_bundle_p95_ms": budgets.context_bundle_p95_ms,
                "file_accept_p95_ms": budgets.file_accept_p95_ms,
            },
            "detail": {k: {**_stats(v), "status_codes": statuses[k]} for k, v in lat.items()},
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
        out["within_budget"] = all(out[k] <= v for k, v in out["budgets_ms"].items())
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
    for key, budget in payload["budgets_ms"].items():
        flag = "ok" if payload[key] <= budget else "OVER"
        print(f"{key:24} {payload[key]:8.2f} ms  (budget {budget})  {flag}")


if __name__ == "__main__":
    main()
