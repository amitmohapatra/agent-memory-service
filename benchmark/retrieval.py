"""Retrieval benchmark: index a synthetic corpus derived from the fixtures, then measure
recall / context-bundle latency (p50/p95) and the golden-set quality metrics with whatever
providers the environment configures.

    uv run python -m benchmark.retrieval --copies 40 --queries 30

Provider selection comes from the normal settings (env ``MEMORY__MODELS__EMBEDDING__PROVIDER``
etc.), so the same script benchmarks the hash stand-in, Granite via sentence-transformers,
or fastembed/ONNX. The result records the provider fingerprints and whether they are
representative. PostgreSQL must be reachable; Qdrant runs in local mode unless
``MEMORY__SEARCH__PROVIDER=qdrant`` and a URL are set.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from pathlib import Path

from sqlalchemy import text

from benchmark.common import provenance, write_result
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.ids import new_id
from memory_service.modules.evaluation.golden import (
    GoldenSet,
    RetrievedChunk,
    evaluate_question,
    summarize,
)
from memory_service.modules.jobs.registry import register_handlers

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "eval" / "golden" / "acme_fy26.json"
TABLES = (
    "graph_relations, graph_entities, memories, context_edges, chunks, document_nodes, document_versions, file_staging, documents, "
    "archive_segments, job_outbox, idempotency_keys, revisions, observations, turn_run_links, "
    "agent_runs, message_attachments, message_versions, messages, turns, sessions, threads"
)


def _settings() -> Settings:
    """Sandbox-friendly defaults; any ``MEMORY__*`` environment variable overrides them."""
    defaults = {
        "service": {"environment": "test", "log_json": False, "log_level": "WARNING"},
        "authentication": {"mode": "trusted_dev", "trusted_dev_api_keys": ["bench"]},
        "authorization": {"provider": "memory"},
        "cache": {"provider": "memory"},
        "search": {"provider": "memory"},
        "blob": {"provider": "memory"},
        "tasks": {"provider": "memory"},
        "models": {
            "embedding": {"provider": "hash", "dimension": 64},
            "reranker": {"provider": "lexical"},
            "llm": {"enabled": False},
        },
        "documents": {"parser": "builtin"},
        "observability": {"otel_enabled": False},
        "database": {
            "url": os.environ.get(
                "MEMORY__DATABASE__URL", "postgresql+psycopg://memory:memory@localhost:5432/memory"
            )
        },
    }
    env_only = Settings().model_dump(exclude_unset=True)
    for key, value in env_only.items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            defaults[key] = {**defaults[key], **value}
        else:
            defaults[key] = value
    return Settings(**defaults)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))
    return round(xs[idx], 2)


async def run(copies: int, queries: int) -> dict:
    settings = _settings()
    container = await build_container(settings, __version__)
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
        register_handlers(container)
        uow_factory = container.services["uow_factory"]
        ingestion = container.services["ingestion"]
        engine = container.services["retrieval"]
        builder = container.services["context_builder"]
        golden = GoldenSet.load(GOLDEN)
        ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")

        # --- corpus: the golden documents + (copies-1) salted copies (crowding pressure) -----
        t0 = time.perf_counter()
        aliases: dict[str, str] = {}
        for alias, filename in golden.documents.items():
            data = (FIXTURES / filename).read_bytes()
            for n in range(copies):
                salt = b"" if n == 0 else f"\n\n<!-- copy {n} -->\n".encode()
                async with uow_factory() as uow:
                    ack = await ingestion.accept_file(
                        uow,
                        ctx,
                        filename=filename,
                        media_type="text/markdown",
                        data=data + salt,
                        title=f"{alias}-{n}",
                    )
                    await uow.commit()
                aliases[ack.document_id] = alias  # copies are equivalent evidence
        await container.tasks.drain()
        await container.tasks.drain()
        index_seconds = round(time.perf_counter() - t0, 2)
        indexer = container.services["indexer"]
        from memory_service.modules.rag.indexer import KNOWLEDGE
        from memory_service.ports.search import SearchFilter

        points = await container.search.count(
            indexer.collection(KNOWLEDGE), SearchFilter(tenant_id="acme")
        )

        # --- quality on the golden set (duplicates count as distinct distractors) -----------
        k = settings.evaluation.critical_recall_k
        results = []
        for q in golden.questions:
            res = await engine.retrieve(ctx, q.query, limit=k)
            retrieved = [
                RetrievedChunk(
                    document_alias=aliases.get(str(c.payload.get("document_id"))),
                    page=c.payload.get("page"),
                    text=c.text,
                )
                for c in res.candidates
            ]
            results.append(
                evaluate_question(q, retrieved, k=k, observed_type=res.routed.query_type.value)
            )
        quality = summarize(results, k=k)
        quality["note"] = (
            "salted copies are exact duplicates of the golden documents (near-duplicate "
            "crowding): identical chunks collapse on text_hash and graph facts are "
            "de-duplicated per triple, so secondary evidence groups survive. Informational; "
            "the release gate is tests/eval on the un-duplicated corpus."
        )

        # --- latency ------------------------------------------------------------------------
        qs = [q.query for q in golden.questions]
        recall_ms: list[float] = []
        for i in range(queries):
            t = time.perf_counter()
            await engine.retrieve(ctx, qs[i % len(qs)], limit=k)
            recall_ms.append((time.perf_counter() - t) * 1000)
        thread_ctx = ctx.model_copy(
            update={
                "thread_id": new_id("thread"),
                "session_id": new_id("session"),
                "turn_id": new_id("turn"),
            }
        )
        cold_ms: list[float] = []
        warm_ms: list[float] = []
        for i in range(queries):
            q = f"{qs[i % len(qs)]} #{i}"  # unique -> cold
            t = time.perf_counter()
            await builder.build(thread_ctx, q)
            cold_ms.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            b = await builder.build(thread_ctx, q)
            warm_ms.append((time.perf_counter() - t) * 1000)
            assert b.cache_hit
        return {
            "corpus": {
                "documents": copies * len(golden.documents),
                "indexed_points": points,
                "index_seconds": index_seconds,
            },
            "providers": {
                "embedding": indexer.embedding.fingerprint(),
                "sparse": indexer.sparse.fingerprint(),
                "reranker": engine.reranker.fingerprint() if engine.reranker else None,
                "search": type(container.search).__name__,
                "representative": not indexer.embedding.fingerprint().startswith("hash-"),
            },
            "quality": quality,
            "latency_ms": {
                "recall_p50": _pct(recall_ms, 50),
                "recall_p95": _pct(recall_ms, 95),
                "context_cold_p50": _pct(cold_ms, 50),
                "context_cold_p95": _pct(cold_ms, 95),
                "context_cached_p50": _pct(warm_ms, 50),
                "context_cached_p95": _pct(warm_ms, 95),
                "samples": queries,
                "mean_recall": round(statistics.fmean(recall_ms), 2) if recall_ms else 0.0,
            },
            "budgets_ms": {
                "recall_p95": settings.budgets.recall_p95_ms,
                "context_bundle_p95": settings.budgets.context_bundle_p95_ms,
                "cached_context_p95": settings.budgets.cached_context_p95_ms,
            },
            "provenance": provenance(),
        }
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=25)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--out", default="retrieval.json")
    args = parser.parse_args()
    payload = asyncio.run(run(args.copies, args.queries))
    path = write_result(args.out, payload)
    lat = payload["latency_ms"]
    q = payload["quality"]
    print(f"wrote {path}")
    print(
        f"points={payload['corpus']['indexed_points']} recall p95={lat['recall_p95']}ms "
        f"context cold p95={lat['context_cold_p95']}ms cached p95={lat['context_cached_p95']}ms "
        f"| critical Recall@{q['k']}={q['critical_recall_at_k']} "
        f"EGR={q['critical_evidence_group_recall']} representative={payload['providers']['representative']}"
    )


if __name__ == "__main__":
    main()
