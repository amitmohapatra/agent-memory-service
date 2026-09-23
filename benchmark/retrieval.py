"""Retrieval benchmark: index a synthetic corpus derived from the fixtures, then measure
recall / context-bundle latency (p50/p95) and the golden-set quality metrics with whatever
providers the environment configures.

    uv run python -m benchmark.retrieval --copies 40 --queries 30

The encoder is ``BENCH_EMBEDDING`` (``benchmark/env.py``): ``frozen`` for the shipped
Granite weights, ``hash`` for the deterministic stand-in, defaulting to whichever the model
roots can satisfy. The result records the provider fingerprints and whether they are
representative.
PostgreSQL must be reachable; Qdrant runs in local mode unless ``BENCH_SEARCH=qdrant`` and
``MEMORY__SEARCH__QDRANT_URL`` are set (see ``benchmark/env.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.common import provenance, reset_store, write_result
from benchmark.env import bench_llm_settings, bench_overrides, bench_retrieval
from benchmark.evaluation import BUDGETS, CRITICAL_RECALL_K
from benchmark.evaluation.golden import (
    GoldenSet,
    RetrievedChunk,
    evaluate_question,
    summarize,
)
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "eval" / "golden" / "acme_fy26.json"
TABLES = (
    "tool_invocations, run_outcomes, tools, graph_relations, graph_entities, memories, context_edges, chunks, document_nodes, document_versions, file_staging, documents, "
    "archive_segments, job_outbox, idempotency_keys, revisions, observations, turn_run_links, "
    "agent_runs, message_attachments, message_versions, messages, turns, sessions, threads"
)


def _settings() -> Settings:
    """Sandbox-friendly defaults; any ``MEMORY__*`` environment variable overrides them."""
    defaults = {
        "service": {"environment": "test", "log_json": False, "log_level": "WARNING"},
        "authentication": {"mode": "trusted_dev", "trusted_dev_api_keys": ["bench"]},
        "models": {"llm": {"enabled": False, **bench_llm_settings()}},
        "database": {
            "url": os.environ.get(
                "MEMORY__DATABASE__URL", "postgresql+psycopg://memory:memory@localhost:5432/memory"
            )
        },
    }
    for path in _env_paths():
        _overlay(defaults, Settings().model_dump(), path)
    return Settings(**defaults)


ENV_PREFIX = "MEMORY__"


def _env_paths() -> set[tuple[str, ...]]:
    """The settings paths a ``MEMORY__*`` variable actually names, e.g. ``models.llm.model``.

    This replaces ``Settings().model_dump(exclude_unset=True)``, which does not do what its
    name suggests on a pydantic-settings object: every field counts as "set" because a
    settings *source* provided it, so the dump came back complete - every section, every
    default inlined - and overlaying it onto the benchmark's defaults overwrote them all.

    Measured on a judged LoCoMo run, which intends ``max_tokens=16384, timeout_seconds=120,
    max_retries=0`` and was getting ``1024, 30.0, 2``; ``service.environment`` and
    ``log_level`` were lost the same way. Each is its own defect. A reasoning model given
    1024 tokens spends them thinking and returns an empty string - the "output budget was
    exhausted" failure that lost 19 of v6's 233 answerable rows. Thirty seconds cuts off a
    judge call 120 would have allowed. And retries at 2 means one logical call can send three
    wire requests, so the Makefile's claim that "with retries off, the pacer's rate is the
    actual request rate, and --calls-per-minute means what it says" was false for every
    judged run ever taken.

    Reading the variable names instead is exact: a value is taken from the environment when,
    and only when, someone set a variable for it.
    """
    return {
        tuple(name[len(ENV_PREFIX) :].lower().split("__"))
        for name in os.environ
        if name.startswith(ENV_PREFIX) and name != ENV_PREFIX
    }


def _overlay(target: dict[str, Any], source: dict[str, Any], path: tuple[str, ...]) -> None:
    """Copy one leaf from ``source`` into ``target``, creating the branches it needs.

    A path the settings model does not have is ignored rather than invented: an unknown
    ``MEMORY__*`` variable is a typo, and inventing a key for it would turn a typo into a
    validation error somewhere unrelated.
    """
    for key in path[:-1]:
        if not isinstance(source, dict) or key not in source:
            return
        source = source[key]
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            return
    leaf = path[-1]
    if isinstance(source, dict) and leaf in source:
        target[leaf] = source[leaf]


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))
    return round(xs[idx], 2)


async def run(copies: int, queries: int, *, ablate: dict[str, bool] | None = None) -> dict:
    settings = _settings()
    overrides = bench_overrides()
    if ablate:
        # Expansion flags cannot be measured on a flat corpus: SciFact abstracts are one
        # chunk each, so parent/neighbour/definition expansion has no parent, no neighbour
        # and no cross-chunk definition to reach for, and every ablation of them scored
        # exactly zero difference. This golden set has section hierarchy, tables, footnotes
        # and `required_groups` — the mechanism those flags exist to serve — so it is the
        # instrument that can actually tell whether they earn their cost.
        overrides = replace(
            overrides, retrieval=bench_retrieval(overrides).model_copy(update=ablate)
        )
    container = await build_container(settings, __version__, overrides=overrides)
    try:
        # both stores, not just SQL: the vector store is a separate server and survives a
        # TRUNCATE, so every previous run's vectors would otherwise compete with this one
        await reset_store(container, "acme")
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
        k = CRITICAL_RECALL_K
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
                "recall_p95": BUDGETS.recall_p95_ms,
                "context_bundle_p95": BUDGETS.context_bundle_p95_ms,
                "cached_context_p95": BUDGETS.cached_context_p95_ms,
            },
            "provenance": provenance(),
        }
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=25)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument(
        "--off", nargs="*", default=[], metavar="FLAG", help="retrieval flags to disable"
    )
    parser.add_argument("--out", default="retrieval.json")
    args = parser.parse_args()
    payload = asyncio.run(run(args.copies, args.queries, ablate=dict.fromkeys(args.off, False)))
    payload["ablation"] = dict.fromkeys(args.off, False)
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
