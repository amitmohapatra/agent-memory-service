"""Embedding runtime benchmark: Granite ``english-r2`` (768d) vs ``small-english-r2`` (384d)
across the sentence-transformers backends (torch / ONNX / OpenVINO). Every candidate goes
through the real pipeline: a container built with that embedding setting, the golden corpus
re-indexed (the fingerprint changes the collection names) and the golden questions retrieved.

    uv run python -m benchmark.embedding              # both models x three backends
    uv run python -m benchmark.embedding --quick      # torch backends only
    uv run python -m benchmark.embedding --stand-in   # hash embedding; representative=false

Weights are read from ``./models/<model-dir>`` (or ``/models`` in the image). A backend that
cannot be loaded (missing extra, impossible ONNX/OpenVINO export, ...) is recorded as
``{"skipped": "<ExceptionType>: reason"}`` — never dropped. ``MEMORY__MODELS__EMBEDDING__THREADS``
bounds the CPU threads of every candidate. The challengers here are benchmark candidates,
never the product: the service ships exactly ``constants.FROZEN_MODELS``.

Verdict rule ("quality first, then speed"): the default is the candidate with the lowest
(query embed p95 ms, dimension) among those whose critical Recall@k and critical evidence-group
recall equal the best observed pair.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from benchmark.advanced import _corpus
from benchmark.common import provenance, reset_store, write_result
from benchmark.env import BENCH, bench_overrides
from benchmark.evaluation import CRITICAL_RECALL_K
from benchmark.evaluation.golden import (
    GoldenSet,
    RetrievedChunk,
    evaluate_question,
    summarize,
)
from benchmark.retrieval import FIXTURES, GOLDEN, _pct, _settings
from memory_service.__about__ import __version__
from memory_service.adapters.models.embeddings import HashEmbedding, SentenceTransformersEmbedding
from memory_service.application.container import Overrides, build_container
from memory_service.config.constants import DenseModel
from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES
from memory_service.ports.search import SearchFilter

MODELS: dict[str, tuple[str, int]] = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", 384),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", 768),
    "granite-embedding-small-english-r2": ("ibm-granite/granite-embedding-small-english-r2", 384),
    "granite-embedding-english-r2": ("ibm-granite/granite-embedding-english-r2", 768),
    "qwen3-embedding-0.6b": ("Qwen/Qwen3-Embedding-0.6B", 1024),
}
BACKENDS = ("torch", "onnx", "openvino")
QUICK_BACKENDS = ("torch",)
STAND_IN = "hash-stand-in"
STAND_IN_DIMENSION = 64
QUALITY_KEYS = ("critical_recall_at_k", "critical_evidence_group_recall")
BATCH_DOCUMENTS = 32


def models_dir() -> Path:
    return Path(os.environ.get("BENCH_MODELS_DIR", "models"))


def candidates(*, quick: bool = False, stand_in: bool = False) -> dict[str, DenseModel | None]:
    """Name -> dense encoder spec; ``None`` is the hash stand-in."""
    if stand_in:
        return {STAND_IN: None}
    root = models_dir()
    out: dict[str, DenseModel | None] = {}
    for short, (model, dimension) in MODELS.items():
        for backend in QUICK_BACKENDS if quick else BACKENDS:
            out[f"{short}/{backend}"] = DenseModel(
                id=model,
                local_dir=short,
                model_path=str(root / short),
                backend=backend,  # type: ignore[arg-type]
                dimension=dimension,
            )
    return out


def candidate_overrides(spec: DenseModel | None, **changes: Any) -> Overrides:
    """The benchmark stand-ins with this candidate as the encoder (``None``: the hash)."""
    if spec is None:
        return bench_overrides(
            embedding="hash", embedding_dimension=STAND_IN_DIMENSION, dense_model=None, **changes
        )
    return bench_overrides(embedding=None, dense_model=spec, **changes)


def bench_settings() -> Settings:
    """``benchmark.retrieval._settings()``: environment providers apply. The in-process task
    queue the corpus is drained through comes from ``bench_overrides()``."""
    return _settings()


async def reset_index(container: Any) -> None:
    """Server-side Qdrant collections and a shared cache outlive the PostgreSQL truncate."""
    stand_ins = container.overrides
    if stand_ins.search is None and stand_ins.search_local_path is None:
        indexer = container.services["indexer"]
        for base in (KNOWLEDGE, MEMORIES):
            await container.search.drop_collection(indexer.collection(base))
        await indexer.ensure_collections()
    if container.cache is not None and stand_ins.cache is None:
        keys = [key async for key in container.cache.scan("*")]
        if keys:
            await container.cache.delete(*keys)


def load_embedding(spec: DenseModel | None, *, threads: int | None = None) -> Any:
    if spec is None:
        return HashEmbedding(STAND_IN_DIMENSION)
    return SentenceTransformersEmbedding(spec, threads=threads)


def sample_documents(golden: GoldenSet, n: int = BATCH_DOCUMENTS) -> list[str]:
    """``n`` paragraphs of the golden documents (cycled when the corpus is shorter)."""
    paragraphs: list[str] = []
    for filename in golden.documents.values():
        body = (FIXTURES / filename).read_text(encoding="utf-8")
        paragraphs.extend(p.strip() for p in body.split("\n\n") if len(p.strip()) > 40)
    if not paragraphs:
        raise ValueError("golden fixtures contain no paragraphs")
    return [paragraphs[i % len(paragraphs)] for i in range(n)]


def quality_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(row["quality"][k]) for k in QUALITY_KEYS)


def pick_default(
    rows: dict[str, dict[str, Any]], cost: Callable[[dict[str, Any]], tuple[float, ...]]
) -> dict[str, Any]:
    """Quality first, then cost: among the rows that ran, keep those whose critical
    Recall@k / EGR pair equals the best observed, then take the lowest ``cost``."""
    ran = {name: row for name, row in rows.items() if "skipped" not in row}
    if not ran:
        return {
            "default": None,
            "eligible": [],
            "best": None,
            "reason": "every candidate was skipped",
        }
    best = max(quality_key(row) for row in ran.values())
    eligible = sorted(
        (name for name, row in ran.items() if quality_key(row) == best),
        key=lambda name: (*cost(ran[name]), name),
    )
    return {
        "default": eligible[0],
        "eligible": eligible,
        "best": dict(zip(QUALITY_KEYS, best, strict=True)),
        "reason": "lowest cost among candidates matching the best critical recall / EGR",
    }


def embedding_cost(row: dict[str, Any]) -> tuple[float, ...]:
    return (float(row["embed_ms"]["query_p95"]), float(row["dimension"]))


async def golden_quality(
    engine: Any, golden: GoldenSet, aliases: dict[str, str], ctx: MemoryExecutionContext, *, k: int
) -> tuple[dict[str, Any], list[float]]:
    """Golden questions through ``engine.retrieve``; returns the summary and per-query ms."""
    results, latencies = [], []
    for q in golden.questions:
        t = time.perf_counter()
        res = await engine.retrieve(ctx, q.query, limit=k)
        latencies.append((time.perf_counter() - t) * 1000)
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
    summary = summarize(results, k=k)
    summary.pop("per_question", None)
    return summary, latencies


async def embed_latency(
    embedding: Any, documents: Sequence[str], queries: Sequence[str], *, batches: int
) -> dict[str, Any]:
    await embedding.embed_documents(documents[:4])
    await embedding.embed_query(queries[0])
    batch_ms: list[float] = []
    for _ in range(batches):
        t = time.perf_counter()
        await embedding.embed_documents(documents)
        batch_ms.append((time.perf_counter() - t) * 1000)
    query_ms: list[float] = []
    for q in queries:
        t = time.perf_counter()
        await embedding.embed_query(q)
        query_ms.append((time.perf_counter() - t) * 1000)
    return {
        "batch_size": len(documents),
        "batches": batches,
        "batch_mean": round(statistics.fmean(batch_ms), 2),
        "batch_p95": _pct(batch_ms, 95),
        "queries": len(queries),
        "query_mean": round(statistics.fmean(query_ms), 2),
        "query_p95": _pct(query_ms, 95),
    }


async def run_candidate(
    spec: DenseModel | None, *, base: Settings, copies: int, batches: int
) -> dict[str, Any]:
    head = (
        {"model": spec.id, "backend": spec.backend, "model_path": spec.model_path}
        if spec is not None
        else {"model": STAND_IN, "backend": "hash", "model_path": None}
    )
    threads = base.models.embedding.threads
    t0 = time.perf_counter()
    try:
        model = load_embedding(spec, threads=threads)
    except Exception as exc:  # noqa: BLE001 - a benchmark reports, it does not crash
        return {**head, "skipped": f"{type(exc).__name__}: {exc}"}
    load_seconds = round(time.perf_counter() - t0, 2)
    golden = GoldenSet.load(GOLDEN)
    embed_ms = await embed_latency(
        model, sample_documents(golden), [q.query for q in golden.questions], batches=batches
    )
    del model

    container = await build_container(base, __version__, overrides=candidate_overrides(spec))
    try:
        # both stores: the vector store is a separate server and a SQL TRUNCATE
        # leaves its vectors behind for the next run to retrieve
        await reset_store(container, "acme")
        await reset_index(container)
        ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
        t0 = time.perf_counter()
        aliases = await _corpus(container, golden, copies, ctx)
        index_seconds = round(time.perf_counter() - t0, 2)
        indexer = container.services["indexer"]
        points = await container.search.count(
            indexer.collection(KNOWLEDGE), SearchFilter(tenant_id="acme")
        )
        k = CRITICAL_RECALL_K
        quality, recall_ms = await golden_quality(
            container.services["retrieval"], golden, aliases, ctx, k=k
        )
        fingerprint = container.embedding.fingerprint()
        return {
            **head,
            "fingerprint": fingerprint,
            "dimension": int(container.embedding.dimension),
            "threads": threads,
            "load_seconds": load_seconds,
            "embed_ms": embed_ms,
            "index_seconds": index_seconds,
            "indexed_points": points,
            "quality": quality,
            "recall_p95_ms": _pct(recall_ms, 95),
            "quality_per_ms": round(
                quality["critical_recall_at_k"] / max(embed_ms["query_p95"], 1e-3), 4
            ),
            "representative": not fingerprint.startswith("hash-"),
        }
    finally:
        await container.close()


async def run(
    *, copies: int, batches: int, quick: bool = False, stand_in: bool = False
) -> dict[str, Any]:
    base = bench_settings()
    rows: dict[str, dict[str, Any]] = {}
    for name, spec in candidates(quick=quick, stand_in=stand_in).items():
        rows[name] = await run_candidate(spec, base=base, copies=copies, batches=batches)
    ran = [row for row in rows.values() if "skipped" not in row]
    golden = GoldenSet.load(GOLDEN)
    return {
        "corpus": {"copies": copies, "documents": copies * len(golden.documents)},
        "mode": {"quick": quick, "stand_in": stand_in},
        "candidates": rows,
        "verdict": pick_default(rows, embedding_cost),
        "providers": {
            "reranker": "none",
            "search": BENCH.search,
            "representative": bool(ran) and all(row["representative"] for row in ran),
        },
        "note": (
            "Per-candidate numbers come from a container built with that embedding setting on "
            "the golden corpus; latencies are CPU wall-clock (batch of 32 documents, single "
            "query). representative=false means the hash stand-in ran instead of Granite."
        ),
        "provenance": provenance(),
    }


def print_table(rows: dict[str, dict[str, Any]], verdict: dict[str, Any]) -> None:
    print(
        f"{'candidate':52} {'dim':>4} {'load_s':>7} {'q_p95ms':>8} {'b32_p95ms':>9} "
        f"{'index_s':>7} {'R@k':>6} {'EGR':>6} {'q/ms':>7}"
    )
    for name, row in rows.items():
        if "skipped" in row:
            print(f"{name:52} skipped: {row['skipped'][:100]}")
            continue
        q, e = row["quality"], row["embed_ms"]
        print(
            f"{name:52} {row['dimension']:>4} {row['load_seconds']:>7} {e['query_p95']:>8} "
            f"{e['batch_p95']:>9} {row['index_seconds']:>7} {q['critical_recall_at_k']:>6} "
            f"{q['critical_evidence_group_recall']:>6} {row['quality_per_ms']:>7}"
        )
    print(f"default -> {verdict['default']} ({verdict['reason']})")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=5)
    parser.add_argument("--batches", type=int, default=5, help="batch-of-32 embed samples")
    parser.add_argument("--quick", action="store_true", help="torch backends only")
    parser.add_argument(
        "--stand-in", action="store_true", help="hash embedding only (representative=false)"
    )
    parser.add_argument("--out", default="embedding.json")
    args = parser.parse_args(argv)
    payload = asyncio.run(
        run(copies=args.copies, batches=args.batches, quick=args.quick, stand_in=args.stand_in)
    )
    path = write_result(args.out, payload)
    print(f"wrote {path} representative={payload['providers']['representative']}")
    print_table(payload["candidates"], payload["verdict"])


if __name__ == "__main__":
    main()
