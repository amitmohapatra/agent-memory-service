"""How much parallelism the model tier actually wants, and what it costs in memory.

Two defaults in this repository have never been measured: ``service.worker_concurrency = 4``
and ``models.threads = None``. Together they mean four jobs call into the *same* torch module
while torch fans each operation across every core — sixteen compute threads on four cores, on
a shared object with no lock anywhere in ``adapters/models``. Nobody had checked whether that
is faster than one, or what it costs in resident memory, or whether concurrent entry into a
shared module returns the same numbers as sequential entry.

This measures all three, for the two models on the hot path:

* **throughput** — items per second at each concurrency level;
* **peak RSS** — because parallelism that doubles throughput and triples memory is not a win
  on a container with a limit, and every deployment has a limit;
* **agreement** — concurrent results compared against a sequential baseline, element by
  element. PyTorch inference on a shared module in eval mode is *usually* safe, and "usually"
  is not a property you can deploy. If concurrent entry ever disagrees with sequential entry,
  that is state being shared that should not be, and it is worth far more than the throughput
  number next to it.

Nothing here is a pass/fail gate. It prints the shape of the trade-off so the defaults can be
chosen from evidence rather than from habit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import sys
import time

from benchmark.common import provenance, write_result
from benchmark.env import bench_overrides
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container

#: Sentences long enough to be realistic work rather than tokeniser overhead.
_TEXTS = [
    "The Federal Open Market Committee held the target range at 5.25 to 5.50 percent, "
    f"citing persistent core services inflation in the {month} report."
    for month in (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
]

QUERY = "What did the committee decide about interest rates?"


def _peak_rss_mb() -> float:
    """Peak resident set size for this process, in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _close(a: list[float], b: list[float], tol: float = 1e-5) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


async def _measure(container, workers: int, rounds: int) -> dict:
    """One (workers) point: run `rounds` batches split across `workers` concurrent callers."""
    embedding = container.embedding
    reranker = container.reranker

    # sequential baseline for the agreement check
    baseline = await embedding.embed_documents(_TEXTS)

    started = time.perf_counter()
    batches = [_TEXTS] * rounds

    async def one(batch: list[str]) -> list[list[float]]:
        return await embedding.embed_documents(batch)

    results: list[list[list[float]]] = []
    for i in range(0, len(batches), workers):
        group = batches[i : i + workers]
        results.extend(await asyncio.gather(*(one(b) for b in group)))
    embed_seconds = time.perf_counter() - started
    embedded = rounds * len(_TEXTS)

    # every concurrent result must match the sequential one
    disagreements = sum(
        1
        for out in results
        for got, want in zip(out, baseline, strict=True)
        if not _close(got, want)
    )

    rerank_seconds = None
    reranked = 0
    if reranker is not None and getattr(reranker, "representative", True):
        docs = list(_TEXTS)
        started = time.perf_counter()

        async def score() -> list[float]:
            return await reranker.rerank(QUERY, docs)

        for i in range(0, rounds, workers):
            group = min(workers, rounds - i)
            await asyncio.gather(*(score() for _ in range(group)))
            reranked += group * len(docs)
        rerank_seconds = time.perf_counter() - started

    return {
        "workers": workers,
        "embed_per_second": round(embedded / embed_seconds, 1) if embed_seconds else 0.0,
        "rerank_pairs_per_second": (
            round(reranked / rerank_seconds, 1) if rerank_seconds else None
        ),
        "disagreements_vs_sequential": disagreements,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }


async def run(workers: list[int], rounds: int) -> dict:
    settings = _settings()
    container = await build_container(settings, __version__, overrides=bench_overrides())
    try:
        import torch

        points = []
        for w in workers:
            point = await _measure(container, w, rounds)
            points.append(point)
            print(f"[concurrency] {point}", file=sys.stderr, flush=True)
    finally:
        await container.close()

    best = max(points, key=lambda p: p["embed_per_second"])
    return {
        "benchmark": "model_concurrency",
        "environment": {
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "default_executor_max_workers": min(32, (os.cpu_count() or 1) + 4),
            "configured_models_threads": settings.models.embedding.threads,
            "configured_worker_concurrency": settings.tasks.worker_concurrency,
        },
        "points": points,
        "fastest_workers": best["workers"],
        "speedup_over_serial": round(best["embed_per_second"] / points[0]["embed_per_second"], 2)
        if points and points[0]["embed_per_second"]
        else None,
        # The number that decides whether the shared-module hazard is real on this stack.
        "any_disagreement": any(p["disagreements_vs_sequential"] for p in points),
        "provenance": provenance(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="1,2,4", help="concurrent callers to compare")
    parser.add_argument("--rounds", type=int, default=12, help="batches per point")
    parser.add_argument("--out", default="concurrency.json")
    args = parser.parse_args()
    workers = [int(w) for w in args.workers.split(",") if w.strip()]
    result = asyncio.run(run(workers, args.rounds))
    write_result(args.out, result)
    sys.stdout.write(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
