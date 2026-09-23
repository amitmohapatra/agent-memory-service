"""Which encoder runtime to ship, measured rather than argued.

The dense encoder is 61% of the query p99 budget (docs/LATENCY-LAYERS-2026-09.md), so the
choice between torch, ONNX fp32 and ONNX int8 decides both targets on its own. Two things
make a naive timing of them useless: the runtimes are loaded one after another, so whichever
runs last meets a warmer page cache, and this box is shared with Postgres, Qdrant and the
gateway, so a run of forty timings for one runtime and then forty for the next compares two
different machines.

So the variants are rotated one query at a time. Each runtime meets the same queries in the
same order under the same contention, and the *ratio* between them survives a noisy box even
when the absolute numbers do not.

The cosine column is the part that matters as much as the speed: a runtime that returns
different vectors cannot be swapped in without re-measuring retrieval, while one that returns
identical vectors can only change the clock.

    python -m benchmark.encoder_ab --threads 1 --queries 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics as st
import time
from pathlib import Path
from typing import Any

from memory_service.adapters.models.embeddings import OnnxEmbedding, SentenceTransformersEmbedding
from memory_service.config.constants import FROZEN_MODELS

#: Every runtime the image can load for the frozen dense model, as (label, overrides).
VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("torch-fp32", {"runtime": "torch", "backend": "torch"}),
    ("onnx-fp32", {"runtime": "onnx", "graph_file": "onnx/model.onnx"}),
    ("onnx-int8", {"runtime": "onnx", "graph_file": "onnx/model_qint8.onnx"}),
)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low, high = math.floor(k), math.ceil(k)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _queries(count: int, dataset: Path) -> list[str]:
    """Real questions, because token length drives the cost being measured."""
    if not dataset.exists():
        return ["What did Melanie say about her painting hobby?"] * count
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    found = [q["question"] for conv in raw for q in conv.get("qa", []) if q.get("question")]
    return found[:count] or ["What did Melanie say about her painting hobby?"] * count


def _build(model_dir: str | None, threads: int) -> dict[str, Any]:
    base = FROZEN_MODELS.dense
    if model_dir:
        base = base.model_copy(update={"model_path": model_dir})
    built: dict[str, Any] = {}
    for label, overrides in VARIANTS:
        spec = base.model_copy(update=overrides)
        cls = SentenceTransformersEmbedding if spec.runtime == "torch" else OnnxEmbedding
        try:
            built[label] = cls(spec, threads=threads)
        except Exception as exc:  # noqa: BLE001 - a missing graph is a skip, not a failure
            print(f"[encoder_ab] skipping {label}: {type(exc).__name__}: {exc}")
    return built


async def _measure(
    built: dict[str, Any], queries: list[str]
) -> tuple[dict[str, list[float]], dict[str, list[list[float]]]]:
    for encoder in built.values():
        await encoder.embed_query("warm the graph and the page cache")
    times: dict[str, list[float]] = {name: [] for name in built}
    vectors: dict[str, list[list[float]]] = {name: [] for name in built}
    for query in queries:
        for name, encoder in built.items():  # rotate: contention is shared, not stacked
            start = time.perf_counter()
            vector = await encoder.embed_query(query)
            times[name].append((time.perf_counter() - start) * 1000)
            vectors[name].append(vector)
    return times, vectors


def _report(times: dict[str, list[float]], vectors: dict[str, list[list[float]]]) -> None:
    reference = next(iter(times))
    baseline = st.median(times[reference])
    print(f"\n{'runtime':<12}{'p50':>8}{'p90':>8}{'p99':>8}{'mean':>8}{'speedup':>9}{'cosine':>10}")
    for name, samples in times.items():
        if name == reference:
            fidelity = "-"
        else:
            sims = [_cosine(a, b) for a, b in zip(vectors[reference], vectors[name], strict=True)]
            fidelity = f"{min(sims):.5f}"
        print(
            f"{name:<12}{_percentile(samples, 50):>8.1f}{_percentile(samples, 90):>8.1f}"
            f"{_percentile(samples, 99):>8.1f}{st.mean(samples):>8.1f}"
            f"{baseline / st.median(samples):>8.2f}x{fidelity:>10}"
        )
    print("\ncosine is the WORST case against the first runtime; 1.00000 means identical vectors")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="intra-op threads per runtime")
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--model-dir", default=None, help="overrides the frozen model's source")
    parser.add_argument("--dataset", default="benchmark/data/locomo10.json")
    args = parser.parse_args()

    built = _build(args.model_dir, args.threads)
    if len(built) < 2:
        raise SystemExit("need at least two loadable runtimes to compare")
    queries = _queries(args.queries, Path(args.dataset))
    print(f"[encoder_ab] {len(queries)} queries x {len(built)} runtimes, {args.threads} thread(s)")
    times, vectors = await _measure(built, queries)
    _report(times, vectors)
    for encoder in built.values():
        encoder.close()


if __name__ == "__main__":
    asyncio.run(main())
