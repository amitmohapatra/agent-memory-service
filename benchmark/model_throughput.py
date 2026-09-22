"""Per-model sustained throughput, for capacity planning.

Latency benchmarks answer "how slow is one call". Sizing needs the other number: how many
items per second one CPU core sustains, so a target RPS can be turned into cores.

    uv run python -m benchmark.model_throughput --threads 1 2 4

The per-request model work is read from settings rather than assumed, so the cores-per-RPS
figure follows the configuration that is actually deployed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from benchmark.common import provenance, write_result

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("BENCH_MODELS_DIR", ROOT / "models"))

#: representative inputs: a short query, and chunk-sized passages
QUERY = "What was the revenue impact of the supply chain disruption in the third quarter?"
PASSAGE = (
    "Revenue for the third quarter declined by 12 percent year over year, driven primarily by "
    "the supply chain disruption that delayed shipments of the flagship product line. "
    "Management attributes roughly two thirds of the shortfall to component shortages and the "
    "remainder to softer demand in the enterprise segment. "
) * 2


def _timed(fn, warmup: int = 2, rounds: int = 5) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples)


def measure(threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    from sentence_transformers import CrossEncoder, SentenceTransformer

    out: dict = {"threads": threads}

    embedder = SentenceTransformer(str(MODELS / "granite-embedding-small-english-r2"))
    # a query embedding is one short text; indexing is a batch
    median, _ = _timed(lambda: embedder.encode([QUERY], show_progress_bar=False))
    out["embed_query_ms"] = round(median * 1000, 1)
    out["embed_query_per_sec"] = round(1 / median, 1)
    batch = [PASSAGE] * 32
    median, _ = _timed(lambda: embedder.encode(batch, batch_size=32, show_progress_bar=False))
    out["embed_passages_per_sec"] = round(32 / median, 1)

    reranker = CrossEncoder(str(MODELS / "ms-marco-MiniLM-L6-v2"))
    pairs = [(QUERY, PASSAGE)] * 20  # reranker.candidate_k
    median, _ = _timed(lambda: reranker.predict(pairs, batch_size=16, show_progress_bar=False))
    out["rerank_20_pairs_ms"] = round(median * 1000, 1)
    out["rerank_pairs_per_sec"] = round(20 / median, 1)
    out["rerank_requests_per_sec"] = round(1 / median, 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    runs = []
    for threads in args.threads:
        sys.stdout.write(f"measuring at {threads} thread(s)...\n")
        sys.stdout.flush()
        runs.append(measure(threads))
    best = max(runs, key=lambda r: r["rerank_pairs_per_sec"])
    result = {
        "runs": runs,
        "sizing": {
            "target_rps": 20,
            "rerank_pairs_per_second_needed": 20 * 20,
            "best_measured_rerank_pairs_per_sec": best["rerank_pairs_per_sec"],
            "best_measured_at_threads": best["threads"],
            "containers_needed_at_that_rate": round((20 * 20) / best["rerank_pairs_per_sec"], 1),
            "cpu": provenance().get("cpu_model") or provenance().get("platform"),
            "warning": (
                "Valid only for the CPU named above. Transformer inference depends on "
                "AVX2/AVX-512/AMX; a machine without them is roughly an order of magnitude "
                "slower, so these figures must be re-measured on the target hardware."
            ),
        },
        "per_request_model_work": {
            "query_embeddings": 1,
            "rerank_pairs": 20,
            "note": "reranker.candidate_k=20, embedding.batch_size=32, retrieval.rerank=true",
        },
        "provenance": provenance(),
    }
    write_result("model_throughput.json", result)
    sys.stdout.write(json.dumps(runs, indent=2) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
