"""Measure CPU query interference from concurrent indexing, without model API calls.

Run sequentially against each checkout's src with the same model specification. These
are component timings under a synthetic workload, not application latency percentiles.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import time
from pathlib import Path

from benchmark.harness import stats
from benchmark.model_throughput import PASSAGE, QUERY
from memory_service.adapters.models import embeddings
from memory_service.config.constants import DenseModel


async def run(args) -> None:
    spec = DenseModel.model_validate_json(args.spec.read_text())
    encoder = embeddings.load_dense(spec)
    result = {
        "spec": spec.model_dump(),
        "encoder": encoder.fingerprint(),
        "embedding_source_sha256": args.source_sha256,
        "platform": platform.platform(),
        "documents": args.batches * spec.batch_size,
        "queries_per_phase": args.queries,
        "document_sha256": hashlib.sha256(PASSAGE.encode()).hexdigest(),
        "llm_calls": 0,
        "limitations": [
            "Synthetic component contention; excludes HTTP, authorization, database and search.",
            "Small sample percentiles are exploratory, not a production p99 guarantee.",
        ],
    }
    try:
        await encoder.embed_query(QUERY)
        idle, mixed = [], []
        for _ in range(args.queries):
            start = time.perf_counter()
            await encoder.embed_query(QUERY)
            idle.append((time.perf_counter() - start) * 1000)

        start_index = time.perf_counter()
        indexing = asyncio.create_task(encoder.embed_documents([PASSAGE] * result["documents"]))
        await asyncio.sleep(0)  # indexing enters the executor before the first query
        try:
            for _ in range(args.queries):
                start = time.perf_counter()
                await encoder.embed_query(QUERY)
                mixed.append((time.perf_counter() - start) * 1000)
        finally:
            vectors = await indexing
        assert len(vectors) == result["documents"]
        result.update(
            idle_query_ms=stats(idle),
            concurrent_query_ms=stats(mixed),
            concurrent_query_samples_ms=mixed,
            workload_elapsed_ms=(time.perf_counter() - start_index) * 1000,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("idle_query_ms", "concurrent_query_ms")}))
    finally:
        encoder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--queries", type=int, choices=range(10, 1001), default=100)
    args = parser.parse_args()
    args.source_sha256 = hashlib.sha256(Path(embeddings.__file__).read_bytes()).hexdigest()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
