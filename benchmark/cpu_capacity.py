"""Fixed-arrival CPU encoder screen; never a claim about endpoint capacity.

The arrival schedule continues during overload. Latency starts at the scheduled arrival,
and requests rejected by the bounded queue remain in the denominator. CPU time measures
all threads of this process, rather than multiplying wall latency by a thread setting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import resource
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from benchmark.common import file_sha256
from benchmark.harness import stats
from benchmark.multilingual_sparse import load_dataset
from memory_service.adapters.models.embeddings import load_dense
from memory_service.config.constants import DenseModel


async def fixed_arrivals(
    operation: Callable[[int], Awaitable[object]],
    *,
    rate: float,
    requests: int,
    max_pending: int,
    deadline_seconds: float,
) -> dict:
    """O(requests) result storage, at most max_pending live request tasks."""
    if (
        not math.isfinite(rate)
        or rate <= 0
        or requests < 1
        or max_pending < 1
        or not math.isfinite(deadline_seconds)
        or deadline_seconds <= 0
    ):
        raise ValueError("Rate, count, queue bound and timeout must be positive")
    rows: list[dict] = []
    pending: set[asyncio.Task] = set()
    start = time.perf_counter()
    cpu_start = time.process_time()

    async def one(index: int, scheduled: float) -> None:
        status = "ok"
        error = None
        try:
            remaining = deadline_seconds - (time.perf_counter() - scheduled)
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                await operation(index)
        except TimeoutError:
            status = "timeout"
        except Exception as exc:
            status, error = "error", type(exc).__name__
        elapsed = time.perf_counter() - scheduled
        # A native inference thread can outlive cancellation. Its eventual completion
        # must not convert a missed deadline into a successful response.
        if status == "ok" and elapsed > deadline_seconds:
            status = "timeout"
        rows.append(
            {"index": index, "status": status, "latency_ms": elapsed * 1000, "error": error}
        )

    try:
        for index in range(requests):
            scheduled = start + index / rate
            await asyncio.sleep(max(0, scheduled - time.perf_counter()))
            if len(pending) >= max_pending:
                rows.append(
                    {"index": index, "status": "overload", "latency_ms": None, "error": None}
                )
                continue
            task = asyncio.create_task(one(index, scheduled))
            pending.add(task)
            task.add_done_callback(pending.discard)
        if pending:
            await asyncio.gather(*pending)
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    elapsed = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    successful = [row for row in rows if row["status"] == "ok"]
    window = max(requests / rate, elapsed)
    return {
        "offered_rps": rate,
        "offered_requests": requests,
        "successful_requests": len(successful),
        "successful_rps_including_drain": len(successful) / window,
        "failure_ratio": 1 - len(successful) / requests,
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu_seconds,
        "cpu_seconds_per_success": cpu_seconds / len(successful) if successful else None,
        "successful_latency_ms": stats([row["latency_ms"] for row in successful])
        if successful
        else None,
        "rows": sorted(rows, key=lambda row: row["index"]),
    }


def cpu_projection(cpu_seconds_per_request: float, *, rate: float, cores: int) -> dict:
    """Necessary CPU budget only: excludes other processes and concurrency limits."""
    if not math.isfinite(cpu_seconds_per_request) or cpu_seconds_per_request <= 0:
        raise ValueError("A finite positive CPU measurement is required")
    if not math.isfinite(rate) or rate <= 0 or cores < 1:
        raise ValueError("Rate and cores must be positive")
    available = cores * 0.70
    demand = rate * cpu_seconds_per_request
    return {
        "target_rps": rate,
        "target_vcpus": cores,
        "cpu_utilization_ceiling": 0.70,
        "encoder_core_demand": demand,
        "cores_remaining_with_headroom": available - demand,
        "encoder_only_cpu_ceiling_rps": available / cpu_seconds_per_request,
        "endpoint_capacity_established": False,
    }


async def run(args) -> None:
    spec = DenseModel.model_validate_json(args.spec.read_text())
    language_queries = []
    for path in sorted(args.data.glob("xquad.*.json")):
        _, questions = load_dataset(path)
        language_queries.append([question["query"] for question in questions])
    queries = [query for group in zip(*language_queries, strict=True) for query in group]
    if not queries:
        raise ValueError("No multilingual queries found")
    model = load_dense(spec)
    try:
        for query in queries[:3]:
            await model.embed_query(query)

        async def encode(index: int) -> object:
            return await model.embed_query(queries[index % len(queries)])

        result = await fixed_arrivals(
            encode,
            rate=args.rate,
            requests=args.requests,
            max_pending=args.max_pending,
            deadline_seconds=args.timeout,
        )
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result.update(
            spec=spec.model_dump(),
            encoder=model.fingerprint(),
            platform=platform.platform(),
            peak_process_rss_mib=rss / (1024**2 if platform.system() == "Darwin" else 1024),
            benchmark_sha256=file_sha256(Path(__file__)),
            data_manifest_sha256=file_sha256(args.data / "manifest.json"),
            paid_llm_calls=0,
            limitations=[
                "Encoder only; excludes API, authorization, database, search and OCR.",
                "Shared host; projection assumes the same CPU instruction performance.",
                "20 RPS on 8 vCPU/16 GB requires a separate complete-service load test.",
                "Peak RSS covers this process, not the complete deployment.",
            ],
        )
        if result["cpu_seconds_per_success"] and result["failure_ratio"] == 0:
            result["projection"] = cpu_projection(
                result["cpu_seconds_per_success"],
                rate=20,
                cores=8,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=20)
    parser.add_argument("--requests", type=int, default=1200)
    parser.add_argument("--max-pending", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=2)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
