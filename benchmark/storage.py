"""Archive segment benchmark: real compression ratios + segment sizes on synthetic chat data,
and a GCS storage-class cost model driven by assumed reopen rates.

Prices are *assumptions* recorded in the artifact (us-central1 list prices, USD/GB-month and
per-10k class-A/B ops, retrieval per GB) and must be refreshed against the GCP price list
before any cost decision. The model output is a comparison, not a quote.
"""

from __future__ import annotations

import random
import statistics
import time
from datetime import UTC, datetime, timedelta

from benchmark.common import provenance, write_result
from memory_service.domain.conversation import Message
from memory_service.domain.enums import MessageRole
from memory_service.domain.ids import content_hash
from memory_service.modules.archive.segments import build_segment, plan_segments

WORDS = (
    "revenue ebitda margin restructuring quarter guidance customer churn pipeline forecast "
    "deploy incident rollback latency cache index migration tenant workspace agent memory "
    "the a of to and in for with on is are was were be this that it as at by from"
).split()


def _synthetic_messages(n: int, seed: int = 7) -> list[Message]:
    rng = random.Random(seed)
    base = datetime(2026, 9, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        length = int(rng.lognormvariate(4.2, 0.9))  # median ~66 words, long tail
        content = " ".join(rng.choice(WORDS) for _ in range(max(3, length)))
        out.append(
            Message(
                thread_id="thr_bench",
                session_id="ses_bench",
                turn_id=f"trn_{i // 2}",
                tenant_id="acme",
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                sequence=i + 1,
                content=content,
                content_hash=content_hash(content),
                author_principal="user:u1",
                occurred_at=base + timedelta(seconds=i * 30),
            )
        )
    return out


PRICES_USD = {  # assumptions, per GB-month; retrieval per GB; min storage days
    "STANDARD": {"storage": 0.020, "retrieval": 0.0, "min_days": 0},
    "NEARLINE": {"storage": 0.010, "retrieval": 0.01, "min_days": 30},
    "COLDLINE": {"storage": 0.004, "retrieval": 0.02, "min_days": 90},
    "ARCHIVE": {"storage": 0.0012, "retrieval": 0.05, "min_days": 365},
}


def cost_model(gb_stored: float, monthly_reopen_fraction: float) -> dict[str, float]:
    """Monthly cost per class for `gb_stored` GB when `monthly_reopen_fraction` of it is re-read."""
    out = {}
    for cls, p in PRICES_USD.items():
        out[cls] = round(
            gb_stored * p["storage"] + gb_stored * monthly_reopen_fraction * p["retrieval"], 4
        )
    # autoclass approximation: recent data (assume 40%) in STANDARD, 30% NEARLINE, 20% COLDLINE, 10% ARCHIVE
    out["AUTOCLASS(approx)"] = round(
        gb_stored * (0.4 * 0.020 + 0.3 * 0.010 + 0.2 * 0.004 + 0.1 * 0.0012)
        + gb_stored * monthly_reopen_fraction * (0.3 * 0.01 + 0.2 * 0.02 + 0.1 * 0.05)
        + gb_stored * 0.0025,  # autoclass management fee assumption
        4,
    )
    return out


def main() -> None:
    messages = _synthetic_messages(20_000)
    results = {}
    for level in (3, 6, 12):
        for target_mb in (1, 4, 8):
            t0 = time.perf_counter()
            plans = plan_segments(
                messages, target_compressed_bytes=target_mb * 1024 * 1024, max_messages=5000
            )
            built = [
                build_segment(
                    p.messages, tenant_id="acme", thread_id="thr_bench", shards=64, zstd_level=level
                )
                for p in plans
            ]
            elapsed = time.perf_counter() - t0
            sizes = [len(b.data) for b in built]
            raw = sum(b.raw_bytes for b in built)
            results[f"zstd{level}_target{target_mb}MB"] = {
                "segments": len(built),
                "raw_bytes": raw,
                "compressed_bytes": sum(sizes),
                "ratio": round(raw / max(1, sum(sizes)), 2),
                "segment_bytes_median": int(statistics.median(sizes)),
                "segment_bytes_max": max(sizes),
                "seconds": round(elapsed, 3),
                "mb_per_second": round(raw / 1024 / 1024 / elapsed, 1),
            }
    payload = {
        "provenance": provenance(dataset="synthetic-chat-20k", messages=len(messages)),
        "compression": results,
        "cost_model_usd_per_month": {
            f"{gb}GB_reopen{int(f * 100)}pct": cost_model(gb, f)
            for gb in (100, 1000)
            for f in (0.01, 0.10, 0.50)
        },
        "price_assumptions": PRICES_USD,
        "notes": [
            "Synthetic English chat; real corpora will compress differently. Re-run on a sample of production segments.",
            "Cost figures are a model with list-price assumptions, not measurements.",
            "Recommendation pending real reopen rates: Autoclass unless reopen rate is known to be <1%/month, then explicit Coldline after 90d.",
        ],
    }
    path = write_result("storage.json", payload)
    print(f"wrote {path}")
    for k, v in results.items():
        print(
            f"{k:22s} segments={v['segments']:3d} ratio={v['ratio']:5.2f} median={v['segment_bytes_median']:>9d}B {v['mb_per_second']} MB/s"
        )


if __name__ == "__main__":
    main()
