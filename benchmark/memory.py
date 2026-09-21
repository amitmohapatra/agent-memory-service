"""Memory intelligence benchmark: consolidation quality on the labelled pair set for every
configured provider, plus native pipeline throughput/latency on a synthetic observation
stream (PostgreSQL + Qdrant local).

    uv run python -m benchmark.memory --observations 200

External providers (mem0, langmem, cognee) are benchmarked only when
``MEMORY__MODELS__LLM__ENABLED=true`` and the SDK is installed; otherwise they are listed as
``skipped`` with the reason. Results carry provenance and are never presented as
representative of LLM-based providers unless they actually ran.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

from sqlalchemy import text

from benchmark.common import provenance, reset_store, write_result
from benchmark.retrieval import _pct, _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.modules.evaluation.memory_pairs import evaluate_pairs, load_pairs
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.memory.native import NativeMemoryIntelligence

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "tests" / "eval" / "golden" / "memory_pairs.json"

_TEMPLATES = [
    "My timezone is {tz}.",
    "I prefer {pref}.",
    "We decided to use {tech} for the {component}.",
    "Remind me to follow up with {team} by {day}.",
    "The {service} service runs on {platform} and costs {cost} USD per month.",
    "Yesterday the deploy of {service} failed because of {cause}.",
    "What is the status of {service}?",
    "Thanks, that helps!",
]
_FILL = {
    "tz": ["Europe/Berlin", "America/New_York", "Asia/Kolkata", "UTC"],
    "pref": ["concise answers", "code examples", "dark mode", "British English", "tabs"],
    "tech": ["PostgreSQL", "Qdrant", "Dragonfly", "OpenFGA", "Procrastinate"],
    "component": ["canonical store", "retrieval", "cache", "authorization", "queue"],
    "team": ["legal", "finance", "security", "platform"],
    "day": ["Monday", "Friday", "EOD", "next week"],
    "service": ["billing", "search", "ingest", "auth", "archive"],
    "platform": ["Cloud Run", "Kubernetes", "Lambda"],
    "cost": ["400", "1200", "90"],
    "cause": ["a missing migration", "a certificate expiry", "an OOM kill"],
}


def _stream(n: int, seed: int = 11) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = rng.choice(_TEMPLATES)
        out.append(t.format(**{k: rng.choice(v) for k, v in _FILL.items()}))
    return out


async def _provider_quality(settings) -> dict[str, Any]:
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="t1")
    pairs = load_pairs(PAIRS)
    results: dict[str, Any] = {}
    native = NativeMemoryIntelligence(settings.memory_intelligence)
    rep = await evaluate_pairs(native, pairs, ctx)
    rep.pop("per_pair", None)
    results["native"] = rep
    llm_on = settings.models.llm.enabled
    for name, path, cls in (
        ("mem0", "memory_service.adapters.intelligence.mem0_provider", "Mem0MemoryIntelligence"),
        ("langmem", "memory_service.adapters.intelligence.langmem_provider", "LangMemIntelligence"),
    ):
        if not llm_on:
            results[name] = {"skipped": "models.llm.enabled=false (provider requires an LLM)"}
            continue
        try:
            import importlib

            provider = getattr(importlib.import_module(path), cls)(settings)
            rep = await evaluate_pairs(provider, pairs, ctx)
            rep.pop("per_pair", None)
            results[name] = rep
        except Exception as exc:  # noqa: BLE001 - benchmark must report, not crash
            results[name] = {"skipped": f"{type(exc).__name__}: {exc}"}
    return results


async def run(n_observations: int) -> dict[str, Any]:
    settings = _settings()
    quality = await _provider_quality(settings)
    container = await build_container(settings, __version__)
    try:
        # both stores: the vector store is a separate server and a SQL TRUNCATE
        # leaves its vectors behind for the next run to retrieve
        await reset_store(container, "acme")
        register_handlers(container)
        uow_factory = container.services["uow_factory"]
        service = container.services["memory"]
        pipeline = container.services["observation_pipeline"]
        ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
        accept_ms: list[float] = []
        process_ms: list[float] = []
        decisions: dict[str, int] = {}
        t0 = time.perf_counter()
        for content in _stream(n_observations):
            t = time.perf_counter()
            async with uow_factory() as uow:
                ack = await service.submit_observation(
                    uow, ctx, kind=ObservationKind.MESSAGE, content=content
                )
                await uow.commit()
            accept_ms.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            outcomes = await pipeline.run(
                {"tenant_id": "acme", "observation_id": ack.observation_id}
            )
            process_ms.append((time.perf_counter() - t) * 1000)
            for o in outcomes:
                decisions[o.decision.value] = decisions.get(o.decision.value, 0) + 1
        await container.tasks.drain()  # index jobs
        wall = time.perf_counter() - t0
        async with container.database.engine.connect() as conn:
            rows = (await conn.execute(text("SELECT count(*) FROM memories"))).scalar_one()
            current = (
                await conn.execute(
                    text("SELECT count(*) FROM memories WHERE temporal_status = 'CURRENT'")
                )
            ).scalar_one()
        return {
            "quality": quality,
            "throughput": {
                "observations": n_observations,
                "wall_seconds": round(wall, 2),
                "observations_per_second": round(n_observations / wall, 1) if wall else None,
                "accept_p50_ms": _pct(accept_ms, 50),
                "accept_p95_ms": _pct(accept_ms, 95),
                "process_p50_ms": _pct(process_ms, 50),
                "process_p95_ms": _pct(process_ms, 95),
                "process_mean_ms": round(statistics.fmean(process_ms), 2) if process_ms else None,
                "decisions": decisions,
                "memories_rows": rows,
                "memories_current": current,
            },
            "budgets_ms": {"chat_accept_p95": settings.budgets.chat_accept_p95_ms},
            "provider": settings.memory_intelligence.provider,
            "llm_enabled": settings.models.llm.enabled,
            "provenance": provenance(),
        }
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, default=int(os.environ.get("BENCH_OBS", "200")))
    parser.add_argument("--out", default="memory.json")
    args = parser.parse_args()
    payload = asyncio.run(run(args.observations))
    path = write_result(args.out, payload)
    q = payload["quality"]["native"]
    t = payload["throughput"]
    print(f"wrote {path}")
    print(
        f"native false_merge_rate={q['false_merge_rate']} dedup_recall={q['dedup_recall']} | "
        f"{t['observations']} obs in {t['wall_seconds']}s ({t['observations_per_second']}/s), "
        f"accept p95={t['accept_p95_ms']}ms process p95={t['process_p95_ms']}ms, "
        f"decisions={t['decisions']}"
    )


if __name__ == "__main__":
    main()
