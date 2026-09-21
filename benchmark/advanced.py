"""Advanced retrieval benchmark (M10): baseline vs each benchmark-gated strategy on the
critical gates (Recall@20, Evidence-Group Recall, evidence-complete rate) and latency,
with a verdict per strategy. Strategies whose model weights are absent are recorded as
``skipped`` with the reason — never as a number.

    uv run python -m benchmark.advanced --copies 5 --queries 20

Verdict rule (the adoption gate): a strategy is ``adoptable`` only when it does not lower any
critical gate below the baseline AND its recall p95 stays within ``budgets.recall_p95_ms``.
Anything else is ``rejected`` (with the numbers) or ``skipped``.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

from benchmark.common import provenance, reset_store, write_result
from benchmark.retrieval import FIXTURES, GOLDEN, _pct, _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import DependencyUnavailable
from memory_service.modules.evaluation.golden import (
    GoldenSet,
    RetrievedChunk,
    evaluate_question,
    summarize,
)
from memory_service.modules.jobs.registry import register_handlers

STRATEGIES: dict[str, dict[str, Any]] = {
    "baseline": {},
    # Seven strategies used to sit here — pageindex, raptor, graph_ppr, colbert, minicoil,
    # late_chunking and graphrag_global. They were removed rather than left off: each named a
    # capability something already-on provides, and tests/eval/test_capability_coverage.py
    # demonstrates each capability surviving without them. `splade` is the one genuinely
    # uncovered experiment left, so this benchmark is now baseline against it.
    "splade": {"splade": True},
}
_NOT_IMPLEMENTED: dict[str, str] = {}


def _with_flags(settings: Settings, flags: dict[str, Any]) -> Settings:
    data = settings.model_dump()
    data["retrieval"] = {**data["retrieval"], **flags}
    # model-backed flags: the wiring raises DependencyUnavailable when weights are absent
    return Settings(**data)


async def _corpus(container, golden: GoldenSet, copies: int, ctx) -> dict[str, str]:
    register_handlers(container)
    uow_factory = container.services["uow_factory"]
    ingestion = container.services["ingestion"]
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
            aliases[ack.document_id] = alias
    await container.tasks.drain()
    await container.tasks.drain()
    return aliases


async def run_strategy(
    name: str, flags: dict[str, Any], *, copies: int, queries: int
) -> dict[str, Any]:
    if name in _NOT_IMPLEMENTED:
        return {"skipped": _NOT_IMPLEMENTED[name]}
    settings = _with_flags(_settings(), flags)
    try:
        container = await build_container(settings, __version__)
    except (DependencyUnavailable, NotImplementedError) as exc:
        return {"skipped": f"{type(exc).__name__}: {exc}"}
    try:
        # both stores: the vector store is a separate server and a SQL TRUNCATE
        # leaves its vectors behind for the next run to retrieve
        await reset_store(container, "acme")
        golden = GoldenSet.load(GOLDEN)
        ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
        t0 = time.perf_counter()
        aliases = await _corpus(container, golden, copies, ctx)
        index_seconds = round(time.perf_counter() - t0, 2)
        engine = container.services["retrieval"]
        k = settings.evaluation.critical_recall_k
        results, statuses, latencies = [], {}, []
        for q in golden.questions:
            t = time.perf_counter()
            res = await engine.retrieve(ctx, q.query, limit=k)
            latencies.append((time.perf_counter() - t) * 1000)
            statuses[q.id] = str((res.diagnostics.get("evidence") or {}).get("status"))
            retrieved = [
                RetrievedChunk(
                    document_alias=aliases.get(str(c.payload.get("document_id"))),
                    page=c.payload.get("page"),
                    text=c.text,
                )
                for c in res.candidates
                if c.kind == "chunk"
            ]
            results.append(
                evaluate_question(q, retrieved, k=k, observed_type=res.routed.query_type.value)
            )
        summary = summarize(results, k=k)
        summary.pop("per_question", None)
        critical = [q.id for q in golden.questions if q.critical]
        summary["critical_evidence_complete_rate"] = round(
            sum(1 for i in critical if statuses[i] == "COMPLETE") / len(critical), 4
        )
        qs = [q.query for q in golden.questions]
        for i in range(max(0, queries - len(qs))):
            t = time.perf_counter()
            await engine.retrieve(ctx, qs[i % len(qs)], limit=k)
            latencies.append((time.perf_counter() - t) * 1000)
        return {
            "flags": flags,
            "quality": summary,
            "latency_ms": {
                "recall_p50": _pct(latencies, 50),
                "recall_p95": _pct(latencies, 95),
                "samples": len(latencies),
            },
            "index_seconds": index_seconds,
            "providers": {
                "embedding": container.embedding.fingerprint(),
                "sparse": container.sparse.fingerprint(),
                "representative": not container.embedding.fingerprint().startswith("hash-"),
            },
        }
    finally:
        await container.close()


def verdict(row: dict[str, Any], baseline: dict[str, Any], budget_ms: float) -> str:
    if "skipped" in row:
        return "skipped"
    q, b = row["quality"], baseline["quality"]
    for key in (
        "critical_recall_at_k",
        "critical_evidence_group_recall",
        "critical_evidence_complete_rate",
    ):
        if q[key] < b[key]:
            return f"rejected: {key} {q[key]} < baseline {b[key]}"
    if row["latency_ms"]["recall_p95"] > budget_ms:
        return f"rejected: recall p95 {row['latency_ms']['recall_p95']}ms > budget {budget_ms}ms"
    return "adoptable"


async def run(copies: int, queries: int) -> dict[str, Any]:
    settings = _settings()
    rows: dict[str, Any] = {}
    for name, flags in STRATEGIES.items():
        rows[name] = await run_strategy(name, flags, copies=copies, queries=queries)
    baseline = rows["baseline"]
    for name, row in rows.items():
        row["verdict"] = (
            "baseline"
            if name == "baseline"
            else verdict(row, baseline, settings.budgets.recall_p95_ms)
        )
    return {
        "corpus": {"copies": copies, "documents": copies * 2},
        "strategies": rows,
        "budget_recall_p95_ms": settings.budgets.recall_p95_ms,
        "note": (
            "Quality numbers come from the deterministic hash embedding unless "
            "providers.representative is true; verdicts are relative to the baseline run in "
            "the same environment. Model-backed strategies are skipped without local weights."
        ),
        "provenance": provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=5)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--out", default="advanced_retrieval.json")
    args = parser.parse_args()
    payload = asyncio.run(run(args.copies, args.queries))
    path = write_result(args.out, payload)
    print(f"wrote {path}")
    for name, row in payload["strategies"].items():
        if "skipped" in row:
            print(f"{name:18} skipped: {row['skipped'][:90]}")
        else:
            q = row["quality"]
            print(
                f"{name:18} R@20={q['critical_recall_at_k']} EGR={q['critical_evidence_group_recall']} "
                f"complete={q['critical_evidence_complete_rate']} p95={row['latency_ms']['recall_p95']}ms "
                f"-> {row['verdict']}"
            )


if __name__ == "__main__":
    main()
