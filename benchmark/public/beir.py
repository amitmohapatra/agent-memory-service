"""BEIR subset through the real ingestion, indexing and retrieval path.

Each corpus document becomes one ingested document (title + text, ``text/plain``), every
test query runs through ``RetrievalEngine.retrieve`` and the chunk hits are folded to a
document ranking (a document ranks at its best chunk). nDCG@10 / Recall@20 / Recall@100 are
computed per strategy configuration (baseline hybrid, then each advanced flag), together
with index time and query latency p50/p95.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import text

from benchmark.public.data import BeirDataset
from benchmark.public.metrics import doc_ranking, evaluate_run
from benchmark.retrieval import TABLES, _pct, _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import DependencyUnavailable
from memory_service.modules.jobs.registry import register_handlers

STRATEGIES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "splade": {"splade": True},
    "minicoil": {"minicoil": True},
    "colbert": {"colbert": True},
    "pageindex": {"pageindex": True},
    "raptor": {"raptor": True},
    "graph_ppr": {"graph_ppr": True},
    "late_chunking": {"late_chunking": True},
}
NDCG_KS = (10,)
RECALL_KS = (20, 100)
MAX_K = max(*NDCG_KS, *RECALL_KS)


def strategy_settings(base: Settings, flags: dict[str, Any]) -> Settings:
    """Apply one strategy's flags and widen the candidate pool to the deepest cutoff so
    Recall@100 measures ranking, not the production ``fused_k`` prune."""
    data = base.model_dump()
    retrieval = {**data["retrieval"], **flags}
    retrieval["prefetch_k"] = max(int(retrieval["prefetch_k"]), MAX_K)
    retrieval["fused_k"] = max(int(retrieval["fused_k"]), MAX_K)
    data["retrieval"] = retrieval
    return Settings(**data)


async def _ingest(
    container: Any, dataset: BeirDataset, ctx: MemoryExecutionContext, *, batch: int
) -> dict[str, str]:
    uow_factory = container.services["uow_factory"]
    ingestion = container.services["ingestion"]
    mapping: dict[str, str] = {}
    for n, (doc_id, (title, body)) in enumerate(dataset.corpus.items(), start=1):
        content = f"{title}\n\n{body}".strip() if title else body
        async with uow_factory() as uow:
            ack = await ingestion.accept_file(
                uow,
                ctx,
                filename=f"{doc_id}.txt",
                media_type="text/plain",
                data=content.encode("utf-8"),
                title=title or doc_id,
                custom_metadata={"beir_id": doc_id},
            )
            await uow.commit()
        mapping[ack.document_id] = doc_id
        if n % batch == 0:
            await container.tasks.drain()
    await container.tasks.drain()
    await container.tasks.drain()
    return mapping


async def run_strategy(
    dataset: BeirDataset, name: str, flags: dict[str, Any], *, batch: int = 200
) -> dict[str, Any]:
    settings = strategy_settings(_settings(), flags)
    try:
        container = await build_container(settings, __version__)
    except (DependencyUnavailable, NotImplementedError) as exc:
        return {"skipped": f"{type(exc).__name__}: {exc}", "flags": flags}
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
        register_handlers(container)
        ctx = MemoryExecutionContext(tenant_id="beir", user_id="bench", workspace_id=dataset.name)
        t0 = time.perf_counter()
        mapping = await _ingest(container, dataset, ctx, batch=batch)
        index_seconds = round(time.perf_counter() - t0, 2)
        engine = container.services["retrieval"]
        run: dict[str, list[str]] = {}
        latencies: list[float] = []
        for qid in dataset.scored_queries:
            t = time.perf_counter()
            res = await engine.retrieve(ctx, dataset.queries[qid], limit=MAX_K, kinds=("chunk",))
            latencies.append((time.perf_counter() - t) * 1000)
            run[qid] = doc_ranking(
                mapping.get(str(c.payload.get("document_id")))
                for c in res.candidates
                if c.kind == "chunk" and c.expansion_edge is None
            )
        metrics = evaluate_run(run, dataset.qrels, ndcg_ks=NDCG_KS, recall_ks=RECALL_KS)
        embedding_fp = container.embedding.fingerprint()
        return {
            "flags": flags,
            "metrics": metrics,
            "index_seconds": index_seconds,
            "documents": len(mapping),
            "latency_ms": {
                "p50": _pct(latencies, 50),
                "p95": _pct(latencies, 95),
                "samples": len(latencies),
            },
            "retrieval": {
                "prefetch_k": settings.retrieval.prefetch_k,
                "fused_k": settings.retrieval.fused_k,
                "rerank": settings.retrieval.rerank,
            },
            "providers": {
                "embedding": embedding_fp,
                "sparse": container.sparse.fingerprint(),
                "reranker": engine.reranker.fingerprint() if engine.reranker else None,
                "late_interaction": container.late_interaction.fingerprint()
                if container.late_interaction
                else None,
                "embedding_representative": not embedding_fp.startswith("hash-"),
            },
        }
    finally:
        await container.close()


async def run_dataset(
    dataset: BeirDataset, strategies: list[str], *, batch: int = 200
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name in strategies:
        rows[name] = await run_strategy(dataset, name, STRATEGIES[name], batch=batch)
    return {
        "source": dataset.source,
        "documents": len(dataset.corpus),
        "queries": len(dataset.scored_queries),
        "qrels": sum(len(v) for v in dataset.qrels.values()),
        "strategies": rows,
    }


def format_table(datasets: dict[str, Any]) -> str:
    lines = [
        f"{'dataset':10} {'strategy':14} {'nDCG@10':>8} {'R@20':>7} {'R@100':>7} "
        f"{'p95 ms':>8} {'index s':>8}"
    ]
    for dname, block in datasets.items():
        for sname, row in block["strategies"].items():
            if "skipped" in row:
                lines.append(f"{dname:10} {sname:14} skipped: {row['skipped'][:70]}")
                continue
            m = row["metrics"]
            lines.append(
                f"{dname:10} {sname:14} {m.get('ndcg@10', 0):8.4f} {m.get('recall@20', 0):7.4f} "
                f"{m.get('recall@100', 0):7.4f} {row['latency_ms']['p95']:8.1f} "
                f"{row['index_seconds']:8.1f}"
            )
    return "\n".join(lines)
