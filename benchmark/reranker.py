"""Reranker benchmark: ``cross-encoder/ms-marco-MiniLM-L6-v2`` (sentence-transformers and ONNX)
vs the lexical stand-in vs no reranker, each with the bounded ``candidate_k`` in 15 / 20 / 25,
on the golden set at ``final_k`` through the real retrieval engine.

    uv run python -m benchmark.reranker
    uv run python -m benchmark.reranker --stand-in   # lexical / disabled only

The corpus is indexed once (the embedding does not change between candidates); each
candidate swaps the engine's reranker and ``rerank_k``. The cross-encoder is loaded from
``$MEMORY_MODELS_DIR/ms-marco-MiniLM-L6-v2``; a provider that cannot be loaded is recorded as
``{"skipped": "<ExceptionType>: reason"}`` for every candidate_k.

Verdict rule: the default is the cheapest ``candidate_k`` (then the lowest rerank p95) among the
candidates whose critical Recall@k / EGR pair equals the best observed.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections.abc import Sequence
from typing import Any

from sqlalchemy import text

from benchmark.advanced import _corpus
from benchmark.common import provenance, write_result
from benchmark.embedding import (
    bench_settings,
    golden_quality,
    models_dir,
    pick_default,
    reset_index,
    stand_in_base,
    with_models,
)
from benchmark.env import bench_overrides
from benchmark.evaluation.golden import GoldenSet
from benchmark.retrieval import GOLDEN, TABLES, _pct
from memory_service.__about__ import __version__
from memory_service.adapters.models.rerankers import CrossEncoderReranker, LexicalReranker
from memory_service.application.container import build_container
from memory_service.config.settings import RerankerSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo, Reranker, RerankResult

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_DIR = "ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_PROVIDERS = ("sentence_transformers", "onnx")
STAND_IN_PROVIDERS = ("lexical", "disabled")
PROVIDERS = (*CROSS_ENCODER_PROVIDERS, *STAND_IN_PROVIDERS)
CANDIDATE_KS = (15, 20, 25)


def candidate_name(provider: str, candidate_k: int) -> str:
    return f"{provider}@k{candidate_k}"


def candidates(*, stand_in: bool = False) -> dict[str, RerankerSettings]:
    out: dict[str, RerankerSettings] = {}
    for provider in STAND_IN_PROVIDERS if stand_in else PROVIDERS:
        for candidate_k in CANDIDATE_KS:
            out[candidate_name(provider, candidate_k)] = RerankerSettings(
                provider=provider,  # type: ignore[arg-type]
                model=MODEL,
                model_path=str(models_dir() / MODEL_DIR)
                if provider in CROSS_ENCODER_PROVIDERS
                else None,
                candidate_k=candidate_k,
            )
    return out


def load_reranker(cfg: RerankerSettings) -> Reranker | None:
    if cfg.provider == "disabled":
        return None
    if cfg.provider == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(cfg)


def reranker_cost(row: dict[str, Any]) -> tuple[float, ...]:
    return (float(row["candidate_k"]), float(row["rerank_ms"]["p95"]))


class TimedReranker:
    """Delegates to the real reranker and records the wall-clock of every call."""

    info: ProviderInfo

    def __init__(self, inner: Reranker) -> None:
        self.inner = inner
        self.info = inner.info
        self.ms: list[float] = []

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int
    ) -> list[RerankResult]:
        t = time.perf_counter()
        out = await self.inner.rerank(query, documents, top_k=top_k)
        self.ms.append((time.perf_counter() - t) * 1000)
        return out

    def fingerprint(self) -> str:
        return self.inner.fingerprint()


def _timing(ms: list[float]) -> dict[str, Any]:
    return {
        "p50": _pct(ms, 50),
        "p95": _pct(ms, 95),
        "mean": round(statistics.fmean(ms), 2) if ms else 0.0,
        "samples": len(ms),
    }


async def run(*, copies: int, stand_in: bool = False) -> dict[str, Any]:
    settings = stand_in_base(bench_settings()) if stand_in else bench_settings()
    settings = with_models(settings, reranker=RerankerSettings(provider="disabled"))
    container = await build_container(settings, __version__, overrides=bench_overrides())
    engine = container.services["retrieval"]
    original = (engine.reranker, engine.rerank_k)
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
        await reset_index(container)
        golden = GoldenSet.load(GOLDEN)
        ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
        t0 = time.perf_counter()
        aliases = await _corpus(container, golden, copies, ctx)
        index_seconds = round(time.perf_counter() - t0, 2)
        final_k = settings.retrieval.final_k
        embedding_fp = container.embedding.fingerprint()
        emb_representative = not embedding_fp.startswith("hash-")

        rows: dict[str, dict[str, Any]] = {}
        loaded: dict[str, tuple[Reranker | None, float]] = {}
        errors: dict[str, str] = {}
        for name, cfg in candidates(stand_in=stand_in).items():
            head = {
                "provider": cfg.provider,
                "candidate_k": cfg.candidate_k,
                "model": cfg.model if cfg.model_path else None,
                "model_path": cfg.model_path,
            }
            if cfg.provider in errors:
                rows[name] = {**head, "skipped": errors[cfg.provider]}
                continue
            if cfg.provider not in loaded:
                t0 = time.perf_counter()
                try:
                    reranker = load_reranker(cfg)
                except Exception as exc:  # noqa: BLE001 - a benchmark reports, it does not crash
                    errors[cfg.provider] = f"{type(exc).__name__}: {exc}"
                    rows[name] = {**head, "skipped": errors[cfg.provider]}
                    continue
                loaded[cfg.provider] = (reranker, round(time.perf_counter() - t0, 2))
            reranker, load_seconds = loaded[cfg.provider]
            timed = TimedReranker(reranker) if reranker is not None else None
            engine.reranker = timed
            engine.rerank_k = cfg.candidate_k
            quality, retrieve_ms = await golden_quality(engine, golden, aliases, ctx, k=final_k)
            rows[name] = {
                **head,
                "fingerprint": timed.fingerprint() if timed else None,
                "load_seconds": load_seconds,
                "final_k": final_k,
                "quality": quality,
                "rerank_ms": _timing(timed.ms if timed else []),
                "retrieve_ms": _timing(retrieve_ms),
                "representative": emb_representative and cfg.provider in CROSS_ENCODER_PROVIDERS,
            }
        return {
            "corpus": {
                "copies": copies,
                "documents": copies * len(golden.documents),
                "index_seconds": index_seconds,
            },
            "mode": {"stand_in": stand_in},
            "final_k": final_k,
            "candidate_ks": list(CANDIDATE_KS),
            "candidates": rows,
            "verdict": pick_default(rows, reranker_cost),
            "providers": {
                "embedding": embedding_fp,
                "sparse": container.sparse.fingerprint(),
                "search": type(container.search).__name__,
                "representative": emb_representative,
            },
            "note": (
                "One indexed corpus; each candidate swaps the engine's reranker and rerank_k. "
                "rerank_ms is the reranker call alone, retrieve_ms the whole retrieve(). A row "
                "is representative only with Granite embeddings and the cross-encoder."
            ),
            "provenance": provenance(),
        }
    finally:
        engine.reranker, engine.rerank_k = original
        await container.close()


def print_table(rows: dict[str, dict[str, Any]], verdict: dict[str, Any]) -> None:
    print(
        f"{'candidate':28} {'load_s':>7} {'rerank_p95ms':>12} {'retrieve_p95ms':>14} "
        f"{'R@k':>6} {'EGR':>6}"
    )
    for name, row in rows.items():
        if "skipped" in row:
            print(f"{name:28} skipped: {row['skipped'][:100]}")
            continue
        q = row["quality"]
        print(
            f"{name:28} {row['load_seconds']:>7} {row['rerank_ms']['p95']:>12} "
            f"{row['retrieve_ms']['p95']:>14} {q['critical_recall_at_k']:>6} "
            f"{q['critical_evidence_group_recall']:>6}"
        )
    print(f"default -> {verdict['default']} ({verdict['reason']})")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=5)
    parser.add_argument(
        "--stand-in", action="store_true", help="lexical / disabled only (representative=false)"
    )
    parser.add_argument("--out", default="reranker.json")
    args = parser.parse_args(argv)
    try:
        payload = asyncio.run(run(copies=args.copies, stand_in=args.stand_in))
    except DependencyUnavailable as exc:
        raise SystemExit(
            f"{exc}\nthe environment's embedding indexes the corpus; without the [models] "
            "extra run with --stand-in (hash embedding, representative=false)"
        ) from exc
    path = write_result(args.out, payload)
    print(f"wrote {path} representative={payload['providers']['representative']}")
    print_table(payload["candidates"], payload["verdict"])


if __name__ == "__main__":
    main()
