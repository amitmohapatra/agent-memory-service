"""Public benchmark harness (TARGET_STACK change 18/19).

    uv run python -m benchmark.public --suite all
    uv run python -m benchmark.public --suite beir --datasets nfcorpus --strategies baseline,splade --max-docs 500 --max-queries 50
    uv run python -m benchmark.public --suite longmemeval --configs native,bifrost --judge-runs 5 --max-questions 100
    uv run python -m benchmark.public --suite locomo --configs native --memory-provider mem0

Results carry provenance and ``representative`` (real embedding weights AND an LLM behind
Bifrost); with the hash embedding or ``models.llm.enabled=false`` every number is a
pipeline smoke result, never a quality claim.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.common import provenance, write_result
from benchmark.public import beir, memory_qa
from benchmark.public.data import (
    BEAM_STATUS,
    BEIR_DATASETS,
    MemoryDataset,
    load_beir,
    load_locomo,
    load_longmemeval,
)
from benchmark.retrieval import _settings

SUITES = ("beir", "longmemeval", "locomo")


def representative(settings: Any) -> dict[str, Any]:
    embedding_ok = settings.models.embedding.provider != "hash"
    llm_ok = bool(settings.models.llm.enabled)
    reasons = []
    if not embedding_ok:
        reasons.append("embedding provider is the hash stand-in")
    if not llm_ok:
        reasons.append("models.llm.enabled=false")
    return {
        "representative": embedding_ok and llm_ok,
        "embedding_representative": embedding_ok,
        "llm_enabled": llm_ok,
        "reasons": reasons,
    }


async def run_beir(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    datasets: dict[str, Any] = {}
    for name in args.datasets:
        data = load_beir(name)
        full = {"documents": len(data.corpus), "queries": len(data.scored_queries)}
        if args.max_docs or args.max_queries:
            data = data.subset(max_docs=args.max_docs, max_queries=args.max_queries, seed=args.seed)
        block = await beir.run_dataset(data, args.strategies, batch=args.batch)
        block["full_size"] = full
        datasets[name] = block
    return {
        "benchmark": "beir",
        "metrics": ["ndcg@10", "recall@20", "recall@100"],
        "datasets": datasets,
        "strategies": args.strategies,
        "caps": {"max_docs": args.max_docs, "max_queries": args.max_queries},
        "wall_seconds": round(time.perf_counter() - t0, 1),
        **representative(_settings()),
        "provenance": provenance(),
    }


async def run_memory(args: argparse.Namespace, dataset: MemoryDataset) -> dict[str, Any]:
    t0 = time.perf_counter()
    full = {
        "conversations": len(dataset.conversations),
        "sessions": sum(len(c.sessions) for c in dataset.conversations),
        "questions": len(dataset.questions),
    }
    dataset = dataset.limited(
        max_conversations=args.max_conversations, max_questions=args.max_questions, seed=args.seed
    )
    configs: dict[str, Any] = {}
    for config in args.configs:
        configs[config] = await memory_qa.run_config(
            dataset,
            config,
            judge_runs=args.judge_runs,
            token_budget=args.token_budget,
            keep_answers=args.keep_answers,
        )
    return {
        "benchmark": dataset.name,
        "dataset": {
            "source": dataset.source,
            "size_bytes": dataset.size_bytes,
            "full": full,
            "run": {
                "conversations": len(dataset.conversations),
                "sessions": sum(len(c.sessions) for c in dataset.conversations),
                "questions": len(dataset.questions),
            },
        },
        "configs": configs,
        "judge_runs": args.judge_runs,
        "beam": BEAM_STATUS,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        **representative(_settings()),
        "provenance": provenance(),
    }


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark.public")
    parser.add_argument("--suite", default="all", choices=(*SUITES, "all"))
    parser.add_argument("--datasets", type=_csv, default=list(BEIR_DATASETS))
    parser.add_argument(
        "--strategies", type=_csv, default=list(beir.STRATEGIES), help=",".join(beir.STRATEGIES)
    )
    parser.add_argument("--max-docs", type=int, default=None, help="BEIR corpus cap (smoke runs)")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch", type=int, default=200, help="documents per index drain")
    parser.add_argument("--configs", type=_csv, default=list(memory_qa.NATIVE_CONFIGS))
    parser.add_argument(
        "--memory-provider",
        type=_csv,
        default=[],
        help="add mem0|langmem|cognee configurations (skipped with a reason when unavailable)",
    )
    parser.add_argument("--judge-runs", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=6000)
    parser.add_argument("--keep-answers", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--longmemeval-file", type=Path, default=None)
    parser.add_argument("--locomo-file", type=Path, default=None)
    parser.add_argument("--out-beir", default="public_beir.json")
    parser.add_argument("--out-longmemeval", default="public_longmemeval.json")
    parser.add_argument("--out-locomo", default="public_locomo.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    unknown = [s for s in args.strategies if s not in beir.STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; choose from {list(beir.STRATEGIES)}")
    args.configs = [*args.configs, *[p for p in args.memory_provider if p not in args.configs]]
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    for suite in suites:
        if suite == "beir":
            payload = asyncio.run(run_beir(args))
            path = write_result(args.out_beir, payload)
            print(f"wrote {path}")
            print(beir.format_table(payload["datasets"]))
        else:
            loader = load_longmemeval if suite == "longmemeval" else load_locomo
            file = args.longmemeval_file if suite == "longmemeval" else args.locomo_file
            if not _settings().models.llm.enabled:
                print(f"{suite}: {memory_qa.LLM_DISABLED}")
            payload = asyncio.run(run_memory(args, loader(file)))
            out = args.out_longmemeval if suite == "longmemeval" else args.out_locomo
            path = write_result(out, payload)
            print(f"wrote {path}")
            print(memory_qa.format_table(suite, payload["configs"]))
        print(
            f"representative={payload['representative']}"
            + (f" ({'; '.join(payload['reasons'])})" if payload["reasons"] else "")
        )
