"""Paired CPU reranking of the same English XQuAD sparse candidate pool.

This measures paragraph ranking, not generated answers or application latency. Keep
query/candidate identities fixed so a reranker cannot receive credit for a wider pool.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import time
from pathlib import Path

from benchmark.cross_encoder import CrossEncoderModel, CrossEncoderReranker
from benchmark.harness import stats
from benchmark.multilingual_dense import metrics, record
from benchmark.multilingual_sparse import SparseCorpus, load_dataset


async def run(args) -> None:
    spec = CrossEncoderModel.model_validate_json(args.spec.read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    source = args.data / "xquad.en.json"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["files"]["en"]["sha256"]:
        raise ValueError("Dataset hash mismatch")
    documents, questions = load_dataset(source)
    if args.limit:
        questions = questions[: args.limit]
    corpus = SparseCorpus(documents)
    model = CrossEncoderReranker(spec)
    before, after, latencies, pairs = [], [], [], []
    try:
        await model.rerank("warmup", ["Warmup paragraph."], top_k=1)
        for question in questions:
            candidates, _ = corpus.retrieve(question["query"], k=args.candidates)
            started = time.perf_counter()
            ranked = await model.rerank(
                question["query"], [documents[i] for i in candidates], top_k=10
            )
            latencies.append((time.perf_counter() - started) * 1000)
            reranked = [candidates[item.index] for item in ranked]
            record(before, question, candidates)
            record(after, question, reranked)
            pairs.append({"id": question["id"], "candidates": candidates, "reranked": reranked})
        result = {
            "dataset": manifest,
            "model": spec.model_dump(),
            "fingerprint": model.fingerprint(),
            "platform": platform.platform(),
            "candidate_count": args.candidates,
            "llm_calls": 0,
            "baseline": metrics(before),
            "reranked": metrics(after),
            "reranker_component_ms": stats(latencies),
            "pairs": pairs,
            "limitations": [
                "English only; this reranker is not validated for multilingual requests.",
                "XQuAD paragraph retrieval, not LoCoMo or answer accuracy.",
                "Timing excludes candidate retrieval, HTTP, database and authorization.",
                "Public data may overlap model training; validate held-out product tasks too.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("reranker_component_ms",)}), flush=True)
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("benchmark/data/xquad"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, choices=range(10, 101), default=20)
    parser.add_argument("--limit", type=int)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
