"""CPU dense/hybrid paragraph retrieval on pinned XQuAD, without any LLM calls.

The same-language and cross-language arms share each query encoding. Every query is
encoded separately to measure the production query path, not batch throughput. Exact
dense search over this small corpus excludes network, authorization and database costs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.harness import stats
from benchmark.multilingual_sparse import SparseCorpus, load_dataset
from memory_service.adapters.models.embeddings import load_dense
from memory_service.config.constants import DenseModel


def ranking(vector: list[float], documents: np.ndarray, k: int = 50) -> list[int]:
    scores = documents @ np.asarray(vector, dtype=np.float32)
    # Stable tie-breaking: every arm uses the same deterministic document identities.
    return np.argsort(-scores, kind="stable")[:k].tolist()


def fuse(dense: list[int], sparse: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for arm in (dense, sparse):
        for rank, document in enumerate(arm, 1):
            scores[document] = scores.get(document, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))[:10]


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "questions": len(rows),
        "recall": {
            str(k): sum(r["rank"] is not None and r["rank"] <= k for r in rows) / len(rows)
            for k in (1, 5, 10)
        },
        "mrr_at_10": sum(1 / r["rank"] for r in rows if r["rank"] is not None) / len(rows),
        "rows": rows,
    }


def record(rows: list, question: dict, found: list[int]) -> None:
    top = found[:10]
    rows.append(
        {
            "id": question["id"],
            "rank": top.index(question["gold"]) + 1 if question["gold"] in top else None,
        }
    )


async def run(args: argparse.Namespace) -> None:
    spec = DenseModel.model_validate_json(args.spec.read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    output: dict[str, Any] = {
        "dataset": manifest,
        "spec": spec.model_dump(),
        "hardware": platform.platform(),
        "languages": {},
        "llm_calls": 0,
        "query_limit_per_language": args.limit,
        "caveats": [
            "XQuAD QA repurposed for paragraph retrieval, not generated-answer accuracy.",
            "Exact small-corpus component timing; excludes HTTP, ACL, database and context.",
            "Hybrid uses RRF k=60 and depth 50 per arm, not the full production pipeline.",
            "Models may have seen public benchmark data during training.",
        ],
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (
            previous["spec"] != output["spec"]
            or previous["dataset"] != manifest
            or previous.get("query_limit_per_language", 0) != args.limit
        ):
            raise ValueError("Resume requires the same model specification and dataset")
        output = previous
    requested = args.languages.split(",")
    remaining = [language for language in requested if language not in output["languages"]]
    if not remaining:
        return
    english_path = args.data / "xquad.en.json"
    if hashlib.sha256(english_path.read_bytes()).hexdigest() != manifest["files"]["en"]["sha256"]:
        raise ValueError("English cross-language corpus hash mismatch")
    english, english_questions = load_dataset(args.data / "xquad.en.json")
    english_gold = {q["id"]: q["gold"] for q in english_questions}
    model = load_dense(spec)
    output["encoder"] = model.fingerprint()
    try:
        started = time.perf_counter()
        english_vectors = np.asarray(await model.embed_documents(english), dtype=np.float32)
        english_encoding_ms = (time.perf_counter() - started) * 1000
        english_sparse = SparseCorpus(english)
        for language in remaining:
            path = args.data / f"xquad.{language}.json"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != manifest["files"][language]["sha256"]:
                raise ValueError(f"Dataset hash mismatch: {language}")
            docs, questions = load_dataset(path)
            if args.limit:
                # Fixed, evenly spaced IDs, chosen before any model scores are available.
                indices = np.linspace(0, len(questions) - 1, args.limit, dtype=int)
                questions = [questions[i] for i in indices]
            started = time.perf_counter()
            matrix = (
                english_vectors
                if language == "en"
                else np.asarray(await model.embed_documents(docs), dtype=np.float32)
            )
            indexing_ms = (
                english_encoding_ms if language == "en" else (time.perf_counter() - started) * 1000
            )
            sparse = SparseCorpus(docs)
            arms: dict[str, list] = {
                k: [] for k in ("same_dense", "same_hybrid", "cross_dense", "cross_hybrid")
            }
            latency = []
            await model.embed_query(questions[0]["query"])  # unmeasured warmup
            for q in questions:
                started = time.perf_counter()
                vector = await model.embed_query(q["query"])
                latency.append((time.perf_counter() - started) * 1000)
                dense = ranking(vector, matrix)
                lexical, _ = sparse.retrieve(q["query"], 50)
                record(arms["same_dense"], q, dense)
                record(arms["same_hybrid"], q, fuse(dense, lexical))
                cross = {**q, "gold": english_gold[q["id"]]}
                dense = ranking(vector, english_vectors)
                lexical, _ = english_sparse.retrieve(q["query"], 50)
                record(arms["cross_dense"], cross, dense)
                record(arms["cross_hybrid"], cross, fuse(dense, lexical))
            output["languages"][language] = {
                "documents": len(docs),
                "document_encoding_ms": indexing_ms,
                "warm_query_encoding_ms": stats(latency),
                **{name: metrics(rows) for name, rows in arms.items()},
            }
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            output["peak_process_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
            output["requested_languages"] = requested
            output["complete"] = all(language in output["languages"] for language in requested)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
            print(
                language,
                {k: round(metrics(v)["recall"]["10"], 4) for k, v in arms.items()},
                stats(latency),
                flush=True,
            )
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("benchmark/data/xquad"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--languages", default="en,hi,ar,zh,de,el,es,ro,ru,th,tr,vi")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
