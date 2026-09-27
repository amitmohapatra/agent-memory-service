"""Full SciFact document retrieval with cached CPU vectors and no generation calls.

This isolates encoder/fusion quality. It does not exercise document parsing, chunking,
authorization, graph expansion or the network, and is not an end-to-end RAG score.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from benchmark.common import file_sha256, local_model_runtime
from benchmark.harness import stats
from benchmark.multilingual_dense import fuse, ranking
from benchmark.multilingual_sparse import SparseCorpus
from benchmark.public.metrics import evaluate_run
from memory_service.adapters.models import embeddings
from memory_service.config.constants import DenseModel


def manifest(spec: DenseModel, data: Path) -> dict:
    root = Path(spec.source)
    files = (
        spec.graph_file or embeddings.DEFAULT_GRAPH_FILE,
        "tokenizer.json",
        "config.json",
        "1_Pooling/config.json",
    )
    return {
        "data_sha256": file_sha256(data),
        "spec": spec.model_dump(),
        "embedding_adapter_sha256": file_sha256(Path(embeddings.__file__)),
        "model_files": {name: file_sha256(root / name) for name in files},
    }


async def vectors(spec: DenseModel, documents: list[str], queries: list[str], path: Path) -> tuple:
    if await asyncio.to_thread(path.exists):
        with np.load(path, allow_pickle=False) as saved:
            matrix, query_vectors = saved["documents"], saved["queries"]
        validate_vectors(matrix, query_vectors, len(documents), len(queries), spec.dimension)
        return matrix, query_vectors, {"reused_vectors": True}
    model = embeddings.load_dense(spec)
    try:
        started = time.perf_counter()
        matrix = np.asarray(await model.embed_documents(documents), dtype=np.float32)
        elapsed = time.perf_counter() - started
        query_vectors, times = [], []
        await model.embed_query(queries[0])
        for query in queries:
            started = time.perf_counter()
            query_vectors.append(await model.embed_query(query))
            times.append((time.perf_counter() - started) * 1000)
        query_matrix = np.asarray(query_vectors, dtype=np.float32)
        validate_vectors(matrix, query_matrix, len(documents), len(queries), spec.dimension)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as output:
            np.savez(output, documents=matrix, queries=query_matrix)
        temporary.replace(path)
        return (
            matrix,
            query_matrix,
            {
                "reused_vectors": False,
                "document_encoding_seconds": elapsed,
                "query_encoding_ms": stats(times),
            },
        )
    finally:
        model.close()


def validate_vectors(documents, queries, document_count, query_count, dimension) -> None:
    for matrix, count in ((documents, document_count), (queries, query_count)):
        if matrix.shape != (count, dimension):
            raise ValueError("Vector shapes do not match the bound corpus and model")
        if not np.isfinite(matrix).all():
            raise ValueError("Vector values must be finite")


async def run(args: argparse.Namespace) -> None:
    data = json.loads(args.data.read_text())
    if len(data["corpus"]) != 5183 or len(data["queries"]) != 300:
        raise ValueError("Use the full SciFact corpus (5,183 documents and 300 queries)")
    spec = DenseModel.model_validate_json(args.spec.read_text())
    identity = manifest(spec, args.data)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ids = [str(doc["id"]) for doc in data["corpus"]]
    corpus_ids = set(ids)
    documents = [f"{doc['title']}\n\n{doc['text']}" for doc in data["corpus"]]
    queries = data["queries"]
    qrels = {str(q["id"]): {str(doc): 1 for doc in q["relevant"]} for q in queries}
    if len(corpus_ids) != len(ids) or len(qrels) != len(queries):
        raise ValueError("Document and query IDs must be unique")
    if any(set(rels) - corpus_ids for rels in qrels.values()):
        raise ValueError("The corpus is missing judged relevant documents")
    matrix, query_vectors, timing = await vectors(
        spec, documents, [q["text"] for q in queries], args.cache / f"{key}.npz"
    )
    lexical = SparseCorpus(documents)
    runs = {name: {} for name in ("dense", "sparse", "hybrid_k1", "hybrid_k60")}
    candidates = []
    for question, vector in zip(queries, query_vectors, strict=True):
        dense = ranking(vector.tolist(), matrix, k=50)
        sparse, _ = lexical.retrieve(question["text"], 50)
        arms = {
            "dense": dense,
            "sparse": sparse,
            "hybrid_k1": fuse(dense, sparse, k=1),
            "hybrid_k60": fuse(dense, sparse, k=60),
        }
        for name, ranked in arms.items():
            runs[name][str(question["id"])] = [ids[i] for i in ranked]
        candidates.append(
            {
                "query_id": str(question["id"]),
                "dense": [ids[i] for i in dense],
                "sparse": [ids[i] for i in sparse],
            }
        )
    result = {
        "dataset": {k: data[k] for k in ("name", "split", "source", "licence")},
        "manifest": identity,
        "platform": platform.platform(),
        "runtime": local_model_runtime(),
        "documents": len(documents),
        "queries": len(queries),
        "llm_calls": 0,
        "complete": True,
        "metrics": {name: evaluate_run(run, qrels, recall_ks=(10,)) for name, run in runs.items()},
        "component_timing": timing,
        "candidates": candidates,
        "limitations": [
            "Full SciFact document-level encoder/fusion screening, not the application RAG pipeline.",
            "One title+abstract per document; this cannot test parent/neighbor expansion or chunk collapse.",
            "No generated answers, no LLM judge, no HTTP/SQL/authorization latency.",
            "Public data may overlap model training; these test-set arms are not a tuning license.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".bench_data/document-vectors"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
