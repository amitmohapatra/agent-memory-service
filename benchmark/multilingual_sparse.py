"""Offline sparse retrieval on XQuAD paragraphs, not generated-answer accuracy.

Run with each checkout's src on PYTHONPATH against the same pinned dataset directory.
No model, LLM, database or external service is contacted. This is a component benchmark;
its timings exclude hybrid retrieval, HTTP, authorization and context assembly.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.harness import stats
from memory_service.adapters.models.sparse import Bm25SparseEncoder


def load_dataset(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    documents: list[str] = []
    questions = []
    for article in json.loads(path.read_text())["data"]:
        for paragraph in article["paragraphs"]:
            document = len(documents)
            documents.append(paragraph["context"])
            for qa in paragraph["qas"]:
                questions.append({"id": qa["id"], "query": qa["question"], "gold": document})
    return documents, questions


class SparseCorpus:
    """Exact postings lookup with Qdrant's documented BM25 IDF expression."""

    def __init__(self, documents: list[str]) -> None:
        self.encoder = Bm25SparseEncoder()
        self.postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for index, vector in enumerate(self.encoder.encode_documents(documents)):
            for term, weight in zip(vector.indices, vector.values, strict=True):
                self.postings[term].append((index, weight))
        self.idf = {
            term: math.log(1 + (len(documents) - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in self.postings.items()
        }

    def retrieve(self, query: str, k: int = 10) -> tuple[list[int], bool]:
        vector = self.encoder.encode_query(query)
        scores: dict[int, float] = defaultdict(float)
        for term, weight in zip(vector.indices, vector.values, strict=True):
            for document, value in self.postings.get(term, ()):
                scores[document] += weight * value * self.idf[term]
        return heapq.nlargest(k, scores, key=lambda i: (scores[i], -i)), not vector.indices


def evaluate(documents: list[str], questions: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    corpus = SparseCorpus(documents)
    indexing_ms = (time.perf_counter() - started) * 1000
    rows = []
    latencies = []
    for question in questions:
        started = time.perf_counter()
        found, empty = corpus.retrieve(question["query"])
        latencies.append((time.perf_counter() - started) * 1000)
        rank = found.index(question["gold"]) + 1 if question["gold"] in found else None
        rows.append({"id": question["id"], "rank": rank, "empty_query": empty})
    count = len(rows)
    return {
        "documents": len(documents),
        "questions": count,
        "recall": {
            str(k): sum(r["rank"] is not None and r["rank"] <= k for r in rows) / count
            for k in (1, 5, 10)
        },
        "mrr_at_10": sum(1 / r["rank"] for r in rows if r["rank"] is not None) / count,
        "empty_queries": sum(r["empty_query"] for r in rows),
        "indexing_ms": round(indexing_ms, 2),
        "query_component_ms": stats(latencies),
        "rows": rows,
    }


def run(data: Path) -> dict[str, Any]:
    manifest = json.loads((data / "manifest.json").read_text())
    english_docs, english_questions = load_dataset(data / "xquad.en.json")
    english_gold = {q["id"]: q["gold"] for q in english_questions}
    languages = {}
    for language, source in sorted(manifest["files"].items()):
        path = data / f"xquad.{language}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {language}")
        documents, questions = load_dataset(path)
        cross = [{**q, "gold": english_gold[q["id"]]} for q in questions]
        languages[language] = {
            "same_language": evaluate(documents, questions),
            "english_corpus": evaluate(english_docs, cross),
        }
    return {
        "dataset": manifest,
        "encoder": Bm25SparseEncoder().fingerprint(),
        "languages": languages,
        "llm_calls": 0,
        "caveats": [
            "XQuAD QA data repurposed for paragraph retrieval; this is not the official QA metric.",
            "Small closed corpus; no unanswerable questions, no LoCoMo or agent-task score.",
            "Exact sparse component only; no dense retrieval, reranker, network or database.",
            "Lexical retrieval does not translate; English-corpus runs quantify that limitation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("benchmark/data/xquad"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
