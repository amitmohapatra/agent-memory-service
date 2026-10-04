"""SciFact and XQuAD through the runtime retrieval path: named vectors, script-pruned
prefetch and the store's own fusion, on the frozen encoders.

The component screens that selected the ensemble fused cached candidate lists offline
(``benchmark/results/scifact_qdrant_ensemble_screen.json``, 0.7557 nDCG@10 / 0.8926 R@10;
``multilingual_dense_bekko.json``, 0.9883 same-language R@10). This is the same corpus and
the same metrics through what ships: ``DenseSpaces`` encodes, the indexer's cache and
records carry both vectors and the script tag, ``QdrantSearchStore.search_hybrid`` fuses,
and the query's script decides which spaces are searched. Two arms share one collection:
``english`` searches ``dense_en`` + ``bm25`` (the arm every earlier number was measured
with), ``ensemble`` searches every space the script calls for. No parsing, chunking, graph
or reader is exercised - this is retrieval, not an end-to-end score.

    python -m benchmark.runtime_retrieval --suite scifact --output benchmark/results/phase7/runtime_scifact.json
    python -m benchmark.runtime_retrieval --suite xquad --output benchmark/results/phase7/runtime_xquad.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.common import file_sha256, provenance
from benchmark.env import bench_overrides
from benchmark.harness import stats
from benchmark.multilingual_sparse import load_dataset
from benchmark.public.metrics import evaluate_run
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.constants import RETRIEVAL
from memory_service.domain.ids import content_hash
from memory_service.domain.script import detect_script
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.rag.indexer import KNOWLEDGE
from memory_service.ports.search import SearchFilter, SearchRecord, VectorName

SCIFACT = Path("benchmark/data/scifact.json")
XQUAD = Path("benchmark/data/xquad")
LANGUAGES = ("en", "hi", "ar", "zh", "de", "el", "es", "ro", "ru", "th", "tr", "vi")
#: the depth the offline screens fused at, and the cut they scored
PREFETCH = 50
CUT = 10
#: the M2 gates (docs/history/PLAN-MULTILINGUAL-PLATFORM-2026-09-28.md, section 4)
SCIFACT_GATE = {"ndcg@10": 0.7557, "recall@10": 0.8926}
XQUAD_GATE = {"mean_recall@10": 0.98}
INDEX_BATCH = 64


class Arm:
    ENGLISH = "english"
    ENSEMBLE = "ensemble"


@dataclass
class Timing:
    encode_ms: list[float] = field(default_factory=list)
    search_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"encode_ms": stats(self.encode_ms), "search_ms": stats(self.search_ms)}


def tenant_keys(tenant: str) -> list[str]:
    return [f"tenant:{tenant}"]


async def index_corpus(
    container: Any, tenant: str, ids: Sequence[str], texts: Sequence[str]
) -> float:
    """Every document as one record under ``tenant``: every vector indexing writes for a
    chunk (both dense spaces, BM25, the late-interaction tokens) and the script tag."""
    indexer = container.services["indexer"]
    await indexer.ensure_collections()
    collection = indexer.collection(KNOWLEDGE)
    await container.search.delete_by_filter(collection, SearchFilter(tenant_id=tenant))
    started = time.perf_counter()
    for start in range(0, len(texts), INDEX_BATCH):
        batch_ids = ids[start : start + INDEX_BATCH]
        batch = texts[start : start + INDEX_BATCH]
        dense = await indexer.embed_cached(batch, [content_hash(t) + ":rt" for t in batch])
        sparse = container.sparse.encode_documents(batch)
        # the late-interaction vectors indexing writes for every chunk (ADR 0025)
        late = await indexer.embed_late(batch)
        await container.search.upsert(
            [
                SearchRecord(
                    record_id=f"{tenant}:{doc_id}",
                    collection=collection,
                    tenant_id=tenant,
                    dense={space: vectors[i] for space, vectors in dense.items()},
                    sparse=sparse[i],
                    late=late[i] if late else None,
                    payload={
                        "kind": "chunk",
                        "visibility_keys": tenant_keys(tenant),
                        "document_id": doc_id,
                        "script": detect_script(batch[i]).value,
                        "text": batch[i][:200],
                    },
                )
                for i, doc_id in enumerate(batch_ids)
            ]
        )
        print(
            f"[runtime] {tenant}: indexed {min(start + INDEX_BATCH, len(texts))}/{len(texts)}",
            file=sys.stderr,
            flush=True,
        )
    return time.perf_counter() - started


async def search(container: Any, tenant: str, query: str, arm: str, timing: Timing) -> list[str]:
    """The fused document ranking for one query under one arm, timed."""
    engine = container.services["retrieval"]
    indexer = engine.indexer
    started = time.perf_counter()
    vectors = await engine._encode(query)
    timing.encode_ms.append((time.perf_counter() - started) * 1000)
    dense = dict(vectors.dense)
    if arm == Arm.ENGLISH:
        dense = (
            {VectorName.DENSE_EN: dense[VectorName.DENSE_EN]}
            if VectorName.DENSE_EN in dense
            else {}
        )
    flt = SearchFilter(
        tenant_id=tenant, must={"kind": "chunk"}, must_any={"visibility_keys": tenant_keys(tenant)}
    )
    started = time.perf_counter()
    hits = await container.search.search_hybrid(
        indexer.collection(KNOWLEDGE),
        dense=dense,
        sparse=vectors.sparse,
        flt=flt,
        limit=CUT,
        prefetch_limit=PREFETCH,
        rrf_k=RETRIEVAL.hybrid_rrf_k,
        weights=RETRIEVAL.hybrid_weights,
        # the late-interaction arm (ADR 0025): what the engine's own hybrid search passes
        late=vectors.late,
    )
    timing.search_ms.append((time.perf_counter() - started) * 1000)
    return [str(hit.payload.get("document_id")) for hit in hits]


async def scifact(container: Any, data_path: Path) -> dict[str, Any]:
    data = json.loads(await asyncio.to_thread(data_path.read_text))
    if len(data["corpus"]) != 5183 or len(data["queries"]) != 300:
        raise ValueError("Use the full SciFact corpus (5,183 documents and 300 queries)")
    ids = [str(doc["id"]) for doc in data["corpus"]]
    texts = [f"{doc['title']}\n\n{doc['text']}" for doc in data["corpus"]]
    qrels = {str(q["id"]): {str(doc): 1 for doc in q["relevant"]} for q in data["queries"]}
    tenant = "bench_scifact"
    index_seconds = await index_corpus(container, tenant, ids, texts)
    arms: dict[str, dict[str, Any]] = {}
    for arm in (Arm.ENGLISH, Arm.ENSEMBLE):
        timing = Timing()
        run: dict[str, list[str]] = {}
        for n, question in enumerate(data["queries"], start=1):
            run[str(question["id"])] = await search(
                container, tenant, question["text"], arm, timing
            )
            if n % 100 == 0:
                print(f"[runtime] scifact {arm}: {n}/300", file=sys.stderr, flush=True)
        arms[arm] = {"metrics": evaluate_run(run, qrels, recall_ks=(10,)), **timing.as_dict()}
    ensemble = arms[Arm.ENSEMBLE]["metrics"]
    return {
        "suite": "scifact",
        "dataset": {k: data[k] for k in ("name", "split", "source", "licence")},
        "dataset_sha256": file_sha256(data_path),
        "documents": len(texts),
        "queries": len(qrels),
        "index_seconds": round(index_seconds, 1),
        "arms": arms,
        "gate": {
            "thresholds": SCIFACT_GATE,
            "measured": {k: ensemble[k] for k in SCIFACT_GATE},
            "pass": all(ensemble[k] >= v for k, v in SCIFACT_GATE.items()),
        },
    }


def load_xquad(data_dir: Path, language: str) -> tuple[list[str], list[dict[str, Any]], str]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    path = data_dir / f"xquad.{language}.json"
    digest = file_sha256(path)
    if digest != manifest["files"][language]["sha256"]:
        raise ValueError(f"{path} does not match the pinned manifest")
    documents, questions = load_dataset(path)
    return documents, questions, digest


async def xquad(
    container: Any, data_dir: Path, languages: Sequence[str], limit: int
) -> dict[str, Any]:
    per_language: dict[str, Any] = {}
    english_docs, _, _ = load_xquad(data_dir, "en")
    await index_corpus(
        container, "bench_xquad_en", [str(i) for i in range(len(english_docs))], english_docs
    )
    for language in languages:
        documents, questions, digest = load_xquad(data_dir, language)
        if limit:
            questions = questions[:limit]
        tenant = f"bench_xquad_{language}"
        if language != "en":
            await index_corpus(
                container, tenant, [str(i) for i in range(len(documents))], documents
            )
        arms: dict[str, Any] = {}
        scripts: dict[str, int] = {}
        for arm in (Arm.ENGLISH, Arm.ENSEMBLE):
            timing = Timing()
            same_hits = 0
            cross_hits = 0
            for n, question in enumerate(questions, start=1):
                ranked = await search(container, tenant, question["query"], arm, timing)
                same_hits += str(question["gold"]) in ranked[:CUT]
                cross = await search(container, "bench_xquad_en", question["query"], arm, Timing())
                cross_hits += str(question["gold"]) in cross[:CUT]
                if arm == Arm.ENSEMBLE:
                    script = detect_script(question["query"]).value
                    scripts[script] = scripts.get(script, 0) + 1
                if n % 200 == 0:
                    print(
                        f"[runtime] xquad {language} {arm}: {n}/{len(questions)}",
                        file=sys.stderr,
                        flush=True,
                    )
            arms[arm] = {
                "same_language_recall@10": round(same_hits / len(questions), 4),
                "cross_language_recall@10": round(cross_hits / len(questions), 4),
                **timing.as_dict(),
            }
        per_language[language] = {
            "dataset_sha256": digest,
            "documents": len(documents),
            "questions": len(questions),
            "query_scripts": scripts,
            "arms": arms,
        }
        print(
            f"[runtime] xquad {language}: {json.dumps({a: arms[a]['same_language_recall@10'] for a in arms})}",
            file=sys.stderr,
            flush=True,
        )
    means = {
        arm: round(
            sum(per_language[lang]["arms"][arm]["same_language_recall@10"] for lang in languages)
            / len(languages),
            4,
        )
        for arm in (Arm.ENGLISH, Arm.ENSEMBLE)
    }
    return {
        "suite": "xquad",
        "dataset": json.loads((data_dir / "manifest.json").read_text()),
        "languages": list(languages),
        "per_language": per_language,
        "mean_same_language_recall@10": means,
        "gate": {
            "thresholds": XQUAD_GATE,
            "measured": {"mean_recall@10": means[Arm.ENSEMBLE]},
            "pass": means[Arm.ENSEMBLE] >= XQUAD_GATE["mean_recall@10"],
        },
    }


async def run(args: argparse.Namespace) -> None:
    settings = _settings()
    container = await build_container(
        settings, __version__, overrides=bench_overrides(graph_enrichment="disabled")
    )
    register_handlers(container)
    try:
        spaces = container.dense_spaces
        started = time.perf_counter()
        if args.suite == "scifact":
            body = await scifact(container, args.data or SCIFACT)
        else:
            body = await xquad(container, args.data or XQUAD, args.languages.split(","), args.limit)
        result = {
            **body,
            "encoders": {space.name.value: space.encoder.fingerprint() for space in spaces.spaces},
            "spaces_fingerprint": spaces.fingerprint(),
            "sparse": container.sparse.fingerprint(),
            "retrieval": {
                "hybrid_rrf_k": RETRIEVAL.hybrid_rrf_k,
                "hybrid_weights": RETRIEVAL.hybrid_weights,
            },
            "source_sha256": hashlib.sha256(
                await asyncio.to_thread(Path(__file__).read_bytes)
            ).hexdigest(),
            "llm_calls": 0,
            "complete": True,
            "total_seconds": round(time.perf_counter() - started, 1),
            "provenance": provenance(llm={"enabled": False, "provider": "disabled"}),
            "limitations": [
                "Retrieval through the shipped store path on one record per document; no parsing, chunking, graph expansion or reader.",
                "Test-set measurement on public data that selected the models; not a held-out estimate.",
                "Latencies are sequential single-query timings on a shared host; read host_load beside them.",
            ],
        }
    finally:
        await container.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "arms": {
                    k: v.get("metrics", v.get("same_language_recall@10"))
                    for k, v in body.get("arms", {}).items()
                }
                or body.get("mean_same_language_recall@10"),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("scifact", "xquad"), required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--limit", type=int, default=0, help="questions per language (0 = all)")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
