"""Retrieval quality on a corpus we did not write.

Every gate in ``benchmark/results`` scores 1.0, and all of them are measured against fixtures
authored in this repo alongside the implementation. That measures self-consistency: the
fixtures encode current behaviour, so they pass by construction. A number that means anything
has to come from data nobody here chose.

This indexes BEIR SciFact through the real ingestion pipeline and asks the real retrieval
engine the benchmark's own 300 test queries, scoring recall@k and nDCG@k against its qrels.

    uv run python -m benchmark.external_retrieval --limit 200      # plumbing check
    make bench-external                                            # the full corpus, real models

The embedding provider comes from the environment like every other benchmark, and the result
records it. With the ``hash`` stand-in the scores are noise by construction — the result is
written with ``representative: false`` so nobody quotes it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

from sqlalchemy import text

from benchmark.common import provenance, reset_store, write_result
from benchmark.env import bench_overrides
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.jobs.registry import register_handlers

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark" / "data" / "scifact.json"
#: A tenant of this benchmark's own.
#:
#: The vector store is shared across benchmarks even when their SQL databases are not, and
#: records are partitioned by tenant. With every harness using "bench", two runs could delete
#: each other's vectors and retrieve each other's documents — which happened, concurrently,
#: and would have been invisible in the result files.
TENANT = os.environ.get("BENCH_TENANT", "bench_docs")
#: providers whose output carries no semantic signal; scores from them are not quality numbers
STANDIN_EMBEDDINGS = {"hash"}


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if doc in relevant else 0.0 for doc in retrieved[:k]]
    ideal = [1.0] * min(len(relevant), k)
    best = _dcg(ideal)
    return _dcg(gains) / best if best else 0.0


async def run(
    limit: int | None,
    k: int,
    *,
    ablate: dict[str, bool] | None = None,
    reuse_index: bool = False,
    batch: int = 100,
) -> dict:
    if not DATASET.is_file():
        raise SystemExit(
            f"{DATASET} is missing — run `make bench-external-prepare` to download it "
            "(it is git-ignored: benchmark data is not vendored)"
        )
    data = json.loads(DATASET.read_text())
    corpus = data["corpus"][:limit] if limit else data["corpus"]
    indexed_ids = {doc["id"] for doc in corpus}
    # A query whose relevant documents were cut by --limit cannot be answered, and keeping it
    # would report a low score for a corpus that never held the answer.
    queries = [q for q in data["queries"] if set(q["relevant"]) & indexed_ids]

    settings = _settings()
    if ablate:
        # Same switch as the LoCoMo harness. LoCoMo exercises conversational *memory*
        # retrieval; this exercises document RAG over a real corpus with real relevance
        # judgements, which is the case rerankers are actually published on. A component can
        # earn its cost on one and not the other, and that is a configuration answer rather
        # than a delete-it answer — so both have to be measured before either is decided.
        settings = settings.model_copy(
            update={"retrieval": settings.retrieval.model_copy(update=ablate)}
        )
    embedding_provider = settings.models.embedding.provider
    container = await build_container(settings, __version__, overrides=bench_overrides())
    try:
        if not reuse_index:
            cleared = await reset_store(container, TENANT)
            print(f"[scifact] cleared vectors: {cleared}", file=sys.stderr, flush=True)
        register_handlers(container)
        uow_factory = container.services["uow_factory"]
        ingestion = container.services["ingestion"]
        engine = container.services["retrieval"]
        ctx = MemoryExecutionContext(tenant_id=TENANT, user_id="bench", workspace_id="ws")

        started = time.perf_counter()
        document_to_corpus: dict[str, str] = {}

        # An ablation runs the same corpus twice and changes only what happens at query time,
        # so the second arm re-paid six hours of identical indexing for nothing. With
        # --reuse-index it reads the mapping back out of the store instead. The filename is
        # "<corpus id>.md" by construction above, which is what makes this recoverable.
        if reuse_index:
            async with container.database.engine.connect() as conn:
                rows = await conn.execute(
                    text("SELECT document_id, filename FROM documents WHERE tenant_id = :t"),
                    {"t": TENANT},
                )
                document_to_corpus = {doc_id: Path(name).stem for doc_id, name in rows.fetchall()}
            missing = indexed_ids - set(document_to_corpus.values())
            if missing:
                raise SystemExit(
                    f"--reuse-index found {len(document_to_corpus)} documents but {len(missing)} "
                    "of the corpus is not among them; index the corpus first"
                )
            index_seconds = 0.0
        if not reuse_index:
            for n, doc in enumerate(corpus, start=1):
                body = f"# {doc['title']}\n\n{doc['text']}".encode()
                async with uow_factory() as uow:
                    ack = await ingestion.accept_file(
                        uow,
                        ctx,
                        filename=f"{doc['id']}.md",
                        media_type="text/markdown",
                        data=body,
                        title=doc["title"][:200],
                    )
                    await uow.commit()
                document_to_corpus[ack.document_id] = doc["id"]
                if n % batch == 0:
                    await container.tasks.drain()
                    print(
                        f"[scifact] indexed {n}/{len(corpus)} "
                        f"({(time.perf_counter() - started) / n:.1f}s/doc)",
                        file=sys.stderr,
                        flush=True,
                    )
            await container.tasks.drain()
            await container.tasks.drain()
            index_seconds = round(time.perf_counter() - started, 1)

        # Refuse to score a corpus that was never indexed.
        #
        # There is already a guard for "candidates came back but mapped to nothing". This is
        # the other half, and it is the one that actually fired: indexing runs at roughly
        # 4-5 seconds per document on CPU, so a 5,183-document corpus needs about six hours.
        # A run that spent twenty-one minutes indexing covered ~370 documents and then scored
        # 300 queries against a corpus that was 93% absent. It reported nDCG@10 = 0.012 with
        # no indication that anything was wrong, and that number reads exactly like a
        # retrieval failure. A benchmark that cannot tell "the system is bad" from "the
        # corpus is missing" is worse than no benchmark.
        async with container.database.engine.connect() as conn:
            indexed = (
                await conn.execute(text("SELECT count(*) FROM document_versions"))
            ).scalar_one()
        if indexed < len(corpus):
            raise SystemExit(
                f"only {indexed} of {len(corpus)} documents finished indexing after "
                f"{index_seconds:.0f}s — scoring this would measure the missing corpus, not "
                f"retrieval. Ingest costs ~4-5s/document on CPU; use --limit to pick a corpus "
                f"that fits the time you have, or index against a machine that can keep up."
            )

        hits_at_k = 0
        mapped = returned = unmapped = 0
        ndcgs: list[float] = []
        latencies: list[float] = []
        records: list[dict[str, object]] = []
        # Indexing logs a line per document; the query phase logged nothing at all, so a run
        # sat silent at 600% CPU for forty minutes with no way to tell progress from a hang
        # short of reading /proc inside the container. stderr, so stdout stays clean JSON.
        print(
            f"[scifact] {len(queries)} queries over {len(corpus)} docs", file=sys.stderr, flush=True
        )
        for asked, query in enumerate(queries, start=1):
            relevant = set(query["relevant"])
            call = time.perf_counter()
            result = await engine.retrieve(ctx, query["text"], kinds=("chunk",), limit=k)
            latencies.append((time.perf_counter() - call) * 1000)
            # The document id lives in the candidate's payload, not as an attribute.
            # Reading it with getattr() silently produced "" for every hit, which scored a
            # confident 0.0 on both the stand-in and the real model — a broken benchmark
            # looks exactly like a broken system, so this mapping is asserted below.
            ranked: list[str] = []
            for candidate in result.candidates:
                corpus_id = document_to_corpus.get(candidate.payload.get("document_id") or "")
                if not corpus_id:
                    unmapped += 1
                elif corpus_id not in ranked:
                    ranked.append(corpus_id)
            mapped += len(ranked)
            returned += len(result.candidates)
            hit = bool(relevant & set(ranked[:k]))
            hits_at_k += int(hit)
            ndcg = _ndcg_at_k(ranked, relevant, k)
            ndcgs.append(ndcg)
            # Persisted per query so an ablation can be tested rather than eyeballed: an arm
            # difference of a few nDCG points over forty queries is not obviously signal, and
            # deciding a component's fate on it requires the paired values, not the means.
            records.append(
                {
                    "query": query["text"][:200],
                    "relevant": sorted(relevant),
                    "ranked": ranked[:k],
                    "ndcg": round(ndcg, 4),
                    "hit": hit,
                    "ms": round(latencies[-1], 1),
                }
            )
            if asked % 25 == 0 or asked == len(queries):
                mean = sum(latencies) / len(latencies) / 1000
                print(
                    f"[scifact] {asked}/{len(queries)} mean={mean:.1f}s "
                    f"eta={(len(queries) - asked) * mean / 60:.0f}m "
                    f"recall@{k}={hits_at_k / asked:.3f}",
                    file=sys.stderr,
                    flush=True,
                )

        latencies.sort()
        if returned and not mapped:
            raise SystemExit(
                f"the benchmark returned {returned} candidates and mapped none of them back to "
                "a corpus document — that is a broken harness, not a score of zero"
            )
        return {
            "dataset": {
                "name": data["name"],
                "split": data["split"],
                "source": data["source"],
                "corpus_size": len(corpus),
                "full_corpus": limit is None,
                "queries": len(queries),
            },
            "k": k,
            f"recall_at_{k}": round(hits_at_k / len(queries), 4) if queries else 0.0,
            f"ndcg_at_{k}": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
            "query_p50_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0.0,
            "query_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0.0,
            "index_seconds": index_seconds,
            # `mapped` counts *distinct documents* after chunks from the same document are
            # collapsed, while `returned` counts raw candidates — so this ratio is well below
            # 1 on a healthy run and was never a mapping success rate. `unmapped_candidates`
            # is the one that should be zero.
            "distinct_documents_per_candidates": f"{mapped}/{returned}",
            "unmapped_candidates": unmapped,
            "records": records,
            "embedding_provider": embedding_provider,
            # the honesty flags. `representative` is about the model: a stand-in embedding
            # produces numbers, not quality. `comparable_to_published` is about the corpus:
            # a subset has fewer distractors, so its recall is inflated and must never be
            # quoted against a published BEIR score — it is only valid against our own
            # previous run at the same size.
            "representative": embedding_provider not in STANDIN_EMBEDDINGS,
            "comparable_to_published": limit is None
            and embedding_provider not in STANDIN_EMBEDDINGS,
            "provenance": provenance(),
        }
    finally:
        await container.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="index only the first N documents")
    parser.add_argument("--k", type=int, default=10, help="cut-off for recall/nDCG (default 10)")
    parser.add_argument(
        "--off",
        nargs="*",
        default=[],
        metavar="FLAG",
        help="retrieval flags to disable for this run, e.g. --off rerank",
    )
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="score against the corpus already in the store instead of re-indexing it "
        "(for the second arm of an ablation, which changes only query-time behaviour)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        help="documents accepted between drains; smaller means less work held at once",
    )
    parser.add_argument("--out", default="external_retrieval.json", help="result filename")
    args = parser.parse_args()
    result = asyncio.run(
        run(
            args.limit,
            args.k,
            ablate=dict.fromkeys(args.off, False),
            reuse_index=args.reuse_index,
            batch=args.batch,
        )
    )
    result["ablation"] = dict.fromkeys(args.off, False)
    write_result(args.out, result)
    if not result["representative"]:
        os.write(
            2,
            b"NOT REPRESENTATIVE: embedding provider is a stand-in; these scores are noise.\n",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
