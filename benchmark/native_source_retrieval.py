"""Isolated, no-LLM LoCoMo evaluation using exact source IDs at fixed memory depths.

Uses the existing ingestion helper and production context builder. Supplemental graph
and derived evidence consume positions just like primary memories. Presence of a source
ID proves provenance coverage, not that a retrieved fragment contains its full answer.

Two additions for the accuracy programme. ``--reuse-corpus`` keeps each conversation in its
own tenant and, through ``benchmark.corpus``, skips ingestion when the ledger says the store
already holds this dataset under this index fingerprint and these ingestion settings - the
arms that change only the query side share one corpus. ``--dump-arms`` records, per
question, every retriever's own ranking and where each gold turn landed in it
(``benchmark.arms``), which is what the offline weight fit reads.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import memory_service
from benchmark.arms import dump_arms
from benchmark.common import dedicated_database, isolated_qdrant, provenance, reset_store
from benchmark.corpus import CorpusKey, CorpusLedger, conversation_tenant, ensure_conversation
from benchmark.env import BENCH, bench_overrides, bench_retrieval
from benchmark.harness import stats
from benchmark.locomo import (
    CATEGORY_NAMES,
    TENANT,
    THREADED_INGEST,
    _evidence_ids,
    _ingest_conversation,
)
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.constants import MEMORY_INTELLIGENCE, DenseModel
from memory_service.config.settings import LLMSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.jobs.registry import register_handlers

DEPTHS = (10, 20, 50, 100)


def source_manifest() -> dict[str, str]:
    """Hash the actually imported tree, including uncommitted and new Python files."""
    root = Path(memory_service.__file__).parent
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    files["benchmark/native_source_retrieval.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return files


def observation_sources(evidence: list[EvidenceRef], source_ids: dict[str, str]) -> set[str]:
    return {
        source_ids[e.source_id]
        for e in evidence
        if e.source_type != "memory" and e.source_id in source_ids
    }


class SourceLineage:
    """Bounded provenance expansion for evaluation, outside the measured read path.

    Transitive citations prove only ancestry, not that an insight states the answer.
    Keep this metric separate from directly returned observation evidence.
    """

    def __init__(self, uow_factory, tenant_id: str, source_ids: dict[str, str]) -> None:
        self.uow_factory = uow_factory
        self.tenant_id = tenant_id
        self.source_ids = source_ids

    async def resolve(self, evidence: list[list[EvidenceRef]]) -> list[list[str]]:
        nodes: dict[str, list[EvidenceRef]] = {}
        frontier = {e.source_id for group in evidence for e in group if e.source_type == "memory"}
        for _ in range(16):
            pending = sorted(frontier - nodes.keys())
            if not pending:
                break
            if len(nodes) + len(pending) > 4096:
                raise ValueError("Evaluation source lineage exceeds the node budget")
            async with self.uow_factory() as uow:
                memories = await uow.memories.get_many(self.tenant_id, pending)
            # Missing records are leaves, never guessed or credited as observations.
            nodes.update({key: [] for key in pending})
            nodes.update({m.memory_id: m.evidence for m in memories})
            frontier = {
                e.source_id for m in memories for e in m.evidence if e.source_type == "memory"
            }
        if frontier - nodes.keys():
            raise ValueError("Evaluation source lineage exceeds the depth budget")
        return [sorted(self._sources(group, nodes)) for group in evidence]

    def _sources(
        self, evidence: list[EvidenceRef], nodes: dict[str, list[EvidenceRef]]
    ) -> set[str]:
        found: set[str] = set()
        seen: set[str] = set()
        pending = list(evidence)
        while pending:
            ref = pending.pop()
            if ref.source_type != "memory" and ref.source_id in self.source_ids:
                found.add(self.source_ids[ref.source_id])
            elif ref.source_type == "memory" and ref.source_id not in seen:
                seen.add(ref.source_id)
                pending.extend(nodes.get(ref.source_id, []))
        return found


def coverage(candidates: list[list[str]], gold: list[str]) -> dict:
    expected = set(gold)
    if not expected:
        return {}
    out = {}
    for depth in DEPTHS:
        found = set().union(*map(set, candidates[:depth]))
        hits = expected.intersection(found)
        out[str(depth)] = {"recall": len(hits) / len(expected), "complete": hits == expected}
    return out


def summarize(rows: list[dict]) -> dict:
    categories = defaultdict(list)
    for row in rows:
        if row["category"] != "adversarial" and row["coverage"]:
            categories[row["category"]].append(row)
            categories["all_answerable"].append(row)
    return {
        name: {
            "questions": len(group),
            "at": {
                str(k): {
                    metric: sum(r["coverage"][str(k)][metric] for r in group) / len(group)
                    for metric in ("recall", "complete")
                }
                for k in DEPTHS
            },
        }
        for name, group in categories.items()
    }


def _guard(settings: Any) -> str:
    """The database this run owns, or a refusal: only a dedicated database on the isolated
    Qdrant is ever reset by a harness.

    Both rules now live in ``benchmark.common`` so every harness shares one vocabulary;
    ``reset_store`` enforces the database rule again at the TRUNCATE itself.
    """
    database = dedicated_database(settings.database.url.get_secret_value())
    isolated_qdrant(settings.search.qdrant_url)
    return database


def refuse_silent_reingest(ledger: CorpusLedger, key: CorpusKey, *, allowed: bool) -> None:
    """A reuse arm must not quietly become an ingest arm over somebody else's corpus.

    ``--reuse-corpus`` skips ingestion when the ledger vouches for the store, and otherwise
    TRUNCATEs every conversation's tenant and ingests from scratch - forty minutes on an idle
    box, hours on a busy one, and the corpus that was there is gone with the arms that were
    measured on it. When the ledger holds a DIFFERENT corpus that is a mismatch worth seeing
    (a changed encoder, a changed ingestion setting, the wrong database) rather than an
    instruction, so it is spelled out and ``--allow-reingest`` is what says otherwise. An
    empty ledger is a fresh database and ingests without asking.
    """
    if allowed or ledger.data.get("key") is None:
        return
    held, wanted = ledger.data["key"], asdict(key)
    differs = sorted(field for field, value in wanted.items() if held.get(field) != value)
    raise SystemExit(
        f"{ledger.path} holds a different corpus ({', '.join(differs)} differ), and reusing it "
        "would TRUNCATE every conversation tenant and ingest from scratch. Point the arm at "
        "the database that holds the corpus it needs, or pass --allow-reingest to rebuild it."
    )


def _ingestion_settings(settings: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Everything that shapes the corpus at ingest, for the corpus ledger's key."""
    llm = settings.models.llm
    return {
        "llm": {"enabled": str(llm.enabled), "model": llm.model, "uses": sorted(llm.uses)},
        "consolidation": args.consolidation,
        "threaded_ingest": THREADED_INGEST,
        "graph_enrichment": BENCH.graph_enrichment,
    }


async def run(args) -> None:
    settings = _settings()
    database = _guard(settings)
    settings.models.llm = LLMSettings(enabled=False)  # no .env can authorize model calls
    base_overrides = bench_overrides()
    overrides = replace(
        base_overrides,
        search=None,
        retrieval=bench_retrieval(base_overrides).model_copy(
            update={"semantic_graph": args.semantic_graph}
        ),
        memory_intelligence=MEMORY_INTELLIGENCE.model_copy(
            update={"consolidation_enabled": args.consolidation}
        ),
    )
    if args.spec is not None:
        spec = DenseModel.model_validate_json(args.spec.read_text())
        overrides = replace(overrides, embedding=None, dense_model=spec)
    container = await build_container(settings, __version__, overrides=overrides)
    register_handlers(container)
    raw = args.data.read_bytes()
    dataset = json.loads(raw)
    if args.conversations:
        dataset = dataset[: args.conversations]
    spaces = container.dense_spaces
    indexer = container.services["indexer"]
    key = CorpusKey(
        dataset_sha256=hashlib.sha256(raw).hexdigest(),
        index_fingerprint=indexer.fingerprint,
        ingestion_sha256=CorpusKey.ingestion_digest(_ingestion_settings(settings, args)),
    )
    ledger = CorpusLedger(database)
    rows: list[dict[str, Any]] = []
    corpora: list[dict[str, Any]] = []
    #: Questions whose read raised. A failed call is unmeasured, never wrong, so it leaves the
    #: denominators rather than scoring zero - and it must not end the run either. One Qdrant
    #: read that exceeded its 5 s deadline on a loaded host ended a 1,986-question arm on its
    #: first question, which is hours of host time lost to one slow read. The count and the
    #: reasons travel in the artifact, so a summary built on a degraded run is visible as one.
    failures: list[dict[str, Any]] = []
    result = {
        "provenance": provenance(llm={"enabled": False, "provider": "disabled"}),
        "source_manifest": source_manifest(),
        "dataset_sha256": key.dataset_sha256,
        "spec": args.spec and DenseModel.model_validate_json(args.spec.read_text()).model_dump(),
        "encoders": {space.name.value: space.encoder.fingerprint() for space in spaces.spaces},
        "spaces_fingerprint": spaces.fingerprint(),
        "index_fingerprint": indexer.fingerprint,
        "retrieval": container.tuning.retrieval.model_dump(mode="json"),
        "context": container.tuning.context.model_dump(mode="json"),
        "dense_arm": BENCH.dense,
        "consolidation": args.consolidation,
        "semantic_graph": args.semantic_graph,
        "reuse_corpus": args.reuse_corpus,
        "paid_llm_calls": 0,
        "answer_accuracy": None,
        "expected_questions": sum(len(c["qa"]) for c in dataset),
        "limitations": [
            "Exact source-ID coverage; fragments count as source presence, not full answers.",
            "Transitive source lineage is reported separately; ancestry does not prove answer support.",
            "Not generated-answer accuracy or adversarial abstention accuracy.",
            "Sequential context-builder latency; PostgreSQL and remote Qdrant included, HTTP excluded.",
            "In-process authorization/cache and lexical NLI; not full production load latency.",
        ],
    }
    try:
        if args.reuse_corpus and args.adopt_corpus and not ledger.matches(key):
            result["corpus_adopted"] = {"held": ledger.adopt(key), "as": asdict(key)}
        if args.reuse_corpus and not ledger.matches(key):
            refuse_silent_reingest(ledger, key, allowed=args.allow_reingest)
            # a fresh corpus: every conversation's tenant is cleared once, up front
            for number in range(len(dataset)):
                await reset_store(container, conversation_tenant(number))
        for number, conversation in enumerate(dataset):
            tenant = conversation_tenant(number) if args.reuse_corpus else TENANT
            ctx = MemoryExecutionContext(
                tenant_id=tenant, user_id=f"locomo-{number}", workspace_id="ws"
            )
            print(f"ingesting conversation {number + 1}/{len(dataset)}", flush=True)
            if args.reuse_corpus:
                _, source_ids, reused = await ensure_conversation(
                    container,
                    ctx,
                    conversation["conversation"],
                    index=number,
                    key=key,
                    ledger=ledger,
                    reuse=True,
                )
            else:
                await reset_store(container, TENANT)
                source_ids = {}
                await _ingest_conversation(
                    container, ctx, conversation["conversation"], source_ids=source_ids
                )
                reused = False
            print(
                f"{'reused' if reused else 'ingested'} conversation {number + 1}: {len(source_ids)} turns",
                flush=True,
            )
            async with container.services["uow_factory"]() as uow:
                memories = await container.services["memory"].list_memories(uow, ctx, limit=20000)
            if len(memories) >= 20000:
                raise ValueError("Corpus inventory reached its explicit evaluation bound")
            memory_sources = {
                memory.memory_id: observation_sources(memory.evidence, source_ids)
                for memory in memories
            }
            represented = set().union(*memory_sources.values()) if memory_sources else set()
            annotated = [
                set(_evidence_ids(question))
                for question in conversation["qa"]
                if CATEGORY_NAMES[question["category"]] != "adversarial" and _evidence_ids(question)
            ]
            corpora.append(
                {
                    "conversation": number,
                    "tenant": tenant,
                    "reused": reused,
                    "canonical_memories": len(memories),
                    "observed_turns": len(source_ids),
                    "directly_represented_turns": len(represented),
                    "annotated_answerable_questions": len(annotated),
                    "all_gold_sources_present": sum(gold <= represented for gold in annotated),
                    "missing_source_ids": sorted(set(source_ids.values()) - represented),
                }
            )
            del memories
            builder = container.services["context_builder"]
            lineage = SourceLineage(container.services["uow_factory"], tenant, source_ids)
            for question in conversation["qa"]:
                started = time.perf_counter()
                try:
                    bundle = await builder.build(ctx, question["question"])
                except Exception as error:  # noqa: BLE001 - see the note on `failures`
                    failures.append(
                        {
                            "conversation": number,
                            "category": CATEGORY_NAMES[question["category"]],
                            "error": f"{type(error).__name__}: {error}"[:300],
                        }
                    )
                    continue
                elapsed = (time.perf_counter() - started) * 1000
                evidence = [item.evidence for item in bundle.memories]
                sources = [sorted(observation_sources(group, source_ids)) for group in evidence]
                lineage_sources = await lineage.resolve(evidence)
                gold = _evidence_ids(question)
                row = {
                    "conversation": number,
                    "question": question["question"],
                    "category": CATEGORY_NAMES[question["category"]],
                    "query_script": bundle.diagnostics.get("query_script"),
                    "gold": gold,
                    "sources": sources,
                    "coverage": coverage(sources, gold),
                    "lineage_sources": lineage_sources,
                    "lineage_coverage": coverage(lineage_sources, gold),
                    "returned_memories": len(bundle.memories),
                    "tokens": bundle.token_estimate,
                    "latency_ms": elapsed,
                    "cache_hit": bundle.cache_hit,
                }
                if args.dump_arms and gold and row["category"] != "adversarial":
                    dump = await dump_arms(
                        container,
                        ctx,
                        question["question"],
                        bundle=bundle,
                        gold=set(gold),
                        sources=memory_sources,
                    )
                    row["arms"] = dump.as_dict()
                rows.append(row)
            result.update(
                {
                    "records": rows,
                    "corpora": corpora,
                    "summary": summarize(rows),
                    "lineage_summary": summarize(
                        [{**row, "coverage": row["lineage_coverage"]} for row in rows]
                    ),
                    "completed_questions": len(rows),
                    "complete": len(rows) == result["expected_questions"],
                    "failed_questions": len(failures),
                    "failures": failures[:50],
                    "latency_ms": stats([r["latency_ms"] for r in rows]),
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                number + 1,
                len(rows),
                f"failed={len(failures)}",
                result["summary"]["all_answerable"],
                flush=True,
            )
            await builder.drain()
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--spec", type=Path, default=None, help="a challenger DenseModel JSON for the English space"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conversations", type=int)
    parser.add_argument("--consolidation", action="store_true")
    parser.add_argument("--semantic-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-corpus",
        action="store_true",
        help="one tenant per conversation; skip ingestion when the corpus ledger vouches for the store",
    )
    parser.add_argument(
        "--dump-arms",
        action="store_true",
        help="record every retriever's own ranking and the gold turns' ranks in it, per question",
    )
    parser.add_argument(
        "--adopt-corpus",
        action="store_true",
        help="query-side arms: reuse a corpus whose ingestion settings differ (same dataset "
        "and index); the artifact records what it held",
    )
    parser.add_argument(
        "--allow-reingest",
        action="store_true",
        help="rebuild the corpus when the ledger holds a different one (TRUNCATEs every tenant)",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
