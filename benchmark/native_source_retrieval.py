"""Isolated, no-LLM LoCoMo evaluation using exact source IDs at fixed memory depths.

Uses the existing ingestion helper and production context builder. Supplemental graph
and derived evidence consume positions just like primary memories. Presence of a source
ID proves provenance coverage, not that a retrieved fragment contains its full answer.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from sqlalchemy.engine import make_url

import memory_service
from benchmark.common import provenance, reset_store
from benchmark.env import bench_overrides, bench_retrieval
from benchmark.harness import stats
from benchmark.locomo import CATEGORY_NAMES, TENANT, _evidence_ids, _ingest_conversation
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


async def run(args) -> None:
    settings = _settings()
    database = make_url(settings.database.url.get_secret_value()).database or ""
    if not database.startswith("memory_hi_"):
        raise ValueError("This harness only resets dedicated memory_hi_* databases")
    if settings.search.qdrant_url != "http://localhost:16333":
        raise ValueError("Use the isolated Qdrant on localhost:16333, never the shared server")
    settings.models.llm = LLMSettings(enabled=False)  # no .env can authorize model calls
    spec = DenseModel.model_validate_json(args.spec.read_text())
    base_overrides = bench_overrides()
    overrides = replace(
        base_overrides,
        embedding=None,
        dense_model=spec,
        search=None,
        retrieval=bench_retrieval(base_overrides).model_copy(
            update={"semantic_graph": args.semantic_graph}
        ),
        memory_intelligence=MEMORY_INTELLIGENCE.model_copy(
            update={"consolidation_enabled": args.consolidation}
        ),
    )
    container = await build_container(settings, __version__, overrides=overrides)
    register_handlers(container)
    raw = args.data.read_bytes()
    dataset = json.loads(raw)
    if args.conversations:
        dataset = dataset[: args.conversations]
    rows = []
    corpora = []
    result = {
        "provenance": provenance(llm={"enabled": False, "provider": "disabled"}),
        "source_manifest": source_manifest(),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec.model_dump(),
        "encoder": container.embedding.fingerprint(),
        "consolidation": args.consolidation,
        "semantic_graph": args.semantic_graph,
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
        for number, conversation in enumerate(dataset):
            await reset_store(container, TENANT)
            ctx = MemoryExecutionContext(
                tenant_id=TENANT, user_id=f"locomo-{number}", workspace_id="ws"
            )
            source_ids = {}
            print(f"ingesting conversation {number + 1}/{len(dataset)}", flush=True)
            await _ingest_conversation(
                container, ctx, conversation["conversation"], source_ids=source_ids
            )
            print(f"ingested conversation {number + 1}: {len(source_ids)} turns", flush=True)
            async with container.services["uow_factory"]() as uow:
                memories = await container.services["memory"].list_memories(uow, ctx, limit=20000)
            if len(memories) >= 20000:
                raise ValueError("Corpus inventory reached its explicit evaluation bound")
            represented = set().union(
                *(observation_sources(memory.evidence, source_ids) for memory in memories)
            )
            annotated = [
                set(_evidence_ids(question))
                for question in conversation["qa"]
                if CATEGORY_NAMES[question["category"]] != "adversarial" and _evidence_ids(question)
            ]
            corpora.append(
                {
                    "conversation": number,
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
            lineage = SourceLineage(container.services["uow_factory"], TENANT, source_ids)
            for question in conversation["qa"]:
                started = time.perf_counter()
                bundle = await builder.build(ctx, question["question"])
                elapsed = (time.perf_counter() - started) * 1000
                evidence = [item.evidence for item in bundle.memories]
                sources = [sorted(observation_sources(group, source_ids)) for group in evidence]
                lineage_sources = await lineage.resolve(evidence)
                gold = _evidence_ids(question)
                rows.append(
                    {
                        "conversation": number,
                        "question": question["question"],
                        "category": CATEGORY_NAMES[question["category"]],
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
                )
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
                    "latency_ms": stats([r["latency_ms"] for r in rows]),
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(number + 1, len(rows), result["summary"]["all_answerable"], flush=True)
            await builder.drain()
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conversations", type=int)
    parser.add_argument("--consolidation", action="store_true")
    parser.add_argument("--semantic-graph", action=argparse.BooleanOptionalAction, default=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
