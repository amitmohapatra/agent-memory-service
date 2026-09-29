"""Does a cross-encoder rank better than the fused order at all? One batch, no training.

    python -m benchmark.rerank_offline --dump benchmark/results/phase7/locomo_source_ensemble.json \
        --output benchmark/results/phase9/rerank_offline_ettin17m.json

The competitive review reads the absence of a cross-encoder as a gap, and the full-scale
measurement (``benchmark/results/full_rerank_ettin17m.json``) only ever answered what one
costs: p50 649 ms, p99 2018 ms against a 300 ms target. Whether it *ranks better* on this
corpus, under the exact source-turn ruler the arms use, has never been measured - so the
latency verdict rests on an unmeasured quality assumption, and so does any distillation
project downstream of it.

Nothing needs to be re-run to answer it. ``native_source_retrieval --dump-arms`` already
recorded, per question, the bundle the pipeline returned (``arms.fused``) and which of those
memories carry a gold source turn (``arms.carriers``). So: take the bundle the pipeline
already produced, score every candidate in it with the cross-encoder, reorder by that score,
and read source recall at depth 10 off the new order. Same questions, same candidates, same
ruler; the only thing that changes is the order. Three numbers come out per category:

* ``fused`` - what the pipeline scored, recomputed here from the dump (checked against the
  artifact's own summary, so a mismatch is caught rather than published),
* ``reranked`` - the same candidates in the cross-encoder's order,
* ``oracle`` - the best order any reranker could produce over these candidates. It is the
  honest ceiling for reordering, and the distance between ``reranked`` and ``oracle`` says
  whether a *better* teacher could still pay off if this one does not.

Query cost here is zero: this runs over a finished artifact. The scoring throughput is
reported so the cost of doing it at query time stays visible beside the quality delta.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from benchmark.common import dedicated_database, provenance
from benchmark.corpus import conversation_tenant
from benchmark.retrieval import _settings
from memory_service.adapters.models.rerankers import CrossEncoderReranker
from memory_service.config.constants import CrossEncoderModel

#: the depths the source harness scores, so a delta here lands in the same table as the arms
DEPTHS = (10, 20, 50)
#: the local cross-encoder this closes the question on: 17M parameters, the one whose
#: query-time cost is already measured at full scale
DEFAULT_SPEC = CrossEncoderModel(
    id="jhu-clsp/ettin-reranker-17m", local_dir="ettin-reranker-17m-v1", batch_size=32
)


def rows_of(dump: dict[str, Any]) -> list[dict[str, Any]]:
    """The scorable rows: answerable, with gold turns, a bundle, and its carriers."""
    return [
        {
            "conversation": row["conversation"],
            "question": row["question"],
            "category": row["category"],
            "gold": row["gold"],
            "fused": row["arms"]["fused"],
            "carriers": row["arms"]["carriers"],
        }
        for row in dump["records"]
        if row.get("category") != "adversarial" and row.get("gold") and row.get("arms")
    ]


def coverage(order: list[str], gold: set[str], carriers: dict[str, list[str]]) -> dict[str, Any]:
    """``recall`` and ``complete`` of ``gold`` among the carriers of ``order``, per depth."""
    out: dict[str, Any] = {}
    for depth in DEPTHS:
        found: set[str] = set()
        for memory_id in order[:depth]:
            found.update(carriers.get(memory_id, ()))
        hit = gold & found
        out[str(depth)] = {
            "recall": len(hit) / len(gold) if gold else 0.0,
            "complete": bool(gold) and hit == gold,
        }
    return out


def separation(scores: dict[str, float], carriers: dict[str, list[str]]) -> dict[str, float] | None:
    """How far the cross-encoder puts gold carriers above everything else, in score units.

    A reordering that loses to the order it replaced has two possible causes: the model ranks
    this corpus badly, or the model is not producing ranking scores at all (a checkpoint whose
    head is not a classification head still returns numbers). They look identical in a recall
    table and are not the same finding, so the raw scores are summarized here: a separation
    near zero, or a spread near zero, is a broken instrument rather than a weak teacher.
    """
    gold = [score for memory_id, score in scores.items() if carriers.get(memory_id)]
    rest = [score for memory_id, score in scores.items() if not carriers.get(memory_id)]
    if not gold or not rest:
        return None
    return {
        "gold_mean": sum(gold) / len(gold),
        "rest_mean": sum(rest) / len(rest),
        "separation": sum(gold) / len(gold) - sum(rest) / len(rest),
        "spread": max(scores.values()) - min(scores.values()),
    }


def sampled(
    rows: list[dict[str, Any]], *, category: str | None, sample: int
) -> list[dict[str, Any]]:
    """The rows to score: one category, and/or a deterministic per-category sample of it.

    Sampling is every ``stride``-th row of each category rather than a random draw, so the
    sample is reproducible from the artifact alone and keeps each category's conversations
    mixed. The per-category ``questions`` count travels with every number, because a delta on
    40 questions and a delta on 282 are not the same claim.
    """
    if category:
        rows = [row for row in rows if row["category"] == category]
    if sample <= 0 or sample >= len(rows):
        return rows
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    out: list[dict[str, Any]] = []
    for group in by_category.values():
        take = max(1, round(sample * len(group) / len(rows)))
        stride = max(1, len(group) // take)
        out.extend(group[::stride][:take])
    return out


def oracle_order(order: list[str], carriers: dict[str, list[str]]) -> list[str]:
    """The same candidates, every gold carrier first: the ceiling for any reordering.

    Stable within each group, so the ceiling is a reordering of this pool and not a different
    pool. A reranker cannot beat this and no training run can either.
    """
    carrying = [memory_id for memory_id in order if carriers.get(memory_id)]
    rest = [memory_id for memory_id in order if not carriers.get(memory_id)]
    return carrying + rest


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_category["all_answerable"].append(row)
    return {
        name: {
            "questions": len(group),
            "at": {
                str(depth): {
                    metric: round(
                        sum(r[key][str(depth)][metric] for r in group) / len(group),
                        4,
                    )
                    for metric in ("recall", "complete")
                }
                for depth in DEPTHS
            },
        }
        for name, group in sorted(by_category.items())
    }


async def texts_for(url: str, wanted: dict[str, set[str]]) -> dict[str, str]:
    """``memory_id -> content`` for the ids each conversation's tenant needs.

    The content is what the engine hands a reranker (``engine.py``: ``text=memory.content``),
    so the pairs scored here are the pairs a query-time reranker would have scored. Read
    only: this harness never writes to the corpus it reads.
    """
    engine = create_async_engine(url)
    out: dict[str, str] = {}
    try:
        async with engine.connect() as conn:
            for tenant_id, ids in sorted(wanted.items()):
                result = await conn.execute(
                    text(
                        "SELECT memory_id, content FROM memories "
                        "WHERE tenant_id = :tenant AND memory_id = ANY(:ids)"
                    ),
                    {"tenant": tenant_id, "ids": sorted(ids)},
                )
                out.update(dict(result.all()))  # type: ignore[arg-type]
    finally:
        await engine.dispose()
    return out


def check_fused(measured: dict[str, Any], artifact: dict[str, Any] | None) -> dict[str, Any]:
    """The recomputed fused order against the artifact's own summary, at every shared depth.

    The comparison is only worth reading if the baseline it is measured against is the number
    already published. A drift here means the dump and the summary disagree, which is a
    finding about the artifact, not a rounding difference to be waved through.
    """
    if not artifact:
        return {"checked": False}
    deltas = {
        f"{name}@{depth}": round(
            measured[name]["at"][str(depth)]["recall"] - artifact[name]["at"][str(depth)]["recall"],
            4,
        )
        for name in sorted(measured)
        if name in artifact
        for depth in DEPTHS
        if str(depth) in artifact[name]["at"]
    }
    worst = max((abs(delta) for delta in deltas.values()), default=0.0)
    return {"checked": True, "max_abs_delta": worst, "deltas": deltas}


async def run(args: argparse.Namespace) -> None:
    settings = _settings()
    dedicated_database(settings.database.url.get_secret_value())
    raw = args.dump.read_bytes()
    dump = json.loads(raw)
    rows = sampled(rows_of(dump), category=args.category, sample=args.sample)
    if not rows:
        raise SystemExit(f"{args.dump} carries no bundle dumps; run the harness with --dump-arms")
    wanted: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        wanted[conversation_tenant(row["conversation"])].update(row["fused"][: args.candidates])
    contents = await texts_for(settings.database.url.get_secret_value(), wanted)
    spec = (
        CrossEncoderModel.model_validate_json(args.spec.read_text()) if args.spec else DEFAULT_SPEC
    )
    reranker = CrossEncoderReranker(spec)
    pairs = 0
    started = time.perf_counter()
    try:
        for row in rows:
            pool = row["fused"][: args.candidates]
            candidates = [c for c in pool if c in contents]
            gold, carriers = set(row["gold"]), row["carriers"]
            scored = await reranker.rerank(
                row["question"], [contents[c] for c in candidates], top_k=len(candidates)
            )
            pairs += len(candidates)
            # A candidate with no text cannot be scored, but dropping it would hand the
            # reranker a smaller pool than the fused order it is measured against. It keeps
            # its place behind everything the reranker did score.
            unscored = [c for c in pool if c not in contents]
            reranked = [candidates[result.index] for result in scored] + unscored
            row["separation"] = separation({candidates[r.index]: r.score for r in scored}, carriers)
            row["fused_coverage"] = coverage(row["fused"], gold, carriers)
            row["reranked_coverage"] = coverage(reranked, gold, carriers)
            row["oracle_coverage"] = coverage(oracle_order(pool, carriers), gold, carriers)
            row["missing_texts"] = len(pool) - len(candidates)
    finally:
        reranker.close()
    elapsed = time.perf_counter() - started
    fused = summarize(rows, "fused_coverage")
    separations = [row["separation"] for row in rows if row["separation"]]
    complete_dump = not (args.category or args.sample)
    result = {
        "provenance": provenance(llm={"enabled": False, "provider": "disabled"}),
        "dump": str(args.dump),
        "dump_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": dump.get("dataset_sha256"),
        "index_fingerprint": dump.get("index_fingerprint"),
        "reranker": {**spec.model_dump(mode="json"), "fingerprint": reranker.fingerprint()},
        "candidates_per_question": args.candidates,
        "category": args.category,
        "sample": args.sample or None,
        "questions": len(rows),
        "pairs_scored": pairs,
        "scoring_seconds": round(elapsed, 2),
        "pairs_per_second": round(pairs / elapsed, 2) if elapsed else None,
        "missing_texts": sum(row["missing_texts"] for row in rows),
        "fused": fused,
        "reranked": summarize(rows, "reranked_coverage"),
        "oracle": summarize(rows, "oracle_coverage"),
        "score_separation": {
            "questions": len(separations),
            "gold_above_rest": round(
                sum(s["separation"] > 0 for s in separations) / len(separations), 4
            )
            if separations
            else None,
            "mean_separation": round(
                sum(s["separation"] for s in separations) / len(separations), 6
            )
            if separations
            else None,
            "mean_spread": round(sum(s["spread"] for s in separations) / len(separations), 6)
            if separations
            else None,
        },
        "fused_matches_artifact": check_fused(fused, dump.get("summary"))
        if complete_dump
        else {
            "checked": False,
            "reason": "a sampled run cannot be checked against the full summary",
        },
        "limitations": [
            "Reordering only: the candidate pool is the bundle the pipeline already returned.",
            "At the deepest depth every order holds the same candidates, so they agree there "
            "by construction; the comparison lives at the shallower depths.",
            "Exact source-ID coverage; a fragment counts as source presence, not a full answer.",
            "No query-time cost is measured here; pairs_per_second is the offline throughput.",
            "The oracle is the ceiling for THIS pool, not for retrieval.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=None, help="a CrossEncoderModel JSON")
    parser.add_argument(
        "--candidates",
        type=int,
        default=50,
        help="how deep into the bundle the reranker may reorder (the shipped final cut is 50)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="score a deterministic per-category sample of about N questions (0 = every one)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="score one category only, e.g. multi_hop (the bucket this phase is about)",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
