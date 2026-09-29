"""Fit the hybrid arms' RRF weights offline, from a per-arm rank dump, and spend them on depth.

``benchmark.native_source_retrieval --dump-arms`` records, per question, each arm's ranked
memory ids at prefetch depth and which of those memories carry a gold source turn. Fusion
is arithmetic on those lists, so every weighting and every depth can be scored here without
a single query: for each candidate weight vector the arms are fused with RRF, the top ``K``
memories are read, and a question is *complete* at ``K`` when every gold turn is carried by
one of them. The winner is the weighting that keeps the most questions complete at the
halved depth; the gate that promotes it is a fresh run, not this fit.

    python -m benchmark.fit_rrf_weights benchmark/results/phase7/locomo_source_ensemble.json \
        --output benchmark/results/phase7/rrf_weight_fit.json

No label is edited and no run is re-scored: the dump is read, the fit is written beside it
with the dump's hash, and the constant it proposes is a code change reviewed like one.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: the weights tried per arm; 1.0 everywhere is what every earlier number was measured at
GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
#: the RRF constants this repository fuses at (constants.RetrievalSettings.hybrid_rrf_k, rrf_k)
CONSTANTS = (1, 60)
#: the depths scored: the judged memory depth, its half, and the shipped final cut
DEPTHS = (200, 100, 50)


def fuse(arms: Mapping[str, Sequence[str]], weights: Mapping[str, float], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for name, ranked in arms.items():
        weight = weights.get(name, 1.0)
        for rank, memory_id in enumerate(ranked):
            scores[memory_id] = scores.get(memory_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))


def coverage(
    fused: Sequence[str], gold: set[str], carriers: Mapping[str, Sequence[str]], depth: int
) -> tuple[float, bool]:
    """``(recall, complete)`` of the gold turns among the carriers of the top ``depth``."""
    found: set[str] = set()
    for memory_id in fused[:depth]:
        found.update(carriers.get(memory_id, ()))
    hit = gold & found
    return (len(hit) / len(gold) if gold else 0.0), (hit == gold)


def score(
    questions: Sequence[dict[str, Any]], weights: Mapping[str, float], k: int
) -> dict[str, dict[str, float]]:
    """Every depth read off ONE fusion per question, which is what the depths are: prefixes.

    The fused order does not depend on the depth it is cut at, so fusing once per question and
    cutting it three times is the same arithmetic as fusing three times - and the fusion is
    this fit's whole cost. Measured on the ensemble dump (1,536 questions, three arms), a
    250-point grid at 5.6 s per weighting is 23 minutes of one core; on a contended host,
    where this process got 12% of one, it was three hours.
    """
    totals = {depth: [0.0, 0] for depth in DEPTHS}
    for question in questions:
        fused = fuse(question["arms"], weights, k)
        gold, carriers = set(question["gold"]), question["carriers"]
        for depth in DEPTHS:
            recall, complete = coverage(fused, gold, carriers, depth)
            totals[depth][0] += recall
            totals[depth][1] += complete
    n = len(questions)
    return {
        str(depth): {"recall": round(total / n, 4), "complete": round(hits / n, 4)}
        for depth, (total, hits) in totals.items()
    }


def questions_of(dump: dict[str, Any]) -> list[dict[str, Any]]:
    """The scorable rows: answerable, with gold turns and an arm dump."""
    return [
        {"arms": row["arms"]["arms"], "carriers": row["arms"]["carriers"], "gold": row["gold"]}
        for row in dump["records"]
        if row.get("category") != "adversarial" and row.get("gold") and row.get("arms")
    ]


def fit(questions: Sequence[dict[str, Any]], *, target_depth: int) -> dict[str, Any]:
    names = sorted({name for question in questions for name in question["arms"]})
    baseline = dict.fromkeys(names, 1.0)
    rows: list[dict[str, Any]] = []
    for k in CONSTANTS:
        for combination in itertools.product(GRID, repeat=len(names)):
            weights = dict(zip(names, combination, strict=True))
            rows.append({"k": k, "weights": weights, "at": score(questions, weights, k)})
    key = str(target_depth)
    rows.sort(
        key=lambda row: (
            -row["at"][key]["complete"],
            -row["at"][key]["recall"],
            row["k"],
            sum(abs(w - 1.0) for w in row["weights"].values()),
        )
    )
    best = rows[0]
    equal = next(row for row in rows if row["k"] == 1 and row["weights"] == baseline)
    return {
        "arms": names,
        "questions": len(questions),
        "target_depth": target_depth,
        "grid": list(GRID),
        "constants": list(CONSTANTS),
        "equal_weights_k1": equal,
        "best": best,
        "gain_at_target": {
            metric: round(best["at"][key][metric] - equal["at"][key][metric], 4)
            for metric in ("recall", "complete")
        },
        "top": rows[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument(
        "--target-depth", type=int, default=100, help="the halved depth the fit is spent on"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.dump.read_bytes()
    dump = json.loads(raw)
    questions = questions_of(dump)
    if not questions:
        raise SystemExit(
            f"{args.dump} carries no arm dumps; run the source harness with --dump-arms"
        )
    result = {
        "dump": str(args.dump),
        "dump_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": dump.get("dataset_sha256"),
        "encoders": dump.get("encoders"),
        **fit(questions, target_depth=args.target_depth),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("questions", "equal_weights_k1", "best", "gain_at_target")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
