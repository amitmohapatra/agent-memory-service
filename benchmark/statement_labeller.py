"""Statement labeller benchmark (ADR 0037): accuracy and write-path cost on CPU.

    python -m benchmark.statement_labeller [--lexicon-only] [--locomo benchmark/data/locomo10.json]

* **Accuracy**: macro-F1 per language and per kind over both halves of
  ``tests/eval/golden/statement_kinds.json`` and the two blind sets of
  ``statement_kinds_blind.json`` (``benchmark.evaluation.statement_kinds``), the blind sets
  beside the extractor before the labeller.
* **Cost**, at each ``--threads`` the head may have (1 is what a default deployment gives
  it, 2 the 8-vCPU three-worker target): milliseconds per sentence over the golden set as
  observations of one and of three sentences, over each blind set with every sentence a
  message of its own (the worst case per statement), and over LoCoMo turn by turn as the
  write path labels a message (``--locomo``: per statement and per turn, and the share of
  sentences and turns that reached the head).
* **Precision on text nobody labelled** (``--locomo``): the kinds given to every sentence of
  the LoCoMo dialogues - human chat, written by neither the packs' author nor for them. Rules,
  corrections, statuses and lifecycle changes are rare in it, so their counts are an upper
  bound on false positives.

Writes ``benchmark/results/statement_labeller.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.common import RESULTS, provenance
from benchmark.evaluation.statement_kinds import (
    BLIND,
    below_baseline,
    evaluate,
    evaluate_baseline,
    load,
)
from benchmark.harness import stats
from memory_service.config.constants import FROZEN_MODELS
from memory_service.modules.memory.native import split_sentences
from memory_service.modules.memory.statements import StatementLabeller


async def _cost(labeller: StatementLabeller, sentences: list[str], per_call: int) -> dict[str, Any]:
    latencies: list[float] = []
    reached: Counter[str] = Counter()
    for start in range(0, len(sentences), per_call):
        batch = sentences[start : start + per_call]
        began = time.perf_counter()
        labels = await labeller.label(batch)
        latencies.append((time.perf_counter() - began) * 1000 / len(batch))
        reached.update(label.source for label in labels)
    return {
        "sentences_per_observation": per_call,
        "ms_per_sentence": stats(latencies),
        "sources": dict(reached),
    }


def _locomo(path: Path) -> list[list[str]]:
    """Every LoCoMo turn as the sentences the write path splits it into."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        split_sentences(turn.get("text", ""))
        for conversation in data
        for key, turns in conversation["conversation"].items()
        if key.startswith("session_") and isinstance(turns, list)
        for turn in turns
    ]


async def _write_path(labeller: StatementLabeller, turns: list[list[str]]) -> dict[str, Any]:
    """LoCoMo labelled turn by turn, as the write path labels a message: all its sentences in
    one call; a statement's cost is its turn's time over the turn's sentences."""
    per_turn: list[float] = []
    per_statement: list[float] = []
    kinds: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    reached_turns = 0
    for turn in turns:
        began = time.perf_counter()
        labels = await labeller.label(turn)
        took = (time.perf_counter() - began) * 1000
        per_turn.append(took)
        per_statement.extend([took / len(turn)] * len(turn))
        kinds.update(label.kind.value if label.kind else "NONE" for label in labels)
        sources.update(label.source for label in labels)
        reached_turns += any(label.source != "lexicon" for label in labels)
    return {
        "turns": len(turns),
        "sentences": len(per_statement),
        "kinds": dict(kinds),
        "sources": dict(sources),
        "head_share_sentences": round(1 - sources["lexicon"] / max(1, len(per_statement)), 4),
        "head_share_turns": round(reached_turns / max(1, len(turns)), 4),
        "ms_per_statement": stats(per_statement),
        "ms_per_turn": stats(per_turn),
    }


async def _blind_cost(labeller: StatementLabeller, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Each blind sentence labelled as a message of its own (the worst case per statement)."""
    return await _cost(labeller, [item["text"] for item in items], 1)


async def main(args: argparse.Namespace) -> dict[str, Any]:
    golden, blind = load(), load(BLIND)
    sentences = [item["text"] for item in golden["dev"] + golden["test"]]
    turns = [t for t in _locomo(Path(args.locomo)) if t] if args.locomo else []
    heads: list[tuple[int | None, Any]] = [(None, None)]
    if not args.lexicon_only:
        from memory_service.adapters.models.onnx_nli import OnnxNLI

        heads = [(n, OnnxNLI(FROZEN_MODELS.nli, threads=n)) for n in args.threads]
    report: dict[str, Any] = {"benchmark": "statement_labeller", "cost_by_threads": {}}
    for threads, nli in heads:
        labeller = StatementLabeller(nli=nli)
        await labeller.label(sentences[:4])  # warm the session outside the measurement
        if "dev" not in report:
            # accuracy does not depend on the head's threads: measured once
            report["nli_provider"] = nli.fingerprint() if nli is not None else None
            report["dev"] = await evaluate(labeller, golden["dev"])
            report["test"] = await evaluate(labeller, golden["test"])
            report["blind"] = {}
            for name in ("blind1", "blind2"):
                mine = await evaluate(labeller, blind[name]["items"])
                baseline = evaluate_baseline(blind[name]["items"])
                report["blind"][name] = {
                    "role": blind[name]["role"],
                    "labeller": mine,
                    "baseline": baseline,
                    "below_baseline": below_baseline(mine, baseline),
                }
        cost: dict[str, Any] = {
            "golden": [await _cost(labeller, sentences, n) for n in (1, 3)],
            "blind2": await _blind_cost(labeller, blind["blind2"]["items"]),
            "blind1": await _blind_cost(labeller, blind["blind1"]["items"]),
        }
        if turns:
            cost["locomo"] = await _write_path(labeller, turns)
        report["cost_by_threads"][str(threads or 0)] = cost
        if nli is not None:
            nli.close()
    report["provenance"] = provenance()
    out = RESULTS / (
        "statement_labeller_lexicon.json" if args.lexicon_only else "statement_labeller.json"
    )
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--lexicon-only", action="store_true", help="no NLI head")
    parser.add_argument("--locomo", help="a LoCoMo JSON file (benchmark/data/locomo10.json)")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2],
        help="the head's intra-op threads to measure the cost at (1: a default deployment, "
        "one API worker per CPU; 2: the 8-vCPU, three-worker target)",
    )
    result = asyncio.run(main(parser.parse_args()))
    for half in ("dev", "test"):
        print(
            half,
            result[half]["macro_f1"],
            {k: v["macro_f1"] for k, v in result[half]["per_language"].items()},
            "false rules",
            result[half]["false_rules"],
        )
    for name, b in result["blind"].items():
        print(
            name,
            b["labeller"]["macro_f1"],
            "baseline",
            b["baseline"]["macro_f1"],
            "below",
            b["below_baseline"],
        )
    for threads, cost in result["cost_by_threads"].items():
        print("threads", threads, "blind2", cost["blind2"]["ms_per_sentence"])
        if "locomo" in cost:
            print(
                "  locomo",
                {k: v for k, v in cost["locomo"].items() if k.startswith(("ms", "head"))},
            )
