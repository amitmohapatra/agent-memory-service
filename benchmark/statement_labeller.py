"""Statement labeller benchmark (ADR 0037): accuracy and write-path cost on CPU.

    python -m benchmark.statement_labeller [--lexicon-only] [--locomo benchmark/data/locomo10.json]

* **Accuracy**: macro-F1 per language and per kind over both halves of
  ``tests/eval/golden/statement_kinds.json`` and the two blind sets of
  ``statement_kinds_blind.json`` (``benchmark.evaluation.statement_kinds``), the blind sets
  beside the extractor before the labeller.
* **Cost**: milliseconds per sentence, labelled the way the write path labels them - every
  sentence of an observation in one call, so the NLI pairs of its open sentences share one
  batch - with p50/p95 over observations of one and of three sentences, and the share of
  sentences that reached the NLI head.
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
from benchmark.evaluation.statement_kinds import BLIND, evaluate, evaluate_baseline, load
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


async def main(args: argparse.Namespace) -> dict[str, Any]:
    nli = None
    if not args.lexicon_only:
        from memory_service.adapters.models.onnx_nli import OnnxNLI

        nli = OnnxNLI(FROZEN_MODELS.nli)
    labeller = StatementLabeller(nli=nli)
    golden, blind = load(), load(BLIND)
    sentences = [item["text"] for item in golden["dev"] + golden["test"]]
    await labeller.label(sentences[:4])  # warm the session outside the measurement
    report: dict[str, Any] = {
        "benchmark": "statement_labeller",
        "nli_provider": nli.fingerprint() if nli is not None else None,
        "dev": await evaluate(labeller, golden["dev"]),
        "test": await evaluate(labeller, golden["test"]),
        "blind": {
            name: {
                "role": blind[name]["role"],
                "labeller": await evaluate(labeller, blind[name]["items"]),
                "baseline": evaluate_baseline(blind[name]["items"]),
            }
            for name in ("blind1", "blind2")
        },
        "cost": [await _cost(labeller, sentences, n) for n in (1, 3)],
    }
    if args.locomo:
        labels = [
            label for turn in _locomo(Path(args.locomo)) for label in await labeller.label(turn)
        ]
        report["locomo"] = {
            "sentences": len(labels),
            "kinds": dict(Counter(label.kind.value if label.kind else "NONE" for label in labels)),
            "sources": dict(Counter(label.source for label in labels)),
        }
    report["provenance"] = provenance()
    if nli is not None:
        nli.close()
    out = RESULTS / (
        "statement_labeller_lexicon.json" if args.lexicon_only else "statement_labeller.json"
    )
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--lexicon-only", action="store_true", help="no NLI head")
    parser.add_argument("--locomo", help="a LoCoMo JSON file (benchmark/data/locomo10.json)")
    result = asyncio.run(main(parser.parse_args()))
    for half in ("dev", "test"):
        print(
            half,
            result[half]["macro_f1"],
            {k: v["macro_f1"] for k, v in result[half]["per_language"].items()},
        )
    for name, b in result["blind"].items():
        print(name, b["labeller"]["macro_f1"], "baseline", b["baseline"]["macro_f1"])
    print("cost", [(c["sentences_per_observation"], c["ms_per_sentence"]) for c in result["cost"]])
    if "locomo" in result:
        print("locomo", result["locomo"])
