"""Re-grade an existing LoCoMo result under a different ruler, without re-running it.

    python -m benchmark.locomo_rescore benchmark/results/locomo_judged_v2.json --judge-ruler lenient

Every judged result carries, per question, the answer the model produced and the gold. Grading
is the only thing a ruler changes, so a second ruler is a second pass over those records - a
few hundred judge calls and no retrieval - rather than a second twenty-minute run. The output
is a sibling file, ``<name>_<ruler>.json``, with the same shape and the ruler named in it, so
the two numbers can sit side by side and never be confused for each other.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

from benchmark.locomo import JUDGE_RULERS, _judge, _Pacer, judged_hit
from benchmark.retrieval import _settings
from memory_service.application.container import build_container


async def rescore(result: dict, ruler: str, calls_per_minute: float) -> dict:
    """The same records, graded again under ``ruler``. Pure over its input; I/O is the caller's."""
    container = await build_container(_settings(), "bench")
    llm = container.llm
    if not getattr(llm, "enabled", False):
        raise SystemExit("rescoring needs a generative model: set models.llm.enabled=true")
    pacer = _Pacer(calls_per_minute)
    out = copy.deepcopy(result)
    by_cat: dict[str, list[bool]] = {}
    failures = 0
    try:
        for i, rec in enumerate(out["records"]):
            judged = rec.get("judged") or {}
            produced = judged.get("produced")
            if produced is None:
                failures += 1
                continue
            await pacer.wait()
            try:
                verdict = await _judge(llm, rec["question"], rec["answer"], produced, ruler=ruler)
            except Exception as exc:  # noqa: BLE001 - a rescore must say it failed
                failures += 1
                rec["judged"] = {**judged, "error": f"{type(exc).__name__}: {exc}"[:300]}
                continue
            rec["judged"] = {**judged, **verdict}
            hit = judged_hit(rec["category"], rec["judged"])
            rec["hit"] = hit
            by_cat.setdefault(rec["category"], []).append(bool(hit))
            if (i + 1) % 25 == 0:
                print(f"[rescore:{ruler}] {i + 1}/{len(out['records'])}", file=sys.stderr)
    finally:
        await container.close()
    answerable = [h for c, hs in by_cat.items() if c != "adversarial" for h in hs]
    out["answer_recall_at_k"] = round(sum(answerable) / len(answerable), 4) if answerable else 0.0
    adv = by_cat.get("adversarial") or []
    out["abstention_rate_on_adversarial"] = round(sum(adv) / len(adv), 4) if adv else None
    out["by_category"] = {
        c: {"n": len(hs), "score": round(sum(hs) / len(hs), 4)} for c, hs in by_cat.items()
    }
    out["judge_failures"] = failures
    out["judge_ruler"] = ruler
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--judge-ruler", choices=sorted(JUDGE_RULERS), required=True)
    parser.add_argument("--calls-per-minute", type=float, default=120.0)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if not result.get("judged"):
        raise SystemExit(f"{args.result} was not judged; there are no produced answers to re-grade")
    out = asyncio.run(rescore(result, args.judge_ruler, args.calls_per_minute))
    out["rescored_from"] = args.result.name
    target = args.result.with_name(f"{args.result.stem}_{args.judge_ruler}.json")
    target.write_text(json.dumps(out, indent=2) + "\n")
    keys = ("answer_recall_at_k", "by_category", "judge_failures", "judge_ruler")
    print(json.dumps({k: out[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
