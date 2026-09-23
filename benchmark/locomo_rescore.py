"""Re-grade an existing LoCoMo result under a different ruler, without re-running it.

    python -m benchmark.locomo_rescore benchmark/results/locomo_judged_v5.json --judge-ruler lenient
    python -m benchmark.locomo_rescore ... --judge-ruler lenient --judge-model openai/gpt-4.1-mini

Every judged result carries, per question, the answer the model produced and the gold. Grading
is the only thing a ruler changes, so a second ruler is a second pass over those records - a
few hundred judge calls and no retrieval - rather than a second twenty-minute run. The output
is a sibling file, ``<name>_<ruler>[_<judge>].json``, with the same shape and both the ruler
and the judge model named in it, so two numbers can sit side by side and never be confused.

The judge is half of a ruler. Every published LoCoMo number was produced by a GPT-class judge,
and a judge of a different family is reported to score systematically lower on the same
answers - which is exactly the kind of claim that must be measured rather than repeated. With
``--judge-model`` the same produced answers are re-graded by another model through the
gateway, so the judge's contribution to a score becomes a number instead of an argument.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

from benchmark.env import bench_overrides
from benchmark.locomo import JUDGE_RULERS, _judge, _Pacer, judged_hit
from benchmark.retrieval import _settings
from memory_service.application.container import build_container


async def rescore(
    result: dict, ruler: str, calls_per_minute: float, judge_model: str | None = None
) -> dict:
    """The same records, graded again under ``ruler``. Pure over its input; I/O is the caller's."""
    settings = _settings()
    if judge_model:
        # The judge use is routed to the fast model (see BifrostLLM.model_for), so naming a
        # judge means naming both: the grading call must not fall back to the answerer's model.
        settings = settings.model_copy(
            update={
                "models": settings.models.model_copy(
                    update={
                        "llm": settings.models.llm.model_copy(
                            update={"model": judge_model, "fast_model": judge_model}
                        )
                    }
                )
            }
        )
    container = await build_container(settings, "bench", overrides=bench_overrides())
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
            # The record says which ruler and which judge produced its verdict: a rescored
            # file used to carry the new verdicts under the old file's labels, so a row could
            # not be told apart from the run it came from.
            #
            # A row whose original judge failed keeps its produced answer precisely so it can
            # be repaired here, so the failure that row is carrying is now stale: leaving it
            # in place would mark a row that has a real verdict as a judge failure.
            repaired = {k: v for k, v in judged.items() if k not in ("error", "detail")}
            rec["judged"] = {**repaired, **verdict}
            rec["judge_ruler"] = ruler
            rec["judge_model"] = llm.model_for("grounding_judge")
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
    out["judge_model"] = llm.model_for("grounding_judge")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--judge-ruler", choices=sorted(JUDGE_RULERS), required=True)
    parser.add_argument("--calls-per-minute", type=float, default=120.0)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="grade with this model instead of the configured one (the gateway must serve it); "
        "the output file is named after it so the two gradings cannot be confused",
    )
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if not result.get("judged"):
        raise SystemExit(f"{args.result} was not judged; there are no produced answers to re-grade")
    out = asyncio.run(rescore(result, args.judge_ruler, args.calls_per_minute, args.judge_model))
    out["rescored_from"] = args.result.name
    suffix = args.judge_ruler
    if args.judge_model:
        suffix += "_" + args.judge_model.replace("/", "-").replace(".", "_")
    target = args.result.with_name(f"{args.result.stem}_{suffix}.json")
    target.write_text(json.dumps(out, indent=2) + "\n")
    keys = ("answer_recall_at_k", "by_category", "judge_failures", "judge_ruler", "judge_model")
    print(json.dumps({k: out[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
