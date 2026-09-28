"""Offline NLI and grounding evaluation with CPU weights and no generation/judge calls.

Input is a pinned JSON manifest with labelled premise/hypothesis records. Sampling and
translation provenance must be fixed before predictions; every row, including failures,
is retained. Timings measure sequential one-pair NLI calls, not application requests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

from benchmark.common import local_model_runtime
from benchmark.harness import stats
from memory_service.adapters.models.onnx_nli import OnnxNLI
from memory_service.config.constants import NLIModel, NLISettings
from memory_service.modules.grounding.cascade import Evidence, GroundingCascade

LABELS = ("entailment", "neutral", "contradiction")
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src/memory_service"


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['mode']}/{row['language']}"].append(row)
    result = {}
    for key, group in groups.items():
        confusion = {actual: dict.fromkeys(LABELS, 0) for actual in LABELS}
        for row in group:
            confusion[row["label"]][row["predicted"]] += 1
        result[key] = {
            "pairs": len(group),
            "accuracy": sum(row["label"] == row["predicted"] for row in group) / len(group),
            "confusion": confusion,
            "pair_latency_ms": stats([row["latency_ms"] for row in group]),
        }
    return result


async def grounding(model, path: Path) -> dict:
    raw = await asyncio.to_thread(path.read_bytes)
    golden = json.loads(raw)
    cascade = GroundingCascade(model, settings=NLISettings())
    rows = []

    def evidence(keys):
        return [Evidence(key, golden["evidence"][key]) for key in keys]

    for case in golden["cases"]:
        started = time.perf_counter()
        report = await cascade.verify(
            case["answer"], evidence(case["evidence"]), unused=evidence(case.get("unused", []))
        )
        expected = case["expected"] if isinstance(case["expected"], list) else [case["expected"]]
        predicted = [claim.verdict for claim in report.claims]
        rows.append(
            {
                "id": case["id"],
                "expected": expected,
                "predicted": predicted,
                "ok": predicted == expected,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
        if report.llm_tokens or report.judge_consulted:
            raise ValueError("Offline grounding unexpectedly consulted a generative model")
    return {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": len(rows),
        "correct": sum(row["ok"] for row in rows),
        "rows": rows,
        "latency_ms": stats([row["latency_ms"] for row in rows]),
    }


async def run(args) -> None:
    data = json.loads(args.data.read_text())
    records = data["records"]
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Evaluation records must be nonempty with unique IDs")
    if any(row["label"] not in LABELS for row in records):
        raise ValueError("Unknown NLI label")
    spec = NLIModel.model_validate_json(args.spec.read_text())
    model = OnnxNLI(spec)
    result = {
        "spec": spec.model_dump(),
        "model_fingerprint": model.fingerprint(),
        "dataset": {key: value for key, value in data.items() if key != "records"},
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "runtime": local_model_runtime(),
        "llm_calls": 0,
        "complete": False,
        "expected_pairs": len(records),
        "records": [],
        "limitations": [
            "Classification accuracy, not LoCoMo answer accuracy or end-to-end RAG quality.",
            "One-pair sequential CPU component latency; not concurrent HTTP request latency.",
            "Public NLI may overlap training; the sample size and language coverage limit claims.",
        ],
    }
    result["source_sha256"] = {
        str(path.relative_to(SOURCE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        await model.entail([records[0]["premise"]], records[0]["hypothesis"])
        for number, record in enumerate(records, 1):
            started = time.perf_counter()
            score = (await model.entail([record["premise"]], record["hypothesis"]))[0]
            values = score.model_dump()
            result["records"].append(
                {
                    "id": record["id"],
                    "language": record["language"],
                    "mode": record["mode"],
                    "label": record["label"],
                    "predicted": max(LABELS, key=lambda label: values[label]),
                    "probabilities": values,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            if number % 100 == 0:
                result["summary"] = summarize(result["records"])
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(f"{number}/{len(records)}", flush=True)
        if args.golden:
            result["grounding"] = await grounding(model, args.golden)
        result["summary"] = summarize(result["records"])
        result["complete"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"]), flush=True)
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden", type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
