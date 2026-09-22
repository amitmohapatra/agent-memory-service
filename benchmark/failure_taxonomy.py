"""Classify the wrong answers of a judged LoCoMo run, so the next change aims at a class.

    python -m benchmark.failure_taxonomy benchmark/results/locomo_judged_v5.json

A judged result says how many questions were wrong; it does not say *how*. Four very
different defects hide behind one number, and they are fixed in four different places:

* **abstained** — the answerer said "I don't know" with the gold turns in the bundle. That is
  the answer protocol, not the memory: every published comparator forbids abstaining outright.
* **partial** — the answer is right under the lenient ruler and wrong under the strict one,
  which on this corpus almost always means an enumeration that named three of four items.
  Precomputed dated aggregates answer those from one line instead of asking the model to
  reassemble them from a hundred.
* **wrong instance** — a plausible non-gold entry was chosen while the gold one was present.
  That is salience: ranking, rendering order and noise per line.
* **evidence missing** — the only class that is actually a retrieval failure.

The split is computed from the artefacts a run already writes: the judged file for the strict
verdicts and its ``_lenient`` sibling (``make bench-locomo-rescore``) for the ruler gap. No
re-run, no model calls.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

#: Order matters: the first rule that matches wins, most-specific first.
CLASSES = ("evidence missing", "abstained", "partial", "wrong instance")


def _key(record: dict[str, Any]) -> tuple[Any, str]:
    return record.get("conversation"), str(record.get("question", ""))


def classify(record: dict[str, Any], *, lenient_hit: bool | None) -> str:
    """Which failure class this wrong answer belongs to."""
    judged = record.get("judged") or {}
    if not record.get("evidence_all_hit", record.get("evidence_hit", False)):
        return "evidence missing"
    if judged.get("abstained"):
        return "abstained"
    if lenient_hit:
        return "partial"
    return "wrong instance"


def taxonomy(result: dict[str, Any], lenient: dict[str, Any] | None = None) -> dict[str, Any]:
    """Counts per class, per category and per (category, class), plus the rows themselves."""
    lenient_hit = (
        {_key(r): bool(r.get("hit")) for r in lenient.get("records", [])} if lenient else {}
    )
    rows: list[dict[str, Any]] = []
    for record in result.get("records", []):
        if record.get("category") == "adversarial" or record.get("hit"):
            continue
        judged = record.get("judged") or {}
        rows.append(
            {
                "class": classify(record, lenient_hit=lenient_hit.get(_key(record))),
                "category": record.get("category"),
                "question": record.get("question"),
                "gold": record.get("answer"),
                "produced": (judged.get("produced") or "")[:200],
                "reason": (judged.get("reason") or "")[:200],
            }
        )
    by_class = collections.Counter(r["class"] for r in rows)
    by_pair = collections.Counter((r["category"], r["class"]) for r in rows)
    return {
        "wrong_answerable": len(rows),
        "by_class": {c: by_class[c] for c in CLASSES if by_class[c]},
        "by_category_class": {f"{cat}/{cls}": n for (cat, cls), n in sorted(by_pair.items())},
        "rows": rows,
        "lenient_compared": bool(lenient_hit),
    }


def _render(report: dict[str, Any], *, samples: int) -> str:
    out = [f"{report['wrong_answerable']} wrong answerable answers"]
    if not report["lenient_compared"]:
        out.append(
            "  (no lenient sibling found: 'partial' cannot be separated from 'wrong instance')"
        )
    for cls, n in report["by_class"].items():
        out.append(f"  {n:3}  {cls}")
    out.append("")
    for pair, n in report["by_category_class"].items():
        out.append(f"  {pair:48} {n}")
    if samples:
        out.append("")
        seen: collections.Counter[str] = collections.Counter()
        for row in report["rows"]:
            if seen[row["class"]] >= samples:
                continue
            seen[row["class"]] += 1
            out.append(f"  [{row['class']} | {row['category']}]")
            out.append(f"    Q:    {row['question'][:100]}")
            out.append(f"    gold: {str(row['gold'])[:80]}")
            out.append(f"    got:  {row['produced'][:80]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--lenient",
        type=Path,
        default=None,
        help="the _lenient rescore (default: the sibling file, when it exists)",
    )
    parser.add_argument("--samples", type=int, default=3, help="example rows per class")
    parser.add_argument("--out", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    result = json.loads(args.result.read_text())
    lenient_path = args.lenient or args.result.with_name(f"{args.result.stem}_lenient.json")
    lenient = json.loads(lenient_path.read_text()) if lenient_path.is_file() else None

    report = taxonomy(result, lenient)
    sys.stdout.write(_render(report, samples=args.samples) + "\n")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
