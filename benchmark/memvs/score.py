"""Score every system's rows, with 95 % bootstrap intervals, against this service.

    PYTHONPATH=src:sdk/python/src:. python -m benchmark.memvs.score --runs DIR \\
        --workloads build/memvs/workloads --capability-gate RAW.json.gz [--harness-answers RAW] \\
        --out docs/eval/memory-vs-frameworks.json

Per question, so every interval is over questions and every comparison is paired: a metric
where a system beats this service is one whose paired difference has a 95 % interval above 0
(``BOOTSTRAP`` resamples of the questions both answered). Evidence recall uses the regression
gate's own ``coverage`` (``benchmark.native_source_retrieval``); the capability set uses the
gate's own ``score`` on the context a system would push (its hits within the same 2,000-token
budget the gate's context has), and this service's capability rows are the gate run itself
(new session, context surface). Latency intervals are over calls; CPU, memory and durability
are one measurement per run and carry none.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.capabilities.dataset import cases, load_dataset, load_json, select
from benchmark.capabilities.scoring import Item, score
from benchmark.memvs.llm import MEMORY_TOKEN_BUDGET, within_budget
from benchmark.memvs.model import DatasetName, SystemName, Workload
from benchmark.native_source_retrieval import coverage

BOOTSTRAP = 2000
SEED = 7
CONFIDENCE = 0.95
DEPTHS = ("10", "50")
REFERENCE = SystemName.TRELLIS
RECALL_SETS = (
    DatasetName.LME18,
    DatasetName.LME_HELDOUT,
    DatasetName.LOCOMO,
    DatasetName.LOCOMO_C0,
)
JUDGED_SETS = (DatasetName.LME18, DatasetName.LOCOMO_C0)
ADVERSARIAL = "adversarial"
#: systems whose capability rows are their own budgeted context (this service's bundle)
OWN_BUDGET = frozenset({SystemName.TRELLIS.value, SystemName.TRELLIS_REACT.value})


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": round(self.mean, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "n": self.n,
        }


def _interval(samples: list[float]) -> tuple[float, float]:
    samples.sort()
    tail = (1 - CONFIDENCE) / 2
    return samples[int(tail * len(samples))], samples[
        min(len(samples) - 1, int((1 - tail) * len(samples)))
    ]


def bootstrap(
    values: Sequence[float], stat: Callable[[Sequence[float]], float] = statistics.fmean
) -> Estimate | None:
    if not values:
        return None
    rng = random.Random(SEED)
    draws = [stat([values[rng.randrange(len(values))] for _ in values]) for _ in range(BOOTSTRAP)]
    low, high = _interval(draws)
    return Estimate(stat(values), low, high, len(values))


def paired(ours: Mapping[str, float], theirs: Mapping[str, float]) -> Estimate | None:
    """Theirs minus ours over the questions both have: above 0 means they are better."""
    keys = sorted(set(ours) & set(theirs))
    return bootstrap([theirs[k] - ours[k] for k in keys])


def p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def p50(values: Sequence[float]) -> float:
    return statistics.median(values)


# --------------------------------------------------------------------------- loading


def _rows(runs: Path, system: str, dataset: str) -> list[dict[str, Any]]:
    path = runs / f"{system}.{dataset}.rows.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _json(runs: Path, system: str, dataset: str, part: str) -> dict[str, Any] | None:
    path = runs / f"{system}.{dataset}.{part}.json"
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- per question


def recall_values(rows: list[dict[str, Any]], workload: Workload) -> dict[str, dict[str, float]]:
    """metric -> question -> value, for turn and session recall at each depth."""
    gold = {q.query_id: q for u in workload.units for q in u.queries}
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        q = gold[row["query_id"]]
        if q.category == ADVERSARIAL or not q.evidence_turns or row.get("error"):
            continue
        for level, wanted, field in (
            ("turn", q.evidence_turns, "turns"),
            ("session", q.evidence_sessions, "sessions"),
        ):
            got = coverage([list(h[field]) for h in row["hits"]], list(wanted))
            for depth in DEPTHS:
                out[f"{level}_recall@{depth}"][row["query_id"]] = got[depth]["recall"]
    return out


def judged_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {r["query_id"]: float(bool(r["judge"])) for r in rows if r.get("judge") is not None}


def _hit_items(row: Mapping[str, Any]) -> list[Item]:
    """The hits a system would push: as many as the memory budget holds, in rank order (this
    service's capability rows are already its context within that budget)."""
    hits = row["hits"]
    if row["system"] not in OWN_BUDGET:
        kept = within_budget([h["text"] for h in hits], MEMORY_TOKEN_BUDGET).split("\n")
        hits = hits[: len(kept)] if hits else []
    return [Item("memory", h["text"], i, bool(h.get("historical"))) for i, h in enumerate(hits)]


def capability_values(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """(context passes, answer passes, capability) by case key, scored with the gate's rules."""
    expects = _capability_expects()
    context: dict[str, float] = {}
    answered: dict[str, float] = {}
    capability: dict[str, str] = {}
    for row in rows:
        key = row["query_id"]
        expect, lang = expects[key]
        capability[key] = row["category"]
        status = "INSUFFICIENT" if row.get("insufficient") or not row["hits"] else "COMPLETE"
        v = score(expect, _hit_items(row), lang, surface="context", evidence_status=status)
        context[key] = float(v.passed)
        if row.get("answer") is not None:
            items = [Item("answer", row["answer"], 0)] if row["answer"] else []
            a = score(expect, items, lang, surface="answer", refusal_markers=_refusals(lang))
            answered[key] = float(a.passed)
    return context, answered, capability


def gate_capability_values(raw: Path) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """This service's own gate run: new-session context verdicts by case key."""
    data = json.loads(gzip.decompress(raw.read_bytes()))
    context: dict[str, float] = {}
    capability: dict[str, str] = {}
    latency: dict[str, float] = {}
    for r in data["records"]:
        if r.get("placement") == "new_session" and r.get("surface") == "context":
            context[r["key"]] = float(bool(r["verdict"]["passed"]))
            capability[r["key"]] = r["capability"]
            latency[r["key"]] = float(r.get("latency_ms") or 0)
    return context, latency, capability


def harness_answer_values(raw: Path) -> dict[str, float]:
    data = json.loads(gzip.decompress(raw.read_bytes()))
    return {r["key"]: float(bool(r["passed"])) for r in data["records"]}


_EXPECTS: dict[str, tuple[Mapping[str, Any], str]] = {}


def _capability_expects() -> dict[str, tuple[Mapping[str, Any], str]]:
    if not _EXPECTS:
        dataset = load_dataset()
        for s in select(dataset, _all()):
            for lang in (*s.languages, "mixed"):
                for case in cases(s, lang):
                    _EXPECTS[case.key] = (case.query.expect, lang)
    return _EXPECTS


def _all() -> Any:
    from benchmark.capabilities.dataset import Selection

    return Selection()


def _refusals(lang: str) -> list[str]:
    markers = load_json("_languages.json")["refusal"]
    return list(markers.get(lang, markers["en"])) + (markers["en"] if lang != "en" else [])


def macro(
    values: Mapping[str, float], groups: Mapping[str, str], *, without: Sequence[str] = ()
) -> float | None:
    by: dict[str, list[float]] = defaultdict(list)
    for key, v in values.items():
        if groups[key] not in without:
            by[groups[key]].append(v)
    return statistics.fmean(statistics.fmean(v) for v in by.values()) if by else None


# --------------------------------------------------------------------------- isolation


def isolation_leaks(rows: list[dict[str, Any]], workload: Workload) -> dict[str, int]:
    """Hits (and pushed context) of one owner that carry another owner's secret."""
    secrets: dict[str, list[str]] = workload.meta["secrets"]
    leaks_hits = leaks_context = 0
    for row in rows:
        others = [s for owner, codes in secrets.items() if owner != row["unit_id"] for s in codes]
        leaks_hits += sum(any(s in h["text"] for s in others) for h in row["hits"])
        leaks_context += any(s in row["context"] for s in others)
    return {
        "queries": len(rows),
        "foreign_hits": leaks_hits,
        "contexts_with_foreign_secret": leaks_context,
    }


# --------------------------------------------------------------------------- report


def system_report(
    runs: Path, workloads: Mapping[DatasetName, Workload], system: str
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset, workload in workloads.items():
        rows = _rows(runs, system, dataset.value)
        if not rows:
            continue
        entry: dict[str, Any] = {
            "questions": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
        }
        if dataset in RECALL_SETS:
            entry["values"] = recall_values(rows, workload)
        if dataset in JUDGED_SETS:
            entry.setdefault("values", {})["answer_accuracy"] = judged_values(rows)
        if dataset is DatasetName.CAPABILITY:
            context, answered, groups = capability_values(rows)
            entry["values"] = {"capability_context": context, "capability_answer": answered}
            entry["groups"] = groups
        if dataset is DatasetName.ISOLATION:
            entry["isolation"] = isolation_leaks(rows, workload)
        answered_rows = [r for r in rows if r.get("answer_prompt_tokens")]
        if answered_rows:
            entry.setdefault("values", {})["answer_tokens"] = {
                r["query_id"]: float(
                    r["answer_prompt_tokens"] + (r["answer_completion_tokens"] or 0)
                )
                for r in answered_rows
            }
        entry["read_ms"] = [r["latency_ms"] for r in rows if not r.get("error")]
        for part in ("ingest", "query", "answer"):
            if (doc := _json(runs, system, dataset.value, part)) is not None:
                entry[part] = {k: v for k, v in doc.items() if k != "probe"}
        out[dataset.value] = entry
    return out


def summarize(reports: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Estimates per system and dataset, and paired differences against this service."""
    ours = reports.get(REFERENCE.value, {})
    table: dict[str, Any] = {}
    for system, datasets in reports.items():
        table[system] = {}
        for dataset, entry in datasets.items():
            cell: dict[str, Any] = {"questions": entry["questions"], "errors": entry["errors"]}
            for metric, values in (entry.get("values") or {}).items():
                est = bootstrap(list(values.values()))
                if est is None:
                    continue
                cell[metric] = est.as_dict()
                mine = ((ours.get(dataset) or {}).get("values") or {}).get(metric)
                if mine and system != REFERENCE.value:
                    diff = paired(mine, values)
                    if diff is not None:
                        cell[metric]["vs_trellis"] = diff.as_dict()
            if entry.get("groups"):
                cell["capability_macro"] = macro(
                    entry["values"]["capability_context"], entry["groups"]
                )
                cell["capability_macro_without_abstention"] = macro(
                    entry["values"]["capability_context"], entry["groups"], without=("abstention",)
                )
            reads = entry.get("read_ms") or []
            if reads:
                cell["read_ms"] = {
                    "p50": bootstrap(reads, p50).as_dict(),  # type: ignore[union-attr]
                    "p95": bootstrap(reads, p95).as_dict(),  # type: ignore[union-attr]
                }
            ingest = entry.get("ingest") or {}
            if ingest.get("write_ms"):
                writes = ingest["write_ms"]
                cell["write_ms"] = {
                    "p50": bootstrap(writes, p50).as_dict(),
                    "p95": bootstrap(writes, p95).as_dict(),
                }  # type: ignore[union-attr]
                turns = max(1, ingest["turns"])
                cell["ingest"] = {
                    "turns": ingest["turns"],
                    "write_seconds": ingest["write_seconds"],
                    "settle_seconds": ingest["settle_seconds"],
                    "cpu_s_per_1k_turns": round(1000 * ingest["usage"]["cpu_s"] / turns, 2),
                    "peak_rss_mb": ingest["usage"]["peak_rss_mb"],
                    "by_process": ingest["usage"]["by_process"],
                    "llm": ingest.get("llm"),
                }
            query = entry.get("query") or {}
            if query:
                n = max(1, query["queries"])
                cell["query"] = {
                    "cpu_ms_per_query": round(1000 * query["usage"]["cpu_s"] / n, 2),
                    "peak_rss_mb": query["usage"]["peak_rss_mb"],
                    "durability": query["durability"],
                }
            if entry.get("answer"):
                cell["answer_llm"] = entry["answer"]["llm"]
            if entry.get("isolation"):
                cell["isolation"] = entry["isolation"]
            table[system][dataset] = cell
    return table


#: (dataset, metric, header) - the report's columns, in order
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("lme18", "turn_recall@10", "LME-S 18 turn R@10"),
    ("lme18", "session_recall@10", "LME-S 18 session R@10"),
    ("lme_heldout", "turn_recall@10", "LME-S held-out turn R@10"),
    ("lme_heldout", "session_recall@10", "LME-S held-out session R@10"),
    ("locomo", "turn_recall@10", "LoCoMo turn R@10"),
    ("locomo", "session_recall@10", "LoCoMo session R@10"),
    ("locomo_c0", "session_recall@10", "LoCoMo c0 session R@10"),
    ("capability", "capability_context", "capability (context)"),
    ("lme18", "answer_accuracy", "LME-S 18 answer acc."),
    ("locomo_c0", "answer_accuracy", "LoCoMo c0 answer acc."),
    ("capability", "capability_answer", "capability (answer)"),
    ("lme18", "answer_tokens", "tokens / answer (LME-S 18)"),
)
#: the paired difference against this service excludes 0 in its favour / against it
BETTER, WORSE = "▲", "▼"
#: a cell the system was not measured on
MISSING = "n/a"


def _fmt(cell: Mapping[str, Any] | None, *, digits: int = 3) -> str:
    if not cell:
        return MISSING
    text = f"{cell['mean']:.{digits}f} [{cell['low']:.{digits}f}, {cell['high']:.{digits}f}]"
    diff = cell.get("vs_trellis")
    if diff and diff["low"] > 0:
        text += f" {BETTER}"
    elif diff and diff["high"] < 0:
        text += f" {WORSE}"
    return text


def markdown(table: Mapping[str, Any]) -> str:
    """Two tables: quality (one interval per cell, the mark when the paired difference against
    this service excludes 0), then cost and operations (LME-S 18 unless named)."""
    head = "| system | " + " | ".join(h for *_, h in COLUMNS) + " |"
    lines = [head, "|" + "---|" * (len(COLUMNS) + 1)]
    for system, datasets in table.items():
        cells = []
        for dataset, metric, _ in COLUMNS:
            cell = (datasets.get(dataset) or {}).get(metric)
            cells.append(_fmt(cell, digits=0 if metric == "answer_tokens" else 3))
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    ops = [
        "",
        "| system | read p50 ms | read p95 ms | write p95 ms (per session) | ingest CPU s / 1k turns |"
        " ingest peak RSS MB | durable after restart | foreign secrets (hits / contexts) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for system, datasets in table.items():
        lme = datasets.get("lme18") or {}
        iso = (datasets.get("isolation") or {}).get("isolation")
        dur = (lme.get("query") or {}).get("durability")
        ingest = lme.get("ingest") or {}
        ops.append(
            f"| {system} | {_fmt((lme.get('read_ms') or {}).get('p50'), digits=1)}"
            f" | {_fmt((lme.get('read_ms') or {}).get('p95'), digits=1)}"
            f" | {_fmt((lme.get('write_ms') or {}).get('p95'), digits=1)}"
            f" | {ingest.get('cpu_s_per_1k_turns', MISSING)} | {ingest.get('peak_rss_mb', MISSING)}"
            f" | {f'{dur["same_after_restart"]}/{dur["probed"]}' if dur else MISSING}"
            f" | {f'{iso["foreign_hits"]} / {iso["contexts_with_foreign_secret"]}' if iso else MISSING} |"
        )
    return "\n".join(lines + ops) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument(
        "--capability-gate", type=Path, help="this service's gate run (.raw.json.gz)"
    )
    parser.add_argument(
        "--harness-answers", type=Path, help="this service's harness-mode run (.raw.json.gz)"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    workloads = {
        d: Workload.load(args.workloads / f"{d.value}.json")
        for d in DatasetName
        if (args.workloads / f"{d.value}.json").exists()
    }
    reports = {s.value: system_report(args.runs, workloads, s.value) for s in SystemName}
    if args.capability_gate:
        context, latency, groups = gate_capability_values(args.capability_gate)
        cap = reports[REFERENCE.value].setdefault(
            DatasetName.CAPABILITY.value, {"questions": len(context), "errors": 0}
        )
        cap["values"] = {"capability_context": context}
        cap["groups"] = groups
        cap["read_ms"] = list(latency.values())
        if args.harness_answers:
            cap["values"]["capability_answer"] = harness_answer_values(args.harness_answers)
    reports = {s: r for s, r in reports.items() if r}
    table = summarize(reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"systems": table}, indent=1, ensure_ascii=False) + "\n")
    args.out.with_suffix(".tables.md").write_text(markdown(table))
    print(f"wrote {args.out} and its tables")


if __name__ == "__main__":
    main()
