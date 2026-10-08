"""Metrics over the per-phrasing records of a run, and the JSON + Markdown it is kept as.

The primary number is the pushed context in a new session (``surface=context``,
``placement=new_session``): what an agent has in front of it when a user comes back another
day. ``same_session`` (the same thread, the framework keeping its own window) and ``search``
(the agent's own pull) are reported beside it, and gated like it.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.capabilities.dataset import Dataset

PRIMARY = ("context", "new_session")
#: a run without a context surface (harness mode) is primarily its answers
PRIMARY_ANSWER = ("answer", "new_session")
#: failure examples kept per capability
EXAMPLES = 3


def pct(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (the convention of benchmark/retrieval.py)."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return round(ordered[k], 2)


def _acc(rs: Sequence[Mapping[str, Any]]) -> float | None:
    return round(sum(1 for r in rs if r["passed"]) / len(rs), 4) if rs else None


def _partial(rs: Sequence[Mapping[str, Any]]) -> float | None:
    scored = [r for r in rs if r.get("groups_required")]
    if not scored:
        return None
    return round(statistics.fmean(r["groups_met"] / r["groups_required"] for r in scored), 4)


def _mean(values: Iterable[float | None]) -> float | None:
    vs = [v for v in values if v is not None]
    return round(statistics.fmean(vs), 4) if vs else None


def _by(
    rs: Iterable[Mapping[str, Any]], *keys: str
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    out: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for r in rs:
        out[tuple(str(r[k]) for k in keys)].append(r)
    return dict(sorted(out.items()))


def cells(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Accuracy per surface x placement x capability x language - the gate's unit."""
    return [
        {
            "surface": s,
            "placement": p,
            "capability": c,
            "language": lang,
            "n": len(rs),
            "passed": sum(1 for r in rs if r["passed"]),
            "accuracy": _acc(rs),
        }
        for (s, p, c, lang), rs in _by(
            records, "surface", "placement", "capability", "language"
        ).items()
    ]


def _abstention(
    rs: Sequence[Mapping[str, Any]], dataset_expect: Mapping[str, bool]
) -> dict[str, Any]:
    """Precision of the service's INSUFFICIENT, its recall on the unknowable questions, and
    how often it refuses an answerable one."""
    said = [r for r in rs if r.get("abstained")]
    unknowable = [r for r in rs if dataset_expect.get(r["key"].rsplit("/", 2)[0], False)]
    answerable = [r for r in rs if not dataset_expect.get(r["key"].rsplit("/", 2)[0], False)]
    true_pos = [r for r in said if dataset_expect.get(r["key"].rsplit("/", 2)[0], False)]
    return {
        "said_insufficient": len(said),
        "precision": round(len(true_pos) / len(said), 4) if said else None,
        "recall": round(len(true_pos) / len(unknowable), 4) if unknowable else None,
        "false_refusal_rate": round(
            sum(1 for r in answerable if r.get("abstained")) / len(answerable), 4
        )
        if answerable
        else None,
        "unknowable": len(unknowable),
        "answerable": len(answerable),
    }


def _expects_abstain(dataset: Dataset) -> dict[str, bool]:
    """``scenario/query`` -> whether its right answer is "I do not know"."""
    return {
        f"{s.id}/{q.id}": bool(q.expect.get("abstain"))
        for s in dataset.scenarios
        for q in s.queries
    }


def _stale_rate(rs: Sequence[Mapping[str, Any]], with_absent: set[str]) -> dict[str, Any]:
    pool = [r for r in rs if r["key"].rsplit("/", 2)[0] in with_absent]
    leaked = [r for r in pool if r.get("leaks")]
    return {
        "phrasings_with_stale_values": len(pool),
        "leaked": len(leaked),
        "rate": round(len(leaked) / len(pool), 4) if pool else None,
    }


def _failures(rs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (cap,), group in _by(rs, "capability").items():
        bad = [r for r in group if not r["passed"]]
        reasons = Counter(r["reason"] for r in bad)
        examples = []
        seen_q: set[str] = set()
        # one example per query, English first, so the examples show different problems
        for r in sorted(
            bad, key=lambda r: (r["language"] != "en", r["kind"] != "direct", r["key"])
        ):
            q = r["key"].rsplit("/", 2)[0]
            if q in seen_q:
                continue
            seen_q.add(q)
            examples.append(
                {
                    "key": r["key"],
                    "reason": r["reason"],
                    "question": r["question"],
                    "missing": r["missing"][:4],
                    "leaks": r["leaks"],
                    "evidence_status": r.get("evidence_status"),
                    "got": r.get("preview", [])[:3],
                }
            )
            if len(examples) >= EXAMPLES:
                break
        out[cap] = {
            "failed": len(bad),
            "of": len(group),
            "reasons": dict(reasons.most_common()),
            "examples": examples,
        }
    return out


def build(
    dataset: Dataset, records: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]
) -> dict[str, Any]:
    has_context = any(r["surface"] == "context" for r in records)
    primary_key = PRIMARY if has_context or not records else PRIMARY_ANSWER
    primary = [r for r in records if (r["surface"], r["placement"]) == primary_key]
    by_cap = _by(primary, "capability")
    caps = [c.name for c in dataset.capabilities if (c.name,) in by_cap]
    abstain_q = _expects_abstain(dataset)
    with_absent = {
        f"{s.id}/{q.id}"
        for s in dataset.scenarios
        for q in s.queries
        if q.expect.get("absent") and not q.expect.get("abstain")
    }

    def per(rs: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
        return {k: {"accuracy": _acc(g), "n": len(g)} for (k,), g in _by(rs, key).items()}

    by_capability = {}
    for cap in caps:
        rs = by_cap[(cap,)]
        by_capability[cap] = {
            "accuracy": _acc(rs),
            "partial_credit": _partial(rs),
            "n": len(rs),
            "by_language": per(rs, "language"),
            "by_domain": per(rs, "domain"),
            "by_split": per(rs, "split"),
            "by_kind": per(rs, "kind"),
            "retail": _acc([r for r in rs if r["domain"] == "retail"]),
            "origin_cases": _acc([r for r in rs if r.get("origin")]),
        }

    def macro(rs: Sequence[Mapping[str, Any]]) -> float | None:
        return _mean(_acc(g) for g in _by(rs, "capability").values())

    surfaces = {
        f"{s}/{p}": {
            "macro_accuracy": macro(g),
            "n": len(g),
            "by_capability": {c: _acc(gg) for (c,), gg in _by(g, "capability").items()},
        }
        for (s, p), g in _by(records, "surface", "placement").items()
    }
    context_lat = [r["latency_ms"] for r in records if r["surface"] == "context"]
    search_lat = [r["latency_ms"] for r in records if r["surface"] == "search"]
    answer_lat = [r["latency_ms"] for r in records if r["surface"] == "answer"]
    summary = {
        "primary": "/".join(primary_key),
        "macro_accuracy": macro(primary),
        "micro_accuracy": _acc(primary),
        "partial_credit": _partial(primary),
        "retail_track": macro([r for r in primary if r["domain"] == "retail"]),
        "origin_cases": _acc([r for r in primary if r.get("origin")]),
        "by_language": {lang: macro(g) for (lang,), g in _by(primary, "language").items()},
        "by_domain": {d: macro(g) for (d,), g in _by(primary, "domain").items()},
        "by_split": {sp: macro(g) for (sp,), g in _by(primary, "split").items()},
        "surfaces": surfaces,
        #: the service's INSUFFICIENT on the context; the agent's refusal on an answer surface
        "abstention": _abstention(
            [r for r in records if r["surface"] == primary_key[0]], abstain_q
        ),
        "stale_leak": _stale_rate(primary, with_absent),
        #: per surface: a stale value can only leak where something comes back at all
        "stale_leak_by_surface": {
            f"{sf}/{pl}": _stale_rate(g, with_absent)
            for (sf, pl), g in _by(records, "surface", "placement").items()
        },
        "abstention_by_placement": {
            pl: _abstention(g, abstain_q)
            for (pl,), g in _by(
                [r for r in records if r["surface"] == primary_key[0]], "placement"
            ).items()
        },
        "latency_ms": {
            "context_p50": pct(context_lat, 50),
            "context_p95": pct(context_lat, 95),
            "context_max": round(max(context_lat), 2) if context_lat else None,
            "search_p50": pct(search_lat, 50),
            "search_p95": pct(search_lat, 95),
            "answer_p50": pct(answer_lat, 50),
            "answer_p95": pct(answer_lat, 95),
            "answer_max": round(max(answer_lat), 2) if answer_lat else None,
            "by_capability_context_p95": {
                c: pct([r["latency_ms"] for r in g], 95)
                for (c,), g in _by(
                    [r for r in records if r["surface"] == "context"], "capability"
                ).items()
            },
        },
    }
    load = meta.get("host_load_during_asks") or {}
    summary["latency_ms"]["trustworthy"] = load.get("latency_trustworthy")
    summary["latency_ms"]["max_load_per_core"] = load.get("max_load_per_core")
    return {
        "gate": "capability",
        "mode": meta.get("mode"),
        "dataset": {"sha256": dataset.sha256, **dataset.counts()},
        # the users and threads written stay in the raw records (cleanup reads them there)
        "meta": {k: v for k, v in meta.items() if k != "written"},
        "summary": summary,
        "by_capability": by_capability,
        "cells": cells(records),
        "failures": _failures(primary),
        "failures_by_surface": {
            f"{sf}/{pl}": _failures(g)
            for (sf, pl), g in _by(records, "surface", "placement").items()
            if (sf, pl) != primary_key
        },
        "scale": _scale_summary(records),
        "records": list(records),
        "provenance": _provenance(),
    }


def _scale_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The scale capability's own numbers: how big the tenant was, how slow the reads got,
    and how much of the evidence an aggregate answer needs came back (partial credit)."""
    rs = [r for r in records if r["capability"] == "scale"]
    if not rs:
        return {}
    out: dict[str, Any] = {"corpus_tokens_max": max((r.get("corpus_tokens") or 0) for r in rs)}
    for surface, group in _by(rs, "surface").items():
        lat = [r["latency_ms"] for r in group]
        out[surface[0]] = {
            "accuracy": _acc(group),
            "evidence_recall": _partial(group),
            "latency_p50_ms": pct(lat, 50),
            "latency_p95_ms": pct(lat, 95),
            "latency_max_ms": round(max(lat), 2),
            #: the answer budget the capability names (5-10 s for an answer)
            "within_5s": round(sum(1 for x in lat if x <= 5000) / len(lat), 4),
            "within_10s": round(sum(1 for x in lat if x <= 10000) / len(lat), 4),
        }
    return out


def _provenance() -> dict[str, Any]:
    try:
        from benchmark.common import provenance

        return provenance()
    except Exception as exc:  # noqa: BLE001 - outside this repository's environment (harness venv)
        import platform
        import subprocess
        from datetime import UTC, datetime

        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:  # noqa: BLE001
            commit = "unknown"
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit": commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": f"reduced provenance ({type(exc).__name__})",
        }


# --------------------------------------------------------------------------- files


def save_raw(records: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], path: Path) -> Path:
    """The records and the run's metadata as measured, beside the result: a summary can be
    rebuilt from it (``python -m benchmark.capabilities.report <raw>``) without asking again."""
    raw = path.with_suffix(".raw.json.gz")
    raw.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"meta": meta, "records": list(records)}, ensure_ascii=False, default=str)
    raw.write_bytes(gzip.compress(body.encode("utf-8"), mtime=0))
    return raw


def rebuild(raw: Path, out: Path | None = None) -> tuple[Path, Path]:
    from benchmark.capabilities.dataset import load_dataset

    opened = gzip.decompress(raw.read_bytes()) if raw.suffix == ".gz" else raw.read_bytes()
    data = json.loads(opened.decode("utf-8"))
    payload = build(load_dataset(), data["records"], data["meta"])
    if data["meta"].get("sample"):
        payload["meta"]["sample"] = data["meta"]["sample"]
    stem = raw.name.removesuffix(".gz").removesuffix(".raw.json")
    return write(payload, out or raw.with_name(stem + ".json"))


def write(payload: Mapping[str, Any], path: Path) -> tuple[Path, Path]:
    """The result without its records (they are in the sibling ``.raw.json.gz``, written
    before the summary was built) and its Markdown summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in payload.items() if k != "records"}
    body["records_file"] = path.with_suffix(".raw.json.gz").name
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    md = path.with_suffix(".md")
    md.write_text(markdown(body), encoding="utf-8")
    return path, md


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _queues(depths: Mapping[str, float] | None) -> str:
    busy = {q: int(n) for q, n in (depths or {}).items() if n}
    if depths is None:
        return "not sampled"
    return ", ".join(f"{q} {n}" for q, n in busy.items()) or "all empty"


def markdown(p: Mapping[str, Any]) -> str:
    s = p["summary"]
    meta = p["meta"]
    lines = [
        f"# Capability eval — {meta.get('mode')} ({meta.get('run_date')})",
        "",
        f"Dataset `{p['dataset']['sha256'][:12]}`: {p['dataset']['scenarios']} scenarios, "
        f"{p['dataset']['queries']} queries, {p['dataset']['phrasings_per_language']} phrasings per "
        f"language. Selection: `{json.dumps(meta.get('selection'), ensure_ascii=False)}`"
        + (f", sample `{meta['sample']}`" if meta.get("sample") else "")
        + ".",
        "",
        f"**Primary ({s['primary']}) macro accuracy {_fmt(s['macro_accuracy'])}**, retail track "
        f"{_fmt(s['retail_track'])}, the eleven origin cases {_fmt(s['origin_cases'])}, partial "
        f"credit {_fmt(s['partial_credit'])}. Context latency p50/p95 "
        f"{_fmt(s['latency_ms']['context_p50'])}/{_fmt(s['latency_ms']['context_p95'])} ms"
        + (
            ""
            if s["latency_ms"].get("trustworthy")
            else f" — **flagged: measured at load {_fmt(s['latency_ms'].get('max_load_per_core'))}"
            " per core, an upper bound only**"
        )
        + ".",
        "",
        "| # | Capability | " + " | ".join(s["surfaces"]) + " | partial | retail | p95 ms |",
        "|---|---|" + "---|" * len(s["surfaces"]) + "---|---|---|",
    ]
    for i, (cap, c) in enumerate(p["by_capability"].items(), 1):
        row = [str(i), cap]
        row += [_fmt(v["by_capability"].get(cap)) for v in s["surfaces"].values()]
        row += [
            _fmt(c["partial_credit"]),
            _fmt(c["retail"]),
            _fmt(s["latency_ms"]["by_capability_context_p95"].get(cap)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Per language (primary, macro): "
        + ", ".join(f"{k} {_fmt(v)}" for k, v in s["by_language"].items()),
        "",
        "Per domain (primary, macro): "
        + ", ".join(f"{k} {_fmt(v)}" for k, v in s["by_domain"].items()),
        "",
        "Per split (primary, macro): "
        + ", ".join(f"{k} {_fmt(v)}" for k, v in s["by_split"].items()),
        "",
        f"Abstention: precision {_fmt(s['abstention']['precision'])}, recall "
        f"{_fmt(s['abstention']['recall'])}, false-refusal rate "
        f"{_fmt(s['abstention']['false_refusal_rate'])}. Stale-value leak rate "
        f"{_fmt(s['stale_leak']['rate'])} ({s['stale_leak']['leaked']} of "
        f"{s['stale_leak']['phrasings_with_stale_values']}).",
        "",
    ]
    ing = meta.get("ingestion") or {}
    if ing:
        lines += [
            f"Ingestion: {ing.get('turns')} turns, {ing.get('jobs')} jobs "
            f"({ing.get('failed_jobs')} failed), written in {ing.get('write_seconds')} s; settled "
            f"{(ing.get('settle') or {}).get('seconds')} s after the last write. "
            f"Worker queues at the start: {_queues(ing.get('queues_at_start'))}; enrichment "
            f"still queued when the questions started: "
            f"{_queues((ing.get('settle') or {}).get('queued_at_ask'))}.",
            "",
        ]
    lines += ["## Top failure reasons", ""]
    for cap, f in p["failures"].items():
        if not f["failed"]:
            continue
        reasons = ", ".join(f"{k} {v}" for k, v in f["reasons"].items())
        lines.append(f"**{cap}** — {f['failed']}/{f['of']} failed: {reasons}")
        for ex in f["examples"][:2]:
            got = "; ".join(ex["got"][:2]) or "nothing"
            lines.append(
                f"- `{ex['key']}` ({ex['reason']}): “{ex['question']}” — missing "
                f"{ex['missing'] or '-'}, leaked {ex['leaks'] or '-'}; got: {got[:300]}"
            )
        lines.append("")
    for surface, failures in (p.get("failures_by_surface") or {}).items():
        lines += [f"## Failure reasons on {surface}", ""]
        for cap, f in failures.items():
            if not f["failed"]:
                continue
            reasons = ", ".join(f"{k} {v}" for k, v in f["reasons"].items())
            lines.append(f"**{cap}** — {f['failed']}/{f['of']} failed: {reasons}")
            for ex in f["examples"][:1]:
                got = "; ".join(ex["got"][:2]) or "nothing"
                lines.append(
                    f"- `{ex['key']}` ({ex['reason']}): “{ex['question']}” — missing "
                    f"{ex['missing'] or '-'}, leaked {ex['leaks'] or '-'}; got: {got[:300]}"
                )
        lines.append("")
    leaks = s.get("stale_leak_by_surface") or {}
    if leaks:
        lines += [
            "Stale-value leak rate by surface: "
            + ", ".join(
                f"{k} {_fmt(v['rate'])} ({v['leaked']}/{v['phrasings_with_stale_values']})"
                for k, v in leaks.items()
            ),
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m benchmark.capabilities.report <run>.raw.json [out.json]")
        return 2
    json_path, md_path = rebuild(Path(args[0]), Path(args[1]) if len(args) > 1 else None)
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
