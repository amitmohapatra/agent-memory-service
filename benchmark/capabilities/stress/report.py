"""Summaries of a stress run, and a paired comparison of two runs.

    python -m benchmark.capabilities.stress.report A.json B.json --names main,cand [--out cmp.md]

Rates are over the records of a cell (one record = one phrasing x language x placement):

* ``pass``            the verdict passed (``judge.py``);
* ``stale_leak``      answerable records whose context served an earlier value wrongly
                      (``stale_first`` or ``stale_current``);
* ``false_refusal``   answerable records the service called ``INSUFFICIENT``;
* ``missing``         answerable records lacking a group of the final run's values;
* ``correct_refusal`` never-stated records the service called ``INSUFFICIENT``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.capabilities.stress.build import ALL_CATEGORIES

PLACEMENTS = ("same_session", "new_session")


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def cell(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rs = list(records)
    ans = [r for r in rs if not r.get("unknowable")]
    abst = [r for r in rs if r.get("unknowable")]
    ok = [r for r in ans if r.get("reason") != "error"]
    return {
        "n": len(rs),
        "pass_rate": _rate(sum(1 for r in rs if r["passed"]), len(rs)),
        "answerable": len(ans),
        "stale_leak_rate": _rate(sum(1 for r in ok if r.get("stale_leak")), len(ans)),
        "stale_first_rate": _rate(sum(1 for r in ok if r.get("stale_first")), len(ans)),
        "stale_current_rate": _rate(sum(1 for r in ok if r.get("stale_current")), len(ans)),
        "false_refusal_rate": _rate(sum(1 for r in ok if r.get("abstained")), len(ans)),
        "missing_rate": _rate(sum(1 for r in ok if r.get("missing")), len(ans)),
        "abstain_n": len(abst),
        "correct_refusal_rate": _rate(sum(1 for r in abst if r.get("abstained")), len(abst)),
        "near_served_rate": _rate(sum(1 for r in abst if r.get("near_served")), len(abst)),
        "errors": sum(1 for r in rs if r.get("reason") == "error"),
    }


def _group(
    records: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Any]
) -> dict[str, Any]:
    out: dict[Any, list[Mapping[str, Any]]] = {}
    for r in records:
        out.setdefault(key(r), []).append(r)
    return {"/".join(k) if isinstance(k, tuple) else k: cell(v) for k, v in sorted(out.items())}


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": cell(records),
        "by_placement": _group(records, lambda r: r["placement"]),
        "by_category_placement": _group(records, lambda r: (r["category"], r["placement"])),
        "by_language_placement": _group(records, lambda r: (r["language"], r["placement"])),
        "by_split_placement": _group(records, lambda r: (r["split"], r["placement"])),
        "by_script": _group(records, lambda r: r["script"]),
        "reasons": _group(records, lambda r: r.get("reason") or "pass"),
    }


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def to_markdown(payload: Mapping[str, Any]) -> str:
    m, s = payload["meta"], payload["summary"]
    lines = [
        "# stress_accumulated (nollm, context surface)",
        "",
        f"service {m.get('service_rev')} ({m.get('service_root')}), run date {m['run_date']}, seed {m['seed']}, "
        f"{m['prior_runs']} earlier runs per script, languages {','.join(m['languages'])}, split {m['split']}",
        f"ingestion: {m['ingestion']['turns']} turns, failed jobs {m['ingestion']['failed_jobs']}, "
        f"failed writes {len(m['ingestion']['failed_writes'])}; llm calls: {json.dumps(m.get('llm_calls'))[:200]}",
        "",
        "| category | placement | n | pass % | stale-leak % | stale-first % | stale-as-current % | false-refusal % | missing % | correct-refusal % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    cp = s["by_category_placement"]
    for cat in ALL_CATEGORIES:
        for pl in PLACEMENTS:
            c = cp.get(f"{cat}/{pl}")
            if not c:
                continue
            lines.append(
                f"| {cat} | {pl} | {c['n']} | {_pct(c['pass_rate'])} | {_pct(c['stale_leak_rate'])} | "
                f"{_pct(c['stale_first_rate'])} | {_pct(c['stale_current_rate'])} | {_pct(c['false_refusal_rate'])} | "
                f"{_pct(c['missing_rate'])} | {_pct(c['correct_refusal_rate'])} |"
            )
    for pl, c in s["by_placement"].items():
        lines.append(
            f"| **all** | {pl} | {c['n']} | {_pct(c['pass_rate'])} | {_pct(c['stale_leak_rate'])} | "
            f"{_pct(c['stale_first_rate'])} | {_pct(c['stale_current_rate'])} | {_pct(c['false_refusal_rate'])} | "
            f"{_pct(c['missing_rate'])} | {_pct(c['correct_refusal_rate'])} |"
        )
    lines += [
        "",
        "| language/placement | pass % | stale-leak % | false-refusal % | correct-refusal % |",
        "|---|---|---|---|---|",
    ]
    for k, c in s["by_language_placement"].items():
        lines.append(
            f"| {k} | {_pct(c['pass_rate'])} | {_pct(c['stale_leak_rate'])} | {_pct(c['false_refusal_rate'])} | {_pct(c['correct_refusal_rate'])} |"
        )
    lines += [
        "",
        "| split/placement | pass % | stale-leak % | false-refusal % | correct-refusal % |",
        "|---|---|---|---|---|",
    ]
    for k, c in s["by_split_placement"].items():
        lines.append(
            f"| {k} | {_pct(c['pass_rate'])} | {_pct(c['stale_leak_rate'])} | {_pct(c['false_refusal_rate'])} | {_pct(c['correct_refusal_rate'])} |"
        )
    lines += ["", "reasons: " + ", ".join(f"{k} {v['n']}" for k, v in s["reasons"].items()), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- paired compare


def compare(a: Mapping[str, Any], b: Mapping[str, Any], names: Sequence[str]) -> str:
    na, nb = names
    ma, mb = a["meta"], b["meta"]
    same = all(
        ma[k] == mb[k] for k in ("run_date", "seed", "prior_runs", "languages", "scripts", "split")
    )
    lines = [
        f"# stress_accumulated: {na} vs {nb} (nollm, context surface)",
        "",
        f"{na}: {ma.get('service_rev')}  {nb}: {mb.get('service_rev')}  run date {ma['run_date']}/{mb['run_date']}, "
        f"seed {ma['seed']}, {ma['prior_runs']} earlier runs, languages {','.join(ma['languages'])}"
        + (
            ""
            if same
            else "  **WARNING: the two runs differ in run date, seed, runs, languages or split - not paired**"
        ),
        "",
        f"| category | placement | n | pass % {na} | pass % {nb} | stale-leak % {na} | stale-leak % {nb} | "
        f"false-refusal % {na} | false-refusal % {nb} | correct-refusal % {na} | correct-refusal % {nb} |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ca, cb = a["summary"]["by_category_placement"], b["summary"]["by_category_placement"]

    def row(label: str, pl: str, x: Mapping[str, Any], y: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {pl} | {x['n']} | {_pct(x['pass_rate'])} | {_pct(y['pass_rate'])} | "
            f"{_pct(x['stale_leak_rate'])} | {_pct(y['stale_leak_rate'])} | {_pct(x['false_refusal_rate'])} | "
            f"{_pct(y['false_refusal_rate'])} | {_pct(x['correct_refusal_rate'])} | {_pct(y['correct_refusal_rate'])} |"
        )

    for cat in ALL_CATEGORIES:
        for pl in PLACEMENTS:
            k = f"{cat}/{pl}"
            if k in ca and k in cb:
                lines.append(row(cat, pl, ca[k], cb[k]))
    for pl in PLACEMENTS:
        x, y = a["summary"]["by_placement"].get(pl), b["summary"]["by_placement"].get(pl)
        if x and y:
            lines.append(row("**all**", pl, x, y))
    # per-record flips
    ia = {(r["key"], r["placement"]): r for r in a["records"]}
    ib = {(r["key"], r["placement"]): r for r in b["records"]}
    both = sorted(set(ia) & set(ib))
    up = [k for k in both if not ia[k]["passed"] and ib[k]["passed"]]
    down = [k for k in both if ia[k]["passed"] and not ib[k]["passed"]]
    lines += [
        "",
        f"paired records {len(both)}: {nb} fixes {len(up)}, breaks {len(down)} (vs {na})",
    ]
    if down:
        lines.append("")
        lines.append(f"first breaks ({nb} fails where {na} passed):")
        for k in down[:15]:
            r = ib[k]
            lines.append(
                f"- {k[0]} [{k[1]}] {r.get('reason')} stale={r.get('stale_values')} missing={r.get('missing')}"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", help="one result .json (summary) or two (paired comparison)")
    p.add_argument("--names", default=None, help="labels for two runs, e.g. main,cand")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    payloads = [json.loads(Path(x).read_text(encoding="utf-8")) for x in a.runs]
    if len(payloads) == 1:
        text = to_markdown(payloads[0])
    elif len(payloads) == 2:
        names = a.names.split(",") if a.names else [Path(x).stem for x in a.runs]
        text = compare(payloads[0], payloads[1], names)
    else:
        p.error("one or two runs")
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
