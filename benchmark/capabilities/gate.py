"""The capability gate: a ratchet. Accuracy may only go up.

    python -m benchmark.capabilities.gate record <run.json>...
    python -m benchmark.capabilities.gate check  <run.json> [--rerun <second run.json>]

``record`` writes ``benchmark/results/capability_gate.json``. For each mode (or gate set),
every cell (surface x placement x capability x language) gets a threshold: its best recorded
accuracy. Repeats recorded together merge first, each cell's lowest (noise is not progress),
and the cells they disagreed on are kept as noisy with the range they spanned. Then the
threshold is the higher of that value and the one already recorded for the same questions,
so a re-record can only raise it; a cell measured below it is refused. ``targets`` are goals
beside the thresholds (the pre-B1 values of cells B1 lowered): reported, never failed.

``check`` holds a run to it:

* a cell below its floor fails: the threshold, less the noise its repeats showed only when
  the threshold is a carried best value above the measured one (a threshold that is already
  the lower repeat has had its noise allowed once). On a noisy cell it asks for one rerun
  first (exit 3), and fails if the rerun's own value is below the floor too;
* a cell the baseline has and the run does not fails; a cell the run has and the baseline
  does not is reported, and gated once recorded;
* context p95 may not rise more than 10 % over the baseline's; a latency measured on a
  loaded machine fails the latency check;
* a run on another dataset (the questions, the scorer, the generator) or another selection
  is refused: re-record in the same change, where a reviewer sees both numbers. Any byte
  change to scoring.py, generate.py or the language data resets every threshold; a change
  to one capability's file resets that capability's.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.common import RESULTS

GATE = RESULTS / "capability_gate.json"
#: context p95 may rise at most this much over the recorded baseline
LATENCY_RISE_MAX = 0.10


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cells(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{c['surface']}/{c['placement']}/{c['capability']}/{c['language']}": {
            "accuracy": c["accuracy"],
            "n": c["n"],
        }
        for c in run["cells"]
    }


def key_of(run: Mapping[str, Any]) -> str:
    """The baseline a run is held to: its mode, or ``mode/set`` for a named gate set
    (``nollm/core``). A mode's own sample (``llm`` in llm mode) is the mode's baseline."""
    mode, sample = run.get("mode"), (run.get("meta") or {}).get("sample")
    return str(mode) if sample in (None, mode) else f"{mode}/{sample}"


def _subset_of(run: Mapping[str, Any]) -> str | None:
    """A set gated against another baseline's matching cells (``"gate": "subset_of"`` in
    ``_modes.json``): the scale set is held to the full run's scale cells."""
    sample = (run.get("meta") or {}).get("sample")
    if not sample:
        return None
    from benchmark.capabilities.dataset import load_json

    spec = load_json("_modes.json")["samples"].get(sample) or {}
    return spec.get("subset_of")


def baseline_of(run: Mapping[str, Any], source: str) -> dict[str, Any]:
    lat = run["summary"]["latency_ms"]
    return {
        "source": source,
        "run_date": run["meta"].get("run_date"),
        "selection": run["meta"].get("selection"),
        "sample": run["meta"].get("sample"),
        "scale": run["meta"].get("scale"),
        "macro_accuracy": run["summary"]["macro_accuracy"],
        "retail_track": run["summary"]["retail_track"],
        "cells": _cells(run),
        "latency": {
            "context_p50_ms": lat.get("context_p50"),
            "context_p95_ms": lat.get("context_p95"),
            "trustworthy": lat.get("trustworthy"),
            "max_load_per_core": lat.get("max_load_per_core"),
            "max_rise": LATENCY_RISE_MAX,
        },
        "git_commit": (run.get("provenance") or {}).get("git_commit"),
    }


def _merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Two runs of one mode on one dataset: each cell's threshold is the lower of the two, and
    the latency is the quietest trustworthy one. Repeated runs are how the gate learns a cell's
    run-to-run noise without a blanket tolerance (it has been one cell in 192, by 0.03)."""
    if base["selection"] != other["selection"]:
        raise SystemExit(f"repeat runs of {base['source']} differ in selection")
    cells = dict(base["cells"])
    noisy: dict[str, list[float]] = {k: list(v) for k, v in base.get("noisy", {}).items()}
    for key, cell in other["cells"].items():
        mine = cells.get(key)
        if mine is not None and cell["accuracy"] != mine["accuracy"]:
            values = [
                v
                for v in (cell["accuracy"], mine["accuracy"], *noisy.get(key, []))
                if v is not None
            ]
            noisy[key] = [min(values), max(values)]
        if mine is None or (
            cell["accuracy"] is not None and cell["accuracy"] < (mine["accuracy"] or 0)
        ):
            cells[key] = cell
    # the quietest trustworthy measurement (a latency-only re-measure on a quiet machine
    # replaces one taken beside other work); between equally quiet ones, the slower
    lats = [b["latency"] for b in (base, other) if b["latency"].get("trustworthy")]
    latency = (
        min(
            lats,
            key=lambda x: (x.get("max_load_per_core") or 0.0, -(x.get("context_p95_ms") or 0)),
        )
        if lats
        else base["latency"]
    )
    sources = [*base.get("sources", [base["source"]]), other["source"]]
    return {
        **base,
        "cells": cells,
        #: cells the repeats disagreed on, [lowest, highest]: a drop on one of them alone asks
        #: for one rerun before it counts (``check(rerun=)``)
        "noisy": dict(sorted(noisy.items())),
        "latency": latency,
        "sources": sources,
        "repeat_runs": len(sources),
    }


def _carries(existing: Mapping[str, Any], run: Mapping[str, Any]) -> Any:
    """Whether a cell's earlier threshold still holds for this run's questions: the same
    dataset, or (when both sides list their files) the same capability file and the same
    scorer, generator and language data. A cell whose questions changed starts afresh."""
    if existing.get("dataset_sha256") == run["dataset"]["sha256"]:
        return lambda _cell: True
    old, new = existing.get("files") or {}, run["dataset"].get("files") or {}
    if not old or not new:
        return lambda _cell: False
    shared = [f for f in new if not f[:2].isdigit() and not f.startswith("scale_")]
    if any(old.get(f) != new.get(f) for f in shared):
        return lambda _cell: False
    by_capability = {}
    for f in new:
        if f[:2].isdigit():
            by_capability[f.split("_", 1)[1].removesuffix(".json")] = old.get(f) == new.get(f)
    return lambda cell: by_capability.get(cell.split("/")[2], False)


def _latency_gated(run: Mapping[str, Any]) -> bool:
    """False for a set whose read path runs a model on the measuring host (``_modes.json``
    ``latency_gated: false``): its latency is recorded and reported, not gated."""
    sample = (run.get("meta") or {}).get("sample")
    if not sample:
        return True
    from benchmark.capabilities.dataset import load_json

    return (load_json("_modes.json")["samples"].get(sample) or {}).get("latency_gated", True)


def _spread(base: Mapping[str, Any], key: str) -> float:
    lo, hi = (base.get("noisy") or {}).get(key, [0.0, 0.0])
    return round(hi - lo, 4)


def record(paths: Sequence[Path], gate: Path = GATE) -> dict[str, Any]:
    """Record runs, ratcheting. Repeats of one mode merge first (``_merge``: each cell's
    measured value is the lowest of them, the cells they disagreed on are noisy). Then each
    cell's threshold is its best recorded value: the higher of the measured value and the
    threshold already recorded for the same questions. Accuracy may only go up: a cell
    measured below its threshold, beyond its recorded noise, is refused. ``targets`` (goals
    beside the thresholds, reported and never failed) are kept."""
    existing = _load(gate) if gate.is_file() else {}
    runs = [(p, _load(p)) for p in paths]
    for path, run in runs:
        if not run.get("cells") or not (run.get("records") or run["summary"].get("surfaces")):
            raise SystemExit(f"{path}: the run measured nothing; it is not a baseline")
        if (run.get("meta") or {}).get("answer_errors"):
            raise SystemExit(f"{path}: model calls failed in this run; it is not a baseline")
    shas = {r["dataset"]["sha256"] for _, r in runs}
    if len(shas) != 1:
        raise SystemExit(f"runs measured on different datasets: {sorted(shas)}")
    sha = shas.pop()
    carries = _carries(existing, runs[0][1])
    fresh: dict[str, dict[str, Any]] = {}
    for path, run in runs:
        if _subset_of(run):
            raise SystemExit(f"{path}: this set is gated against {_subset_of(run)!r}'s cells")
        b = baseline_of(run, str(path))
        key = key_of(run)
        fresh[key] = _merge(fresh[key], b) if key in fresh else b
    previous = existing.get("baselines") or {}
    baselines = {k: v for k, v in previous.items() if existing.get("dataset_sha256") == sha}
    regressions: list[str] = []
    for key, b in fresh.items():
        prev = previous.get(key) or {}
        prev_cells = prev.get("cells") or {}
        noisy = {k: v for k, v in (prev.get("noisy") or {}).items() if carries(k)}
        for k, (lo, hi) in (b.get("noisy") or {}).items():
            old = noisy.get(k, [lo, hi])
            noisy[k] = [min(lo, old[0]), max(hi, old[1])]
        cells: dict[str, dict[str, Any]] = {}
        for k, cell in b["cells"].items():
            measured = cell["accuracy"]
            before = (prev_cells.get(k) or {}).get("accuracy") if carries(k) else None
            threshold = measured if before is None or measured is None else max(measured, before)
            if before is not None and measured is not None:
                spread = noisy.get(k, [0.0, 0.0])[1] - noisy.get(k, [0.0, 0.0])[0]
                if measured < before - spread - 1e-9:
                    regressions.append(f"{key} {k}: {measured} < threshold {before}")
            cells[k] = {**cell, "accuracy": threshold, "measured": measured}
        baselines[key] = {**b, "cells": cells, "noisy": dict(sorted(noisy.items()))}
    if regressions:
        raise SystemExit(
            "would record a regression (accuracy may only go up):\n  " + "\n  ".join(regressions)
        )
    from benchmark.capabilities.report import _provenance

    out = {
        "gate": "capability",
        "dataset_sha256": sha,
        "files": runs[0][1]["dataset"].get("files") or {},
        "rule": "ratchet: every cell >= its best recorded accuracy, less its recorded noise; "
        "context p95 <= recorded x (1 + max_rise)",
        "baselines": dict(sorted(baselines.items())),
        #: goals beside the thresholds, per baseline: reported when a run is below them, never
        #: failed on
        "targets": existing.get("targets", {}),
        "provenance": _provenance(),
    }
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def check(
    run: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    allow_untrusted_latency: bool = False,
    rerun: Mapping[str, Any] | None = None,
) -> list[str]:
    """Failures of ``run`` against the recorded baseline of its mode (or gate set); empty is
    a pass. A ``subset_of`` set is held to that baseline's cells it measured, and its latency
    (a different mix of questions) is not compared."""
    subset = _subset_of(run)
    key = subset or key_of(run)
    base = (gate.get("baselines") or {}).get(key)
    if base is None:
        return [f"no recorded baseline {key!r}"]
    failures: list[str] = []
    if run["dataset"]["sha256"] != gate.get("dataset_sha256"):
        return [
            "dataset changed since the baseline was recorded "
            f"({run['dataset']['sha256'][:12]} != {str(gate.get('dataset_sha256'))[:12]}): "
            "re-record the baseline in the same change"
        ]
    if not subset and run["meta"].get("selection") != base.get("selection"):
        return [
            f"selection differs from the baseline's: {run['meta'].get('selection')} != {base.get('selection')}"
        ]
    cells = _cells(run)
    if not cells:
        return ["the run measured no cells"]
    if rerun is not None:
        same = {
            "dataset": (rerun["dataset"]["sha256"], run["dataset"]["sha256"]),
            "selection": (rerun["meta"].get("selection"), run["meta"].get("selection")),
            "mode": (rerun.get("mode"), run.get("mode")),
            "code": (
                (rerun.get("provenance") or {}).get("git_commit"),
                (run.get("provenance") or {}).get("git_commit"),
            ),
        }
        differs = [k for k, (a, b) in same.items() if a != b]
        if differs:
            return [f"the rerun is not a rerun of this run: its {', '.join(differs)} differ"]
    again = _cells(rerun) if rerun is not None else None
    pending: list[str] = []
    if subset:
        unknown = sorted(set(cells) - set(base["cells"]))
        if unknown:
            return [f"cells the {subset!r} baseline does not hold: {unknown[:5]}"]
    for key, want in sorted(base["cells"].items()):
        got = cells.get(key)
        if got is None and subset:
            continue
        if got is None:
            failures.append(f"{key}: not measured (baseline {want['accuracy']} over {want['n']})")
        elif got["accuracy"] is None or want["accuracy"] is None:
            continue
        else:
            # the threshold; less the noise the repeats showed only when the threshold is a
            # carried best value above what was measured - a threshold that already is the
            # lower of the repeats has had its noise allowed once
            spread = _spread(base, key)
            measured = want.get("measured")
            carried = measured is not None and want["accuracy"] > measured + 1e-9
            floor = want["accuracy"] - spread if carried else want["accuracy"]
            if got["accuracy"] >= floor - 1e-9:
                continue
            line = f"{key}: accuracy {got['accuracy']} < floor {round(floor, 4)} (n={got['n']})"
            if not spread:
                failures.append(line)
            elif again is None:
                pending.append(line)
            elif ((again.get(key) or {}).get("accuracy") or 0.0) < floor - 1e-9:
                # the one rerun is below it too: a regression, not noise
                failures.append(f"{line}; rerun {(again.get(key) or {}).get('accuracy')}")
    if pending and not failures:
        # only cells the baseline's repeats disagreed on dropped: one rerun decides
        failures += [f"{RERUN}{line}" for line in pending]
    elif pending:
        failures += pending
    lat = run["summary"]["latency_ms"]
    p95, base_p95 = lat.get("context_p95"), base["latency"].get("context_p95_ms")
    if p95 is None or subset or not _latency_gated(run):
        # no context was asked (harness mode scores answers): there is no latency to gate -
        # the answer's includes the model, which on a laptop is the CPU's main load itself
        pass
    elif not lat.get("trustworthy") and not allow_untrusted_latency:
        failures.append(
            f"context latency measured at load {lat.get('max_load_per_core')} per core: not a "
            "measurement; rerun on a quiet machine (under the heavy-job lock)"
        )
    elif p95 is not None and base_p95:
        ceiling = round(base_p95 * (1 + base["latency"].get("max_rise", LATENCY_RISE_MAX)), 2)
        if p95 > ceiling:
            failures.append(f"context p95 {p95} ms > {ceiling} ms (baseline {base_p95} ms + 10%)")
    return failures


#: the prefix of a failure that one rerun may clear (``gate check run.json --rerun run2.json``)
RERUN = "RERUN (noisy cell; run once more and pass it with --rerun): "


def new_cells(run: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    """Cells the run measured that the baseline does not hold yet: reported, not gated."""
    base = (gate.get("baselines") or {}).get(_subset_of(run) or key_of(run)) or {}
    if not base:
        return []
    return sorted(set(_cells(run)) - set(base.get("cells") or {}))


def below_targets(run: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    """Cells under a target (``gate["targets"][key]["cells"]``): reported, not a failure."""
    target = (gate.get("targets") or {}).get(_subset_of(run) or key_of(run)) or {}
    cells = _cells(run)
    out = []
    for key, want in sorted((target.get("cells") or {}).items()):
        got = (cells.get(key) or {}).get("accuracy")
        if got is not None and got < want:
            out.append(f"{key}: {got} below target {want}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record these runs as the baseline of their modes")
    r.add_argument("results", nargs="+", type=Path)
    c = sub.add_parser("check", help="fail on any drop against the recorded baseline")
    c.add_argument("results", nargs="+", type=Path)
    c.add_argument("--allow-untrusted-latency", action="store_true")
    c.add_argument(
        "--rerun", type=Path, default=None, help="the one rerun a noisy-cell drop asks for"
    )
    a = p.parse_args(argv)
    if a.cmd == "record":
        out = record(a.results)
        for mode, b in out["baselines"].items():
            print(
                f"recorded {mode}: macro {b['macro_accuracy']}, {len(b['cells'])} cells, "
                f"context p95 {b['latency']['context_p95_ms']} ms"
            )
        return 0
    if not GATE.is_file():
        print(f"{GATE} missing: no baseline recorded (missing evidence is a failed gate)")
        return 1
    gate = _load(GATE)
    rerun = _load(a.rerun) if a.rerun else None
    status = 0
    for path in a.results:
        run = _load(path)
        failures = check(run, gate, allow_untrusted_latency=a.allow_untrusted_latency, rerun=rerun)
        needs_rerun = bool(failures) and all(f.startswith(RERUN) for f in failures)
        verdict = "RERUN" if needs_rerun else "FAIL" if failures else "PASS"
        print(f"CAPABILITY GATE {path}: {verdict}")
        for c in new_cells(run, gate):
            print(f"  + new cell (not gated until recorded): {c}")
        for t in below_targets(run, gate):
            print(f"  ~ below target (not failed): {t}")
        for f in failures:
            print(f"  - {f}")
        status = max(status, 3 if needs_rerun else 1 if failures else 0)
    # 3: only noisy cells dropped - run once more and check again with --rerun
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
