"""The capability no-regression gate: the recorded baseline is the threshold.

    python -m benchmark.capabilities.gate record benchmark/results/capabilities/nollm.json [...]
    python -m benchmark.capabilities.gate check  benchmark/results/capabilities/nollm.json

``record`` writes ``benchmark/results/capability_gate.json``: for each mode, the accuracy of
every cell (surface x placement x capability x language) and the context latency p95 of the
run it was measured in. ``check`` holds a new run of the same mode to it:

* any cell whose accuracy is below its recorded value fails (no tolerance: a drop is a drop;
  ``record`` given repeated runs of a mode takes each cell's lowest, which is how the gate
  absorbs measured run-to-run noise and nothing more);
* a cell the baseline has and the run does not fails (missing evidence is a failed gate);
* context p95 may not rise more than 10 % over the baseline's;
* a latency measured while the machine was loaded (``latency_trustworthy: false``) fails the
  latency check instead of passing it silently - rerun on a quiet machine, under the lock;
* a run on another dataset (``dataset.sha256``) or another selection is refused outright:
  thresholds hold only for the questions they were measured on. Changing the dataset means
  re-recording the baseline in the same change, where a reviewer sees both numbers.

Never lower a threshold to make a change pass (CONTRIBUTING: never weaken a gate); a real
improvement is recorded by ``record``, which only ever replaces the baseline wholesale.
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
    for key, cell in other["cells"].items():
        mine = cells.get(key)
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
        "latency": latency,
        "sources": sources,
        "repeat_runs": len(sources),
    }


def record(paths: Sequence[Path], gate: Path = GATE) -> dict[str, Any]:
    """The baseline for every mode given, replacing those modes' previous baselines. Several
    runs of one mode merge (``_merge``): the threshold is what every one of them reached."""
    existing = _load(gate) if gate.is_file() else {}
    runs = [(p, _load(p)) for p in paths]
    for path, run in runs:
        if (run.get("meta") or {}).get("answer_errors"):
            raise SystemExit(f"{path}: model calls failed in this run; it is not a baseline")
    shas = {r["dataset"]["sha256"] for _, r in runs}
    if len(shas) != 1:
        raise SystemExit(f"runs measured on different datasets: {sorted(shas)}")
    sha = shas.pop()
    baselines = dict(existing.get("baselines", {})) if existing.get("dataset_sha256") == sha else {}
    fresh: dict[str, dict[str, Any]] = {}
    for path, run in runs:
        if _subset_of(run):
            raise SystemExit(f"{path}: this set is gated against {_subset_of(run)!r}'s cells")
        b = baseline_of(run, str(path))
        key = key_of(run)
        fresh[key] = _merge(fresh[key], b) if key in fresh else b
    baselines.update(fresh)
    from benchmark.capabilities.report import _provenance

    out = {
        "gate": "capability",
        "dataset_sha256": sha,
        "rule": "every cell >= its recorded accuracy; context p95 <= recorded x (1 + max_rise)",
        "baselines": dict(sorted(baselines.items())),
        "provenance": _provenance(),
    }
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def check(
    run: Mapping[str, Any], gate: Mapping[str, Any], *, allow_untrusted_latency: bool = False
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
        elif got["accuracy"] < want["accuracy"]:
            failures.append(
                f"{key}: accuracy {got['accuracy']} < baseline {want['accuracy']} (n={got['n']})"
            )
    lat = run["summary"]["latency_ms"]
    p95, base_p95 = lat.get("context_p95"), base["latency"].get("context_p95_ms")
    if p95 is None or subset:
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
    failed = False
    for path in a.results:
        failures = check(_load(path), gate, allow_untrusted_latency=a.allow_untrusted_latency)
        print(f"CAPABILITY GATE {path}: {'FAIL' if failures else 'PASS'}")
        for f in failures:
            print(f"  - {f}")
        failed |= bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
