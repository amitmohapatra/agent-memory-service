"""Static checks on the dataset: every assertion is satisfiable by what the scenario writes,
in every language it runs in. A gate measured on an assertion no text can meet would read as
a capability failure forever.

    python -m benchmark.capabilities.validate
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from typing import Any

from benchmark.capabilities.dataset import (
    DOMAINS,
    LANGUAGES,
    PHRASING_KINDS,
    ROLES,
    Dataset,
    Scenario,
    load_dataset,
    load_json,
    text_for,
)
from benchmark.capabilities.scoring import expectations, group, normalize, resolve


def _setup_texts(s: Scenario, lang: str) -> list[str]:
    out = []
    for session in s.sessions:
        out += [text_for(t.text, lang) for t in session.turns]
        if session.repeat:
            out.append(text_for(session.repeat.text, lang))
        for run in session.tool_runs:
            out.append(" ".join(c["tool"] for c in run.calls))
    return out


def _anchors(expect: Mapping[str, Any], key: str) -> Iterable[Any]:
    for raw in expect.get(key, ()):
        if key == "absent":
            spec = raw if isinstance(raw, Mapping) and "text" in raw else {"text": raw}
            yield spec["text"]
        else:
            yield from group(raw)


def problems(dataset: Dataset) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    modes = load_json("_modes.json")
    load_json("_languages.json")
    if set(modes["modes"]) != {"nollm", "llm", "harness"}:
        out.append("_modes.json must define nollm, llm and harness")
    for cap in dataset.capabilities:
        domains = {s.domain for s in cap.scenarios}
        if domains != set(DOMAINS):
            out.append(f"{cap.name}: domains {sorted(domains)} != {sorted(DOMAINS)}")
        if not any(s.origin for s in cap.scenarios):
            out.append(f"{cap.name}: no origin (InventorySmart) scenario")
        for s in cap.scenarios:
            where = f"{cap.name}/{s.id}"
            if s.id in seen:
                out.append(f"{where}: duplicate scenario id")
            seen.add(s.id)
            if s.domain not in DOMAINS:
                out.append(f"{where}: unknown domain {s.domain}")
            for session in s.sessions:
                if session.day >= 0:
                    out.append(f"{where}/{session.id}: day {session.day} is not in the past")
                for t in session.turns:
                    if t.role not in ROLES:
                        out.append(f"{where}/{session.id}: role {t.role}")
                    if set(LANGUAGES) - set(t.text):
                        out.append(
                            f"{where}/{session.id}: turn missing {set(LANGUAGES) - set(t.text)}"
                        )
            if not s.sessions and not s.corpus:
                out.append(f"{where}: nothing is written")
            for q in s.queries:
                kinds = [p.kind for p in q.phrasings]
                if sorted(kinds) != sorted(PHRASING_KINDS):
                    out.append(f"{where}/{q.id}: phrasing kinds {kinds}")
                for p in q.phrasings:
                    if set(LANGUAGES) - set(p.text):
                        out.append(
                            f"{where}/{q.id}/{p.kind}: missing {set(LANGUAGES) - set(p.text)}"
                        )
                if s.corpus:
                    if "aggregate" not in q.expect:
                        out.append(f"{where}/{q.id}: a corpus query needs an aggregate")
                    continue
                for lang in s.languages:
                    setup = normalize(" \n ".join(_setup_texts(s, lang)))
                    for surface in ("context", "answer"):
                        e = expectations(q.expect, surface)
                        keys = ("contains", "absent") if surface == "context" else ()
                        for key in keys:
                            for anchor in _anchors(e, key):
                                try:
                                    pat = resolve(anchor, lang)
                                except KeyError:
                                    out.append(f"{where}/{q.id}: anchor {anchor!r} has no {lang}")
                                    continue
                                if pat.regex is None and not pat.matches(setup):
                                    out.append(
                                        f"{where}/{q.id}/{lang}: {key} anchor {pat.label!r} is in no setup turn"
                                    )
                        if e.get("abstain") and e.get("contains"):
                            out.append(f"{where}/{q.id}: abstain with contains")
    return out


def main() -> int:
    found = problems(load_dataset())
    for p in found:
        print(p)
    print(f"{len(found)} problems")
    return 1 if found else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
