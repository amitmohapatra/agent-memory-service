"""Generate the accumulated store: every script re-run N times with fresh values, then the final run.

A script (``scripts.json``) is one tester's scripted conversation. Production testers re-run it
day after day for the same user without clearing memory, and each run draws new values: the
store's DC, the buyer, the standing rule's unlock phrase, the audit's columns. This module draws
those values (deterministically, from the seed), lays the runs out on a timeline, and turns them
into the capability eval's own ``Scenario`` (so ``runner.write_scenario`` writes them) plus, per
question and language, what the context must hold (the final run's values) and what it must not
serve as current (every earlier value of the same slot).

Determinism: the values are a function of (seed, script id, run index) alone; the final run is
drawn first, so its values (the answers) do not depend on how many earlier runs are written.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.capabilities.dataset import Phrasing, Query, Scenario, Session, Turn
from benchmark.capabilities.scoring import Pattern, normalize

SCRIPTS = Path(__file__).resolve().parent / "scripts.json"
DEFAULT_SEED = "stress-v1"
DEFAULT_PRIOR_RUNS = 6
MAX_PRIOR_RUNS = 10
#: the categories every script covers, in report order
CATEGORIES = (
    "single_fact",
    "multi_hop_2",
    "multi_hop_3",
    "update_replacement",
    "recency",
    "abstention",
    "standing_rule",
    "format_rule",
    "preference",
)
#: categories whose question a tester asked in every earlier run too (the run's own Q/A pair is
#: part of the history); a rule, format or preference question gets a long answer, not a value
ECHOED = frozenset(CATEGORIES) - {"standing_rule", "format_rule", "preference"}
SPLITS = ("dev", "heldout")
_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I, O: never read as 1, 0
_TOKEN = re.compile(r"\{(A|D|d|w:[a-z0-9_]+)\}")
_SLOT = re.compile(r"\{([a-z0-9_]+)\}")

Value = str | tuple[str, ...]  # invariant, or one per language (``lang_pools``)


def load(path: Path = SCRIPTS) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rng(*parts: object) -> random.Random:
    return random.Random(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest())


def text_of(value: Value, li: int) -> str:
    return value if isinstance(value, str) else value[li]


def fill(template: str, values: Mapping[str, Value], li: int) -> str:
    return _SLOT.sub(lambda m: text_of(values[m.group(1)], li), template)


# --------------------------------------------------------------------------- values


class _Drawer:
    """Draws one script's values, run after run, keeping every value distinct from every other
    value of the script (in every language) and out of the script's own wording - a value that
    matched inside another value, or inside a template word, would score as present when it is
    not."""

    def __init__(self, data: Mapping[str, Any], script: Mapping[str, Any], seed: str) -> None:
        self.pools = data["pools"]
        self.lang_pools = data["lang_pools"]
        self.n_lang = len(data["languages"])
        self.script = script
        self.seed = seed
        self.taken: list[Value] = []
        self.last_names: set[str] = set()
        words: list[str] = []
        for t in script["turns"]:
            words += t
        for q in script["queries"]:
            for texts in q["phrasings"].values():
                words += texts
        for ack in data["acks"]:
            words += ack
        words += data["no_info"]
        self.wording = [normalize(_SLOT.sub(" ", w)) for w in words]

    def _clashes(self, v: Value) -> bool:
        for li in range(self.n_lang):
            new = normalize(text_of(v, li))
            if not new:
                return True
            p_new = Pattern(new, literal=new)
            for w in self.wording:
                if p_new.matches(w):
                    return True
            for old in self.taken:
                o = normalize(text_of(old, li))
                if o == new or p_new.matches(o) or Pattern(o, literal=o).matches(new):
                    return True
        return False

    def _one(self, spec: Mapping[str, Any], rng: random.Random) -> Value:
        if "pattern" in spec:

            def tok(m: re.Match[str]) -> str:
                t = m.group(1)
                if t == "A":
                    return rng.choice(_LETTERS)
                if t == "D":
                    return str(rng.randint(1, 9))
                if t == "d":
                    return str(rng.randint(0, 9))
                return rng.choice(self.pools[t[2:]])

            return _TOKEN.sub(tok, spec["pattern"])
        if spec.get("name"):
            first = rng.choice(self.pools["first"])
            last = rng.choice([x for x in self.pools["last"] if x not in self.last_names])
            return f"{first} {last}"
        if "pool" in spec:
            return rng.choice(self.pools[spec["pool"]])
        if "lang_pool" in spec:
            return tuple(rng.choice(self.lang_pools[spec["lang_pool"]]))
        if "pick" in spec:
            pool, k, sep = spec["pick"]
            return sep.join(rng.sample(self.pools[pool], k))
        if "int" in spec:
            lo, hi = spec["int"]
            return str(rng.randint(lo, hi))
        raise ValueError(f"unknown slot spec {spec!r}")

    def draw(self, slot: str, spec: Mapping[str, Any], run: str) -> Value:
        if "fixed" in spec:
            v = spec["fixed"]
            value: Value = v if isinstance(v, str) else tuple(v)
            self.taken.append(value)
            return value
        rng = _rng(self.seed, self.script["id"], run, slot)
        for _ in range(2000):
            value = self._one(spec, rng)
            if not self._clashes(value):
                self.taken.append(value)
                if spec.get("name"):
                    self.last_names.add(str(value).split(" ", 1)[1])
                return value
        raise RuntimeError(f"{self.script['id']}/{slot}: no distinct value left in its pool")


@dataclass
class Built:
    """One script, its runs drawn: ``runs[0]`` is the final run, ``runs[k]`` the k-th run
    before it (1 = the most recent earlier run)."""

    script: Mapping[str, Any]
    runs: list[dict[str, Value]]
    languages: tuple[str, ...]
    acks: list[list[str]]
    no_info: list[str]
    split: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return str(self.script["id"])

    @property
    def final(self) -> dict[str, Value]:
        return self.runs[0]


def draw_runs(
    data: Mapping[str, Any], script: Mapping[str, Any], prior_runs: int, seed: str
) -> list[dict[str, Value]]:
    if not 0 <= prior_runs <= MAX_PRIOR_RUNS:
        raise ValueError(f"prior runs must be 0..{MAX_PRIOR_RUNS}")
    d = _Drawer(data, script, seed)
    slots = script["slots"]
    runs: list[dict[str, Value]] = []
    once: dict[str, Value] = {}
    for k in range(prior_runs + 1):
        run: dict[str, Value] = {}
        for slot, spec in slots.items():
            if "fixed" in spec or spec.get("once"):
                if slot not in once:
                    once[slot] = d.draw(slot, spec, "once")
                run[slot] = once[slot]
            else:
                run[slot] = d.draw(slot, spec, "final" if k == 0 else f"prior-{k}")
        runs.append(run)
    return runs


def build(
    prior_runs: int = DEFAULT_PRIOR_RUNS, seed: str = DEFAULT_SEED, path: Path = SCRIPTS
) -> list[Built]:
    data = load(path)
    out = [
        Built(
            s,
            draw_runs(data, s, prior_runs, seed),
            tuple(data["languages"]),
            data["acks"],
            data["no_info"],
        )
        for s in data["scripts"]
    ]
    assign_splits(out)
    return out


def assign_splits(built: Sequence[Built]) -> None:
    """Dev / held-out halves, stratified by category: within a category the questions (of every
    script) are ordered by the hash of their id and alternate dev, held-out, dev, ... - the same
    scheme as the capability dataset (``dataset.assign_splits``)."""
    by_cat: dict[str, list[tuple[str, Built, str]]] = {}
    for b in built:
        for q in b.script["queries"]:
            key = hashlib.sha256(f"{b.id}/{q['id']}".encode()).hexdigest()
            by_cat.setdefault(q["category"], []).append((key, b, q["id"]))
    for items in by_cat.values():
        for i, (_, b, qid) in enumerate(sorted(items, key=lambda x: x[0])):
            b.split[qid] = SPLITS[i % 2]


# --------------------------------------------------------------------------- the timeline


def run_times(prior_runs: int) -> list[tuple[int, str]]:
    """(day, HH:MM) per run, index as ``Built.runs``: the final run the day before the run
    date at 15:00; the most recent earlier run the same day at 09:30 (testers re-run within a
    day too); the others spread back over earlier days, some days skipped."""
    out = [(-1, "15:00"), (-1, "09:30")]
    for k in range(2, prior_runs + 1):
        out.append((-k - k // 3, f"{8 + (k * 3) % 7:02d}:{(k * 17) % 60:02d}"))
    return out[: prior_runs + 1]


def _ml(texts: Sequence[str], languages: Sequence[str]) -> dict[str, str]:
    return dict(zip(languages, texts, strict=True))


def _values_ml(values: Sequence[Value], languages: Sequence[str]) -> dict[str, str]:
    return {lang: ", ".join(text_of(v, li) for v in values) for li, lang in enumerate(languages)}


def session(b: Built, k: int, day: int, at: str) -> Session:
    """Run ``k`` as one conversation: every statement (each acknowledged), then - in an earlier
    run - the questions the tester asked and the answers the agent gave that day."""
    langs = b.languages
    values = b.runs[k]
    turns: list[Turn] = []
    minute = 0
    for i, t in enumerate(b.script["turns"]):
        turns.append(
            Turn("USER", {lang: fill(t[li], values, li) for li, lang in enumerate(langs)}, minute)
        )
        turns.append(Turn("ASSISTANT", _ml(b.acks[i % len(b.acks)], langs), minute + 1))
        minute += 2
    if k > 0:
        for q in b.script["queries"]:
            if q["category"] not in ECHOED:
                continue
            ask = q["phrasings"]["direct"]
            turns.append(
                Turn(
                    "USER",
                    {lang: fill(ask[li], values, li) for li, lang in enumerate(langs)},
                    minute,
                )
            )
            if q.get("abstain"):
                answer = _ml(b.no_info, langs)
            else:
                answer = _values_ml([values[s] for s in q["contains"]], langs)
            turns.append(Turn("ASSISTANT", answer, minute + 1))
            minute += 2
    name = "run-final" if k == 0 else f"run-prior-{k:02d}"
    return Session(id=name, day=day, at=at, turns=tuple(turns))


def scenario(b: Built) -> Scenario:
    """The script as the capability eval's scenario: the runs oldest first (``write_scenario``
    writes them in time order, the final run last), and the final run's questions."""
    times = run_times(len(b.runs) - 1)
    sessions = [session(b, k, *times[k]) for k in range(len(b.runs) - 1, -1, -1)]
    langs = b.languages
    queries = []
    for q in b.script["queries"]:
        phrasings = tuple(
            Phrasing(kind, {lang: fill(texts[li], b.final, li) for li, lang in enumerate(langs)})
            for kind, texts in q["phrasings"].items()
        )
        queries.append(Query(q["id"], phrasings, {"category": q["category"]}, (), b.split[q["id"]]))
    return Scenario(
        id=f"stress-{b.id}",
        capability="stress_accumulated",
        domain=b.script["domain"],
        sessions=tuple(sessions),
        queries=tuple(queries),
        languages=langs,
    )


# --------------------------------------------------------------------------- expectations


@dataclass(frozen=True)
class Expect:
    """What one question's context is scored on, in one language."""

    category: str
    abstain: bool
    #: groups, each met when any item holds it: the final run's values
    contains: tuple[str, ...]
    #: every other value the same slots held in this script's history
    stale: tuple[str, ...]
    #: the final run's values naming what the question is about; ``None``: the slot is the
    #: user's own singleton (my manager, my landlord), so any item stating a stale value states
    #: it as current
    anchor: tuple[str, ...] | None
    #: related facts the store does hold (abstention questions): informational
    near: tuple[str, ...]


def expect(b: Built, query_id: str, language: str) -> Expect:
    q = next(x for x in b.script["queries"] if x["id"] == query_id)
    li = b.languages.index(language)
    current = tuple(text_of(b.final[s], li) for s in q.get("contains", ()))
    anchor = q.get("anchor")
    # the final run's own values are stale only for a slot that is neither the answer nor what
    # the question names (the replaced buyer, the first count); earlier runs' values always are
    keep = {normalize(text_of(b.final[s], li)) for s in (*q.get("contains", ()), *(anchor or ()))}
    stale: list[str] = []
    for slot in q.get("stale", ()):
        for run in b.runs:
            v = text_of(run[slot], li)
            if normalize(v) not in keep and v not in stale:
                stale.append(v)
    return Expect(
        category=q["category"],
        abstain=bool(q.get("abstain")),
        contains=current,
        stale=tuple(stale),
        anchor=tuple(text_of(b.final[s], li) for s in anchor) if anchor else None,
        near=tuple(text_of(b.final[s], li) for s in q.get("near", ())),
    )
