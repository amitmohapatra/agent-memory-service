"""Generated turns: repeated log lines (``Repeat``) and the large-tenant corpus for the scale
capability, with the ground truth of its aggregate questions computed from what was generated.

Both are seeded and depend only on the spec and the run's date, so two runs on the same day
write the same text. The corpus spec (``data/scale_*.json``) names its entities, its record
kinds and their templates; this module knows periods, numbers and templates, nothing else.

A corpus spec::

    {"name": ..., "years": 3, "seed": 7,
     "entities": {"<type>": [{"id": "X-1", "name": {lang: ...}}, ...]},
     "records": {"<kind>": {
         "every": "week" | "month" | "day",
         "role": "EVENT",
         "header": {lang: "... {period} ..."},
         "line": {lang: "{<type>} ({<type>.name}): {field} ..."},
         "lines_from": "<type>",                     # one line per entity of that type
         "fields": {"<field>": {"min": a, "max": b}},
         "boost": [{"period": {...}, "keys": [...], "field": f, "factor": x}]
     }},
     "sizes": {"standard": {...}, "large": {...}}}  # per-size overrides of "every" / "years"

An aggregate question::

    {"record": kind, "period": {"year_offset": -1, "quarter": 3}, "group_by": "<type>",
     "value": {"ratio": [f1, f2]} | {"sum": f} | {"count_if_positive": f},
     "where": {"<type>": "X-1"}, "top": 3}
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from benchmark.capabilities.dataset import Repeat, text_for

# --------------------------------------------------------------------------- repeats


@dataclass(frozen=True)
class GeneratedTurn:
    role: str
    text: str
    occurred_at: datetime


def _seed(*parts: object) -> int:
    import hashlib

    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _value(spec: Mapping[str, float], rng: random.Random) -> str:
    lo, hi = float(spec["min"]), float(spec["max"])
    step = float(spec.get("step", 1))
    n = round((hi - lo) / step)
    v = lo + step * rng.randint(0, n)
    return f"{v:g}" if step < 1 else str(round(v))


def expand_repeat(repeat: Repeat, lang: str, start: datetime, *, seed: str) -> list[GeneratedTurn]:
    """``repeat.count`` turns from the template, ``every_hours`` apart from ``start``."""
    rng = random.Random(_seed(seed))
    template = text_for(repeat.text, lang)
    out = []
    for i in range(repeat.count):
        when = start + timedelta(hours=repeat.every_hours * i)
        values = {k: _value(v, rng) for k, v in repeat.values.items()}
        values.update(n=str(i + 1), date=when.date().isoformat(), time=when.strftime("%H:%M"))
        out.append(GeneratedTurn(repeat.role, template.format(**values), when))
    return out


# --------------------------------------------------------------------------- periods


def _period_start(day: date, every: str) -> date:
    if every == "day":
        return day
    if every == "week":
        return day - timedelta(days=day.weekday())
    return day.replace(day=1)


def _next(day: date, every: str) -> date:
    if every == "day":
        return day + timedelta(days=1)
    if every == "week":
        return day + timedelta(days=7)
    return (day.replace(day=28) + timedelta(days=4)).replace(day=1)


def period_label(day: date, every: str) -> str:
    """The invariant label a record carries: 2025-07-14, 2025-W29, 2025-07."""
    if every == "day":
        return day.isoformat()
    if every == "week":
        y, w, _ = day.isocalendar()
        return f"{y}-W{w:02d}"
    return f"{day.year}-{day.month:02d}"


def in_period(day: date, period: Mapping[str, int], today: date) -> bool:
    """``{"year_offset": -1, "quarter": 3}``, ``{"year_offset": -1, "month": 3}``,
    ``{"year_offset": 0}``, ``{"months_ago": 1}`` - calendar periods relative to the run date."""
    if "months_ago" in period:
        y, m = today.year, today.month - int(period["months_ago"])
        while m < 1:
            y, m = y - 1, m + 12
        return day.year == y and day.month == m
    if day.year != today.year + int(period.get("year_offset", 0)):
        return False
    if "quarter" in period:
        return (day.month - 1) // 3 + 1 == int(period["quarter"])
    if "month" in period:
        return day.month == int(period["month"])
    return True


# --------------------------------------------------------------------------- the corpus


@dataclass
class Record:
    kind: str
    period: date
    label: str
    #: entity type -> id, for the record's own key (a report "for" one entity, if any)
    values: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Corpus:
    name: str
    #: session id -> its turns, oldest first; one session per calendar month
    sessions: dict[str, list[GeneratedTurn]]
    records: list[Record]
    entities: dict[str, list[dict[str, Any]]]

    @property
    def turns(self) -> int:
        return sum(len(t) for t in self.sessions.values())

    @property
    def characters(self) -> int:
        return sum(len(t.text) for ts in self.sessions.values() for t in ts)


def _entity_name(entity: Mapping[str, Any], lang: str) -> str:
    name = entity.get("name", entity["id"])
    return text_for(name, lang) if isinstance(name, Mapping) else str(name)


def build_corpus(spec: Mapping[str, Any], lang: str, today: date, size: str) -> Corpus:
    """Every record of every kind from ``years`` before ``today`` up to yesterday."""
    overrides = (spec.get("sizes") or {}).get(size, {})
    years = int(overrides.get("years", spec.get("years", 3)))
    first = today.replace(year=today.year - years)
    entities: dict[str, list[dict[str, Any]]] = {k: list(v) for k, v in spec["entities"].items()}
    sessions: dict[str, list[GeneratedTurn]] = {}
    records: list[Record] = []
    for kind, rspec in spec["records"].items():
        every = (overrides.get("every") or {}).get(kind, rspec.get("every", "week"))
        rng = random.Random(_seed(spec["name"], kind, size))
        lines_from = rspec["lines_from"]
        day = _period_start(first, every)
        while day < today:
            label = period_label(day, every)
            rec = Record(kind, day, label)
            lines = []
            for entity in entities[lines_from]:
                values: dict[str, Any] = {lines_from: entity["id"]}
                for fname, fspec in rspec["fields"].items():
                    values[fname] = int(_value(fspec, rng))
                for boost in rspec.get("boost", ()):
                    if entity["id"] in boost["keys"] and in_period(day, boost["period"], today):
                        values[boost["field"]] = int(values[boost["field"]] * boost["factor"])
                rec.values.append(values)
                fmt = {
                    **values,
                    f"{lines_from}.name": _entity_name(entity, lang),
                }
                lines.append(_fill(text_for(rspec["line"], lang), fmt))
            header = _fill(text_for(rspec["header"], lang), {"period": label})
            when = datetime.combine(_next(day, every), datetime.min.time(), tzinfo=UTC)
            when = min(when, datetime.combine(today, datetime.min.time(), tzinfo=UTC))
            when -= timedelta(hours=6)  # filed the evening before the next period opens
            session = f"{day.year}-{day.month:02d}"
            sessions.setdefault(session, []).append(
                GeneratedTurn(rspec.get("role", "EVENT"), "\n".join([header, *lines]), when)
            )
            records.append(rec)
            day = _next(day, every)
    for turns in sessions.values():
        turns.sort(key=lambda t: t.occurred_at)
    return Corpus(spec["name"], dict(sorted(sessions.items())), records, entities)


def _fill(template: str, values: Mapping[str, Any]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


# --------------------------------------------------------------------------- ground truth


@dataclass(frozen=True)
class Truth:
    #: the answer: the top keys (or the single value), best first
    answer: list[str]
    #: every key of the grouped type that is not in the answer
    others: list[str]
    #: the labels of the records the answer is computed from (the evidence)
    evidence: list[str]
    values: dict[str, float]


def aggregate(corpus: Corpus, question: Mapping[str, Any], today: date) -> Truth:
    kind = question["record"]
    period = question["period"]
    group_by: str | None = question.get("group_by")
    where: Mapping[str, str] = question.get("where", {})
    value_spec: Mapping[str, Any] = question["value"]
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    evidence: list[str] = []
    for rec in corpus.records:
        if rec.kind != kind or not in_period(rec.period, period, today):
            continue
        used = False
        for v in rec.values:
            if any(v.get(k) != want for k, want in where.items()):
                continue
            key = str(v[group_by]) if group_by else "total"
            used = True
            if "ratio" in value_spec:
                a, b = value_spec["ratio"]
                num[key] = num.get(key, 0) + v[a]
                den[key] = den.get(key, 0) + v[b]
            elif "sum" in value_spec:
                num[key] = num.get(key, 0) + v[value_spec["sum"]]
            elif "count_if_positive" in value_spec:
                num[key] = num.get(key, 0) + (1 if v[value_spec["count_if_positive"]] > 0 else 0)
            else:
                raise ValueError(f"unknown value {value_spec!r}")
        if used:
            evidence.append(rec.label)
    values = {k: (num[k] / den[k] if den.get(k) else num[k]) for k in num}
    ranked = sorted(values, key=lambda k: (-values[k], k))
    if group_by:
        top = ranked[: int(question.get("top", 1))]
        all_keys = [e["id"] for e in corpus.entities[group_by]]
        others = [k for k in all_keys if k not in top]
        return Truth(top, others, sorted(set(evidence)), values)
    total = values.get("total", 0.0)
    return Truth([f"{total:g}"], [], sorted(set(evidence)), values)


def aggregate_expect(truth: Truth, question: Mapping[str, Any]) -> dict[str, Any]:
    """Assertions for an aggregate question: the context and the search must hold every
    record the answer is computed from (each period label is a group - partial credit says
    how much of it came back); the answer must name the right keys or value, and no other
    key of the grouped type."""
    expect: dict[str, Any] = {"contains": [[label] for label in truth.evidence]}
    answer: dict[str, Any] = {"contains": [[a] for a in truth.answer]}
    if question.get("group_by"):
        answer["absent"] = [{"text": k} for k in truth.others]
    expect["answer"] = answer
    return expect


def scale_tokens(corpus: Corpus) -> int:
    """A rough token count (4 characters a token), to say how large a tenant is."""
    return corpus.characters // 4


def corpora_for(
    scenarios: Sequence[Any],
    specs: Mapping[str, Mapping[str, Any]],
    lang: str,
    today: date,
    size: str,
) -> dict[str, Corpus]:
    built: dict[str, Corpus] = {}
    for s in scenarios:
        if s.corpus and s.corpus["spec"] not in built:
            built[s.corpus["spec"]] = build_corpus(specs[s.corpus["spec"]], lang, today, size)
    return built
