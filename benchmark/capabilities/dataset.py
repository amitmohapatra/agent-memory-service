"""The capability dataset: loading, validation, the dev/held-out split and run selection.

One JSON file per capability under ``data/`` (the repository's golden format is JSON, see
``tests/eval/golden``). A file holds scenarios; a scenario holds sessions of time-stamped turns
(or a generated corpus, ``generate.py``) and queries; a query holds its phrasings and the
assertions its answer is scored on (``scoring.py``). Every text is a language map
``{"en": ..., "de": ..., "es": ..., "ar": ..., "hi": ...}`` (plus ``"mixed"`` for the
code-switched scenarios); identifiers, codes and numbers are the same in every language.

Nothing here knows a domain: the words live in the data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data"

#: The languages every scenario is written in. English is the full set; the others are
#: translations of every setup turn and every phrasing.
LANGUAGES: tuple[str, ...] = ("en", "de", "es", "ar", "hi")
#: Code-switched (mixed-language) variants, present on some scenarios only.
CODE_SWITCHED = "mixed"
ALL_LANGUAGES: tuple[str, ...] = (*LANGUAGES, CODE_SWITCHED)

#: How a question can be asked. Every query carries each kind once, so an accuracy can be
#: read per kind as well as per capability.
PHRASING_KINDS: tuple[str, ...] = (
    "direct",
    "indirect",
    "paraphrase",
    "typo",
    "long_winded",
    "implicit",
    "negated",
    "later",
    "terse",
    "imperative",
)
DOMAINS: tuple[str, ...] = ("retail", "personal", "it_support", "finance_hr", "field_service")
ROLES: tuple[str, ...] = ("USER", "ASSISTANT", "EVENT")
SPLITS: tuple[str, ...] = ("dev", "heldout")

Text = Mapping[str, str]
#: An assertion anchor: an invariant string, ``{"re": pattern}``, or a language map.
Anchor = str | Mapping[str, str]


def text_for(text: Text, lang: str) -> str:
    """The text in ``lang``. A code-switched run falls back to English for a text the
    scenario did not mix (setup the user wrote in English, a question asked mixed)."""
    if lang in text:
        return text[lang]
    if lang == CODE_SWITCHED and "en" in text:
        return text["en"]
    raise KeyError(f"no {lang!r} text in {sorted(text)}")


@dataclass(frozen=True)
class Turn:
    role: str
    text: Text
    #: minutes after the session's start
    minute: int = 0


@dataclass(frozen=True)
class Repeat:
    """``count`` generated turns from one template, ``every_hours`` apart, values drawn from
    seeded ranges (``generate.expand_repeat``): the dense, repetitive logging a capability like
    forgetting needs, without writing fifty sentences by hand."""

    count: int
    every_hours: float
    role: str
    text: Text
    values: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class ToolRun:
    """``count`` recorded runs of one tool sequence for ``task``, each labelled by the run's
    own outcome - how an agent's procedure is learned (``record_tool`` + ``feedback``)."""

    agent: str
    task: Text
    calls: Sequence[Mapping[str, Any]]
    count: int = 3
    verdict: str = "confirm"


@dataclass(frozen=True)
class Session:
    id: str
    #: days before the run's date (always negative: everything happened in the past)
    day: int
    #: wall-clock start, HH:MM UTC
    at: str = "09:00"
    turns: tuple[Turn, ...] = ()
    repeat: Repeat | None = None
    tool_runs: tuple[ToolRun, ...] = ()

    def start(self, today: datetime) -> datetime:
        hh, mm = (int(x) for x in self.at.split(":"))
        base = datetime.combine(today.date(), time(hh, mm), tzinfo=UTC)
        return base + timedelta(days=self.day)


@dataclass(frozen=True)
class Phrasing:
    kind: str
    text: Text


@dataclass(frozen=True)
class Query:
    id: str
    phrasings: tuple[Phrasing, ...]
    expect: Mapping[str, Any]
    #: the agent's own tool names, sent with the context request (learned skills, hints)
    tools: tuple[str, ...] = ()
    #: dev or heldout (assigned at load, ``assign_splits``)
    split: str = "dev"


@dataclass(frozen=True)
class Scenario:
    id: str
    capability: str
    domain: str
    sessions: tuple[Session, ...]
    queries: tuple[Query, ...]
    origin: str | None = None
    #: the scenario's queries and writes are made by this agent acting for the user
    agent: str | None = None
    #: a generated corpus (``generate.py``) instead of, or before, the sessions
    corpus: Mapping[str, Any] | None = None
    languages: tuple[str, ...] = LANGUAGES

    @property
    def code_switched(self) -> bool:
        return CODE_SWITCHED in self.languages


@dataclass(frozen=True)
class Capability:
    name: str
    number: int
    title: str
    definition: str
    scenarios: tuple[Scenario, ...]


@dataclass
class Dataset:
    capabilities: tuple[Capability, ...]
    sha256: str
    files: dict[str, str] = field(default_factory=dict)

    @property
    def scenarios(self) -> list[Scenario]:
        return [s for c in self.capabilities for s in c.scenarios]

    def capability(self, name: str) -> Capability:
        for c in self.capabilities:
            if c.name == name:
                return c
        raise KeyError(name)

    def counts(self) -> dict[str, Any]:
        """Scenarios, queries and phrasings per capability (English), and the totals."""
        per: dict[str, dict[str, int]] = {}
        for c in self.capabilities:
            queries = [q for s in c.scenarios for q in s.queries]
            per[c.name] = {
                "scenarios": len(c.scenarios),
                "queries": len(queries),
                "phrasings": sum(len(q.phrasings) for q in queries),
                "dev_queries": sum(1 for q in queries if q.split == "dev"),
                "heldout_queries": sum(1 for q in queries if q.split == "heldout"),
                "code_switched_scenarios": sum(1 for s in c.scenarios if s.code_switched),
                "retail_scenarios": sum(1 for s in c.scenarios if s.domain == "retail"),
            }
        return {
            "per_capability": per,
            "scenarios": sum(p["scenarios"] for p in per.values()),
            "queries": sum(p["queries"] for p in per.values()),
            "phrasings_per_language": sum(p["phrasings"] for p in per.values()),
            "languages": list(LANGUAGES),
            "code_switched_scenarios": sum(p["code_switched_scenarios"] for p in per.values()),
            "domains": list(DOMAINS),
        }


# --------------------------------------------------------------------------- loading


def _session(raw: Mapping[str, Any]) -> Session:
    turns: list[Turn] = []
    for t in raw.get("turns", ()):
        # a turn without its own minute follows the previous one by a minute
        minute = int(t["minute"]) if "minute" in t else (turns[-1].minute + 1 if turns else 0)
        turns.append(Turn(role=t.get("role", "USER"), text=t["text"], minute=minute))
    repeat = None
    if r := raw.get("repeat"):
        repeat = Repeat(
            count=int(r["count"]),
            every_hours=float(r.get("every_hours", 24)),
            role=r.get("role", "USER"),
            text=r["text"],
            values=r.get("values", {}),
        )
    runs = tuple(
        ToolRun(
            agent=t["agent"],
            task=t["task"],
            calls=tuple(t["calls"]),
            count=int(t.get("count", 3)),
            verdict=t.get("verdict", "confirm"),
        )
        for t in raw.get("tool_runs", ())
    )
    return Session(
        id=raw["id"],
        day=int(raw["day"]),
        at=raw.get("at", "09:00"),
        turns=tuple(turns),
        repeat=repeat,
        tool_runs=runs,
    )


def _languages(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """The languages a scenario can run in: the core five, plus ``mixed`` when any of its
    phrasings is code-switched."""
    mixed = any(
        CODE_SWITCHED in p["text"] for q in raw.get("queries", ()) for p in q.get("phrasings", ())
    )
    return (*LANGUAGES, CODE_SWITCHED) if mixed else LANGUAGES


def _scenario(raw: Mapping[str, Any], capability: str) -> Scenario:
    return Scenario(
        id=raw["id"],
        capability=capability,
        domain=raw["domain"],
        origin=raw.get("origin"),
        agent=raw.get("agent"),
        corpus=raw.get("corpus"),
        sessions=tuple(_session(s) for s in raw.get("sessions", ())),
        queries=tuple(
            Query(
                id=q["id"],
                phrasings=tuple(Phrasing(p["kind"], p["text"]) for p in q["phrasings"]),
                expect=q.get("expect", {}),
                tools=tuple(q.get("tools", ())),
            )
            for q in raw["queries"]
        ),
        languages=_languages(raw),
    )


def _digest(paths: Sequence[Path]) -> tuple[str, dict[str, str]]:
    """One hash over every data file, canonicalised: a gate's thresholds hold for exactly the
    dataset they were measured on, and a re-indented file is the same dataset."""
    h = hashlib.sha256()
    files: dict[str, str] = {}
    for path in sorted(paths):
        canonical = json.dumps(
            json.loads(path.read_text(encoding="utf-8")), sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        files[path.name] = hashlib.sha256(canonical).hexdigest()
        h.update(path.name.encode())
        h.update(canonical)
    return h.hexdigest(), files


def capability_files(root: Path = DATA) -> list[Path]:
    return sorted(p for p in root.glob("[0-9][0-9]_*.json"))


def load_dataset(root: Path = DATA) -> Dataset:
    paths = capability_files(root)
    capabilities: list[Capability] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        name = raw["capability"]
        capabilities.append(
            Capability(
                name=name,
                number=int(raw["number"]),
                title=raw["title"],
                definition=raw["definition"],
                scenarios=tuple(_scenario(s, name) for s in raw["scenarios"]),
            )
        )
    # the questions, their assertions and the scorer's language data; not _modes.json, which
    # says who runs what (a run records its own selection), not what is asked
    extra = [*sorted(root.glob("scale_*.json")), root / "_languages.json"]
    sha, files = _digest([*paths, *extra])
    dataset = Dataset(capabilities=tuple(capabilities), sha256=sha, files=files)
    assign_splits(dataset)
    return dataset


def load_json(name: str, root: Path = DATA) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the split


def _rank(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def assign_splits(dataset: Dataset) -> None:
    """Deterministic, stratified dev/held-out halves. Within each capability the queries are
    ordered by the hash of their id and alternate dev, held-out, dev, ... - so each half has
    every capability, the assignment never depends on file order, and adding a query moves
    no other query between halves more than its neighbours. The held-out half is never used
    to tune anything (docs/eval/capabilities.md)."""
    for cap in dataset.capabilities:
        keyed = sorted(
            ((_rank(f"{s.id}/{q.id}"), s, q) for s in cap.scenarios for q in s.queries),
            key=lambda x: x[0],
        )
        assigned: dict[tuple[str, str], str] = {
            (s.id, q.id): SPLITS[i % 2] for i, (_, s, q) in enumerate(keyed)
        }
        new_scenarios = []
        for s in cap.scenarios:
            queries = tuple(
                Query(q.id, q.phrasings, q.expect, q.tools, assigned[(s.id, q.id)])
                for q in s.queries
            )
            new_scenarios.append(_replace_queries(s, queries))
        object.__setattr__(cap, "scenarios", tuple(new_scenarios))


def _replace_queries(s: Scenario, queries: tuple[Query, ...]) -> Scenario:
    return Scenario(
        id=s.id,
        capability=s.capability,
        domain=s.domain,
        sessions=s.sessions,
        queries=queries,
        origin=s.origin,
        agent=s.agent,
        corpus=s.corpus,
        languages=s.languages,
    )


# --------------------------------------------------------------------------- selection


@dataclass(frozen=True)
class Selection:
    """What one run measures. Every field narrows deterministically; ``None`` is all."""

    languages: tuple[str, ...] = ALL_LANGUAGES
    split: str | None = None
    capabilities: tuple[str, ...] | None = None
    domains: tuple[str, ...] | None = None
    kinds: tuple[str, ...] | None = None
    #: per capability, keep this many scenarios: the documented origin ones first, then the
    #: rest by the hash of their id (a stable sample that does not depend on file order)
    scenarios_per_capability: int | None = None
    #: per (capability, domain), at most this many scenarios (the same order)
    scenarios_per_domain: int | None = None
    scenario_ids: tuple[str, ...] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "languages": list(self.languages),
            "split": self.split or "all",
            "capabilities": list(self.capabilities) if self.capabilities else "all",
            "domains": list(self.domains) if self.domains else "all",
            "kinds": list(self.kinds) if self.kinds else "all",
            "scenarios_per_capability": self.scenarios_per_capability,
            "scenarios_per_domain": self.scenarios_per_domain,
            "scenario_ids": list(self.scenario_ids) if self.scenario_ids else None,
        }


def _sample_order(s: Scenario) -> tuple[int, str]:
    return (0 if s.origin else 1, _rank(s.id))


def select(dataset: Dataset, sel: Selection) -> list[Scenario]:
    """The scenarios a run writes (each with only the queries and phrasings it asks)."""
    out: list[Scenario] = []
    for cap in dataset.capabilities:
        if sel.capabilities and cap.name not in sel.capabilities:
            continue
        pool = [s for s in cap.scenarios if not sel.domains or s.domain in sel.domains]
        if sel.scenario_ids:
            pool = [s for s in pool if s.id in sel.scenario_ids]
        pool.sort(key=_sample_order)
        if sel.scenarios_per_domain is not None:
            seen: dict[str, int] = {}
            kept = []
            for s in pool:
                if seen.get(s.domain, 0) < sel.scenarios_per_domain:
                    kept.append(s)
                    seen[s.domain] = seen.get(s.domain, 0) + 1
            pool = kept
        if sel.scenarios_per_capability is not None:
            pool = pool[: sel.scenarios_per_capability]
        for s in pool:
            queries = []
            for q in s.queries:
                if sel.split and q.split != sel.split:
                    continue
                phrasings = tuple(p for p in q.phrasings if not sel.kinds or p.kind in sel.kinds)
                if phrasings:
                    queries.append(Query(q.id, phrasings, q.expect, q.tools, q.split))
            if queries:
                out.append(_replace_queries(s, tuple(queries)))
    return out


def languages_of(s: Scenario, sel: Selection) -> list[str]:
    return [lang for lang in sel.languages if lang in s.languages]


@dataclass(frozen=True)
class Case:
    """One question to ask: a phrasing of a query of a scenario, in one language."""

    scenario: Scenario
    query: Query
    phrasing: Phrasing
    language: str

    @property
    def key(self) -> str:
        return f"{self.scenario.id}/{self.query.id}/{self.phrasing.kind}/{self.language}"

    @property
    def text(self) -> str:
        return text_for(self.phrasing.text, self.language)


def cases(scenario: Scenario, language: str) -> Iterator[Case]:
    for q in scenario.queries:
        for p in q.phrasings:
            if language == CODE_SWITCHED and CODE_SWITCHED not in p.text:
                continue  # a code-switched run asks only the phrasings written mixed
            yield Case(scenario, q, p, language)
