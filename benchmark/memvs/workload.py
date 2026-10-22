"""The datasets as neutral workloads (``model.Workload``), built with this repository's loaders.

    PYTHONPATH=src:. python -m benchmark.memvs.workload --out build/memvs/workloads

* ``lme18``: LongMemEval-S (cleaned), the 18-question seed-7 sample the regression gate uses
  (``longmemeval_retrieval.sample``); every question is a unit with its own haystack.
* ``lme_heldout``: the same stratified sampler with seed 11 over the questions the 18 do not
  contain - questions no change in this repository was tuned on.
* ``locomo``: all ten LoCoMo conversations; turns are the dataset's ``dia_id``s, image
  captions appended the way ``benchmark.locomo`` ingests them; the adversarial category has no
  evidence and is left out of recall.
* ``locomo_c0``: conversation 0 alone, the tier the LLM-extraction systems can afford on a CPU.
* ``capability``: the capability gate's core set, every language and phrasing, written as
  ``benchmark.capabilities.runner.write_scenario`` writes it (same user identities, the same
  run date anchoring every session); tool runs, which only this service records natively, are
  given to every other system as one event turn each.
* ``isolation``: two tenants with two users each; every user tells the memory secrets of
  their own and asks for everyone's.

Answer-accuracy samples (``Query.answer_sample``) are fixed here, so every system answers the
same questions: all of ``lme18``; 30 questions of ``locomo_c0`` stratified by category (seed 7);
the capability eval's own ``harness`` sample (English, the direct phrasing).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

from benchmark.capabilities.dataset import (
    Selection,
    cases,
    languages_of,
    load_dataset,
    load_json,
    select,
    text_for,
)
from benchmark.capabilities.generate import expand_repeat
from benchmark.capabilities.runner import identity
from benchmark.locomo import (
    ADVERSARIAL,
    CATEGORY_NAMES,
    _evidence_ids,
    _session_time,
    _sessions,
)
from benchmark.longmemeval_retrieval import gold_of, sample, session_time
from benchmark.memvs.model import (
    DatasetName,
    Query,
    Role,
    Session,
    Turn,
    Unit,
    Workload,
    turn_session,
)

LME_DATA = Path("benchmark/data/longmemeval/longmemeval_s_cleaned.json")
LOCOMO_DATA = Path("benchmark/data/locomo10.json")
#: the regression gate's sample
LME_GATE_QUESTIONS = 18
LME_GATE_SEED = 7
#: the held-out sample: disjoint from the gate's, drawn with another seed
LME_HELDOUT_QUESTIONS = 30
LME_HELDOUT_SEED = 11
LOCOMO_ANSWER_SAMPLE = 30
LOCOMO_ANSWER_SEED = 7
#: the capability gate's PR set and the sample its harness mode answers
CAPABILITY_SET = "core"
CAPABILITY_ANSWER_SAMPLE = "harness"
#: the stress run's anchor date: sessions are days before it
CAPABILITY_RUN_DATE = datetime(2026, 10, 18, tzinfo=UTC)
ISOLATION_TENANTS = ("tenant-a", "tenant-b")
ISOLATION_USERS = ("user-1", "user-2")
#: generic secrets: a code per (tenant, user, slot); each user asks for every user's
ISOLATION_SLOTS = ("locker code", "parking spot number", "badge PIN", "safe combination")


def _iso(when: datetime | None) -> str | None:
    return when.isoformat() if when else None


# --------------------------------------------------------------------------- LongMemEval


def _lme_unit(entry: dict[str, Any], *, answer_sample: bool) -> Unit:
    sessions: list[Session] = []
    for sid, date, raw in zip(
        entry["haystack_session_ids"],
        entry["haystack_dates"],
        entry["haystack_sessions"],
        strict=True,
    ):
        turns = tuple(
            Turn(f"{sid}:{k}", Role(str(t.get("role") or "user")), str(t["content"]).strip())
            for k, t in enumerate(raw)
            if str(t.get("content") or "").strip()
        )
        if turns:
            sessions.append(Session(sid, _iso(session_time(date)), turns))
    gold = gold_of(entry)
    qid = str(entry["question_id"])
    query = Query(
        query_id=qid,
        text=str(entry["question"]),
        at=_iso(session_time(str(entry.get("question_date") or ""))),
        answer=str(entry["answer"]),
        category=str(entry["question_type"]),
        evidence_turns=tuple(gold),
        evidence_sessions=tuple(sorted({turn_session(g) for g in gold})),
        answer_sample=answer_sample,
    )
    return Unit(f"lme-{qid}", "memvs-lme", f"lme-{qid}", tuple(sessions), (query,))


def longmemeval(entries: list[dict[str, Any]]) -> tuple[Workload, Workload]:
    gate = sample(entries, LME_GATE_QUESTIONS, seed=LME_GATE_SEED)
    taken = {e["question_id"] for e in gate}
    rest = [e for e in entries if e["question_id"] not in taken]
    heldout = sample(rest, LME_HELDOUT_QUESTIONS, seed=LME_HELDOUT_SEED)
    meta = {"source": str(LME_DATA)}
    return (
        Workload(
            DatasetName.LME18,
            tuple(_lme_unit(e, answer_sample=True) for e in gate),
            {**meta, "seed": LME_GATE_SEED},
        ),
        Workload(
            DatasetName.LME_HELDOUT,
            tuple(_lme_unit(e, answer_sample=False) for e in heldout),
            {**meta, "seed": LME_HELDOUT_SEED},
        ),
    )


# --------------------------------------------------------------------------- LoCoMo


def _locomo_turn(turn: dict[str, Any]) -> Turn | None:
    body = str(turn.get("text") or turn.get("clean_text") or "").strip()
    if caption := str(turn.get("blip_caption") or "").strip():
        query = str(turn.get("query") or "").strip()
        body = f"{body} [Shared an image{f' of {query}' if query else ''}: {caption}]".strip()
    dia_id, speaker = turn.get("dia_id"), str(turn.get("speaker") or "")
    if not dia_id or not body:
        return None
    # both speakers are people (no assistant): each turn is a user's, named by its speaker
    return Turn(str(dia_id), Role.USER, body, speaker)


def _locomo_unit(number: int, conv: dict[str, Any], sampled: set[int]) -> Unit:
    c = conv["conversation"]
    sessions = []
    last: datetime | None = None
    for key, raw in _sessions(c):
        when = _session_time(str(c.get(f"{key}_date_time", "")))
        last = when or last
        turns = tuple(t for t in (_locomo_turn(x) for x in raw) if t)
        if turns:
            # the dia_id's own prefix ("D1") is the session the evidence names
            sessions.append(Session(turn_session(turns[0].turn_id), _iso(when), turns))
    queries = []
    for n, qa in enumerate(conv["qa"]):
        category = int(qa["category"])
        evidence = tuple(_evidence_ids(qa)) if category != ADVERSARIAL else ()
        queries.append(
            Query(
                query_id=f"c{number}-q{n}",
                text=str(qa["question"]),
                at=_iso(last + timedelta(days=1) if last else None),
                answer=None if category == ADVERSARIAL else str(qa.get("answer", "")),
                category=CATEGORY_NAMES[category],
                evidence_turns=evidence,
                evidence_sessions=tuple(sorted({turn_session(e) for e in evidence})),
                answer_sample=n in sampled,
            )
        )
    return Unit(
        f"locomo-{number}", "memvs-locomo", f"locomo-{number}", tuple(sessions), tuple(queries)
    )


def _locomo_answer_sample(conv: dict[str, Any]) -> set[int]:
    """``LOCOMO_ANSWER_SAMPLE`` answerable questions, a proportional share per category."""
    by: dict[int, list[int]] = defaultdict(list)
    for n, qa in enumerate(conv["qa"]):
        if int(qa["category"]) != ADVERSARIAL and _evidence_ids(qa):
            by[int(qa["category"])].append(n)
    total = sum(len(v) for v in by.values())
    rng = random.Random(LOCOMO_ANSWER_SEED)
    out: set[int] = set()
    for category in sorted(by):
        share = max(1, round(LOCOMO_ANSWER_SAMPLE * len(by[category]) / total))
        out.update(rng.sample(by[category], min(share, len(by[category]))))
    return out


def locomo(conversations: list[dict[str, Any]]) -> tuple[Workload, Workload]:
    sampled = _locomo_answer_sample(conversations[0])
    units = tuple(
        _locomo_unit(n, c, sampled if n == 0 else set()) for n, c in enumerate(conversations)
    )
    meta = {"source": str(LOCOMO_DATA), "answer_seed": LOCOMO_ANSWER_SEED}
    return Workload(DatasetName.LOCOMO, units, meta), Workload(
        DatasetName.LOCOMO_C0, units[:1], meta
    )


# --------------------------------------------------------------------------- capability


def _tool_run_text(run: Any, lang: str) -> str:
    """A recorded tool run as the one event turn a system without tool records can keep."""
    steps = "; ".join(
        f"{i + 1}. {c['tool']}({json.dumps(c.get('args', {}), ensure_ascii=False)})"
        + (f" -> {c['output']}" if c.get("output") is not None else "")
        for i, c in enumerate(run.calls)
    )
    return (
        f"Agent {run.agent} completed the task '{text_for(run.task, lang)}' {run.count} times "
        f"(outcome: {run.verdict}) with these steps: {steps}"
    )


_ROLE = {"USER": Role.USER, "ASSISTANT": Role.ASSISTANT, "EVENT": Role.USER}


def _capability_sessions(scenario: Any, lang: str) -> tuple[Session, ...]:
    out = []
    for s in sorted(scenario.sessions, key=lambda x: x.day):
        start = s.start(CAPABILITY_RUN_DATE)
        raw = [
            (t.role, text_for(t.text, lang), start + timedelta(minutes=t.minute)) for t in s.turns
        ]
        if s.repeat:
            raw += [
                (g.role, g.text, g.occurred_at)
                for g in expand_repeat(s.repeat, lang, start, seed=f"{scenario.id}/{s.id}")
            ]
        raw.sort(key=lambda x: x[2])
        turns = [
            Turn(f"{s.id}:{k}", _ROLE[role], text, "event" if role == "EVENT" else None)
            for k, (role, text, _) in enumerate(raw)
        ]
        turns += [
            Turn(f"{s.id}:t{k}", Role.USER, _tool_run_text(run, lang), "event")
            for k, run in enumerate(s.tool_runs)
        ]
        if turns:
            out.append(Session(s.id, _iso(start), tuple(turns)))
    return tuple(out)


def _selection(name: str) -> Selection:
    spec = load_json("_modes.json")["samples"][name]
    return Selection(
        languages=tuple(spec.get("languages", Selection().languages)),
        capabilities=tuple(spec["capabilities"]) if "capabilities" in spec else None,
        kinds=tuple(spec["kinds"]) if "kinds" in spec else None,
        scenarios_per_domain=spec.get("scenarios_per_domain"),
        scenario_ids=tuple(spec["scenarios"]) if "scenarios" in spec else None,
    )


def capability() -> Workload:
    dataset = load_dataset()
    full = _selection(CAPABILITY_SET)
    answer_keys = {
        case.key
        for s in select(dataset, _selection(CAPABILITY_ANSWER_SAMPLE))
        for lang in languages_of(s, _selection(CAPABILITY_ANSWER_SAMPLE))
        for case in cases(s, lang)
        if case.phrasing.kind in (_selection(CAPABILITY_ANSWER_SAMPLE).kinds or ())
    }
    units = []
    asked_at = _iso(datetime.combine(CAPABILITY_RUN_DATE.date(), time(12), tzinfo=UTC))
    for s in select(dataset, full):
        for lang in languages_of(s, full):
            user = identity(s.id, lang)
            queries = tuple(
                Query(
                    query_id=case.key,
                    text=case.text,
                    at=asked_at,
                    category=s.capability,
                    language=lang,
                    answer_sample=case.key in answer_keys,
                )
                for case in cases(s, lang)
            )
            units.append(
                Unit(
                    f"{s.id}/{lang}",
                    "memvs-capability",
                    user,
                    _capability_sessions(s, lang),
                    queries,
                )
            )
    return Workload(
        DatasetName.CAPABILITY,
        tuple(units),
        {"set": CAPABILITY_SET, "run_date": CAPABILITY_RUN_DATE.date().isoformat()},
    )


# --------------------------------------------------------------------------- isolation


def _secret(tenant: str, user: str, slot: str) -> str:
    digits = random.Random(f"{tenant}/{user}/{slot}").randrange(10_000, 99_999)
    return str(digits)


def isolation() -> Workload:
    """Each user's own secrets; every user then asks for every slot. A hit that carries
    another user's code is a leak (scored in ``score``)."""
    units = []
    for tenant in ISOLATION_TENANTS:
        for user in ISOLATION_USERS:
            turns = tuple(
                Turn(
                    f"s1:{k}",
                    Role.USER,
                    f"Please remember: my {slot} is {_secret(tenant, user, slot)}.",
                )
                for k, slot in enumerate(ISOLATION_SLOTS)
            )
            queries = tuple(
                Query(
                    query_id=f"{tenant}/{user}/{slot}",
                    text=f"What is my {slot}?",
                    answer=_secret(tenant, user, slot),
                    category="isolation",
                )
                for slot in ISOLATION_SLOTS
            )
            units.append(
                Unit(f"{tenant}/{user}", tenant, user, (Session("s1", None, turns),), queries)
            )
    secrets = {
        f"{t}/{u}": [_secret(t, u, slot) for slot in ISOLATION_SLOTS]
        for t in ISOLATION_TENANTS
        for u in ISOLATION_USERS
    }
    return Workload(DatasetName.ISOLATION, tuple(units), {"secrets": secrets})


# --------------------------------------------------------------------------- main


def build(names: Iterable[DatasetName]) -> list[Workload]:
    wanted = set(names)
    out: list[Workload] = []
    if wanted & {DatasetName.LME18, DatasetName.LME_HELDOUT}:
        out += [w for w in longmemeval(json.loads(LME_DATA.read_text())) if w.dataset in wanted]
    if wanted & {DatasetName.LOCOMO, DatasetName.LOCOMO_C0}:
        out += [w for w in locomo(json.loads(LOCOMO_DATA.read_text())) if w.dataset in wanted]
    if DatasetName.CAPABILITY in wanted:
        out.append(capability())
    if DatasetName.ISOLATION in wanted:
        out.append(isolation())
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--datasets", default=",".join(d.value for d in DatasetName), help="comma-separated"
    )
    args = parser.parse_args(argv)
    for workload in build(DatasetName(d) for d in args.datasets.split(",")):
        path = args.out / f"{workload.dataset.value}.json"
        workload.dump(path)
        turns = sum(len(s.turns) for u in workload.units for s in u.sessions)
        queries = sum(len(u.queries) for u in workload.units)
        print(
            f"{workload.dataset.value}: {len(workload.units)} units, {turns} turns, {queries} queries -> {path}"
        )


if __name__ == "__main__":
    main()
