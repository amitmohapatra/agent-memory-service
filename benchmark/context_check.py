"""What the context check removes and keeps, scored item by item (ADR 0048).

The check's tests script the model: they prove what is sent, removed and reported, not the
model's judgement. This measures the judgement. Each case
(``benchmark/capabilities/data/context_check.json``) is one request with what memory would
attach to it, every item labelled by hand:

* ``keep``: the request needs it - a rule that applies, a fact that is asked for, a default
  the task uses, who the person is, a rule about every answer;
* ``drop``: it does not help - about something else, pasted material listed as the person's
  rule, an instruction for a moment that has passed, steps learned for another kind of task;
* ``either``: on the topic and not needed; removing or keeping it is no error and it is not
  scored.

A case's own items come with items of the shared ``noise`` pool, as a store that holds months
of one person's work brings the same unrelated things to many requests. Each case is built
into the bundle a context would carry (the profile sections, the learned skills, the
memories, the earlier values, the graph facts), read by ``ContextCheck`` exactly as a build
reads it, and what it names is compared with the labels.

Metrics: wrongly removed (``keep`` items removed: the error that costs an answer), noise left
(``drop`` items kept), each also by the item's ``why``; the share of the listed characters
removed; the outcome of every call (checked, late, no_answer) and its latency.

    BENCH_LLM=on LLM_BASE_URL=... LLM_API_KEY=... python -m benchmark.context_check \\
        --model <the fast model, as the endpoint names it> --runs 3 --out context_check.json

``--extra 100`` adds a hundred unrelated memories to every case: the size of a context from a
store that holds months of one person's work, which is what the call's latency is made of.

Without ``BENCH_LLM=on`` nothing is spent: ``--oracle`` answers from the labels (the pipeline
and the dataset, end to end), and with neither the check is off and every item is kept.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from benchmark.common import write_result
from benchmark.retrieval import _settings
from memory_service.adapters.models.llm import EndpointLLM
from memory_service.config.constants import LLM
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    ProfileBlockView,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.tools import SkillView
from memory_service.modules.context.check import (
    FACTS,
    HISTORY,
    ITEM_CHARS,
    MEMORIES,
    SKILLS,
    USE,
    ContextCheck,
    Listed,
    listed,
)
from memory_service.modules.context.sections import LearnedBlock
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.cost import llm_accounting
from memory_service.modules.llm.policy import ModelAccess, ModelBinding, bind
from memory_service.ports.credentials import ModelIdentity

DATA: Final = Path(__file__).parent / "capabilities" / "data" / "context_check.json"
TENANT: Final = "context-check-eval"
LABELS: Final = ("keep", "drop", "either")
#: the profile sections an item may be in, in the order a context shows them
BLOCKS: Final = ("standing_rules", "preferences", "about_user", "user")
#: the sections whose lines say when they were said (``modules.context.standing``)
DATED_BLOCKS: Final = ("standing_rules", "preferences")
ITEM_KINDS: Final = {
    MEMORIES: Representation.MEMORY,
    HISTORY: Representation.MEMORY,
    FACTS: Representation.RELATION,
}
Place = tuple[str, str]


#: What a store that holds months of one person's work adds to every request: things said
#: about something else, each its own statement (``bulk``).
BULK: Final = (
    "Meeting note {n}: the stand-up of team {n} moved to room {n} on the second floor.",
    "Reminder {n}: renew the parking permit for bay {n} before the end of the month.",
    "The supplier for part {n} changed its delivery day to Thursday.",
    "Training session {n} on the new expense tool is recorded on the intranet.",
)


def bulk(extra: int) -> list[dict[str, Any]]:
    """``extra`` more memories about something else, for a context of a real store's size."""
    return [
        {
            "in": MEMORIES,
            "text": BULK[n % len(BULK)].format(n=n + 1),
            "said": -(n % 300) - 1,
            "label": "drop",
            "why": "bulk",
        }
        for n in range(extra)
    ]


def load(path: Path = DATA, *, extra: int = 0) -> list[dict[str, Any]]:
    """The cases, each with the noise items it names added after its own, and ``extra``
    more unrelated memories after those."""
    data = json.loads(path.read_text(encoding="utf-8"))
    noise = data.get("noise", {})
    return [
        {
            **case,
            "items": [
                *case["items"],
                *(noise[name] for name in case.get("noise", [])),
                *bulk(extra),
            ],
        }
        for case in data["cases"]
    ]


def _day(today: datetime, said: int | None) -> str:
    """The day an item was said: ``said`` days from today (negative: ago)."""
    return (today + timedelta(days=said or 0)).date().isoformat()


def _line(section: str, item: dict[str, Any], today: datetime) -> str:
    """The item as the profile section lists it."""
    dated = f" (said {_day(today, item['said'])})" if section in DATED_BLOCKS else ""
    return f"- {item['text']}{dated}"


def bundle_of(
    case: dict[str, Any], today: datetime
) -> tuple[ContextBundle, list[tuple[dict[str, Any], list[Place]]]]:
    """The bundle a context would carry for ``case``, and each of its items with the places
    of the bundle that state it (an item ``also`` in other sections is in all of them)."""
    lines: dict[str, list[str]] = {block: [] for block in BLOCKS}
    items: dict[str, list[ContextItem]] = {name: [] for name in ITEM_KINDS}
    skills: list[SkillView] = []
    placed: list[tuple[dict[str, Any], list[Place]]] = []
    for n, item in enumerate(case["items"]):
        places: list[Place] = []
        for section in (item["in"], *item.get("also", [])):
            if section in BLOCKS:
                line = _line(section, item, today)
                lines[section].append(line)
                places.append((section, line))
            elif section == SKILLS:
                skill_id = f"{case['id']}-s{n}"
                skills.append(
                    SkillView(
                        id=skill_id,
                        name=item["text"],
                        steps=list(item.get("steps", [])),
                        success_rate=1.0,
                        support=3,
                    )
                )
                places.append((SKILLS, skill_id))
            else:
                item_id = f"{case['id']}-{section[0]}{n}"
                observed = f"{_day(today, item.get('said'))}T09:00:00+00:00"
                attributes: dict[str, Any] = {"observed_at": observed}
                if section == HISTORY:
                    attributes["earlier_value"] = True
                items[section].append(
                    ContextItem(
                        item_id=item_id,
                        representation=ITEM_KINDS[section],
                        text=item["text"],
                        citation=f"{section}:{item_id}",
                        token_estimate=max(1, len(item["text"]) // 4),
                        attributes=attributes,
                    )
                )
                places.append((section, item_id))
        placed.append((item, places))
    profile = [
        (LearnedBlock if block == "user" else ProfileBlockView)(
            block=block, text="\n".join(lines[block]), version=1
        )
        for block in BLOCKS
        if lines[block]
    ]
    bundle = ContextBundle(
        query=case["request"],
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(rendered=case.get("conversation", "")),
        profile=profile,
        procedures=skills,
        memories=items[MEMORIES],
        history=items[HISTORY],
        graph_facts=items[FACTS],
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=8000,
        token_estimate=sum(len(i["text"]) for i in case["items"]) // 4,
    )
    return bundle, placed


def unlisted(case: dict[str, Any], today: datetime) -> list[str]:
    """The labelled items of ``case`` the check would not be asked about on their own: one
    folded into another item of a different label (a ``keep`` sentence inside a ``drop``
    text), which no answer could score right. A dataset error; empty for a sound case."""
    bundle, placed = bundle_of(case, today)
    by_place = {place: entry for entry in listed(bundle) for place in entry.places}
    wrong: list[str] = []
    for item, places in placed:
        entries = {id(by_place[p]) for p in places if p in by_place}
        if len(entries) != 1:
            wrong.append(item["text"][:60])
            continue
        entry = by_place[places[0]]
        labels = {
            other["label"]
            for other, theirs in placed
            if any(p in entry.places for p in theirs) and other["label"] != "either"
        }
        if len(labels) > 1:
            wrong.append(item["text"][:60])
    return wrong


def score(
    case: dict[str, Any],
    placed: list[tuple[dict[str, Any], list[Place]]],
    dropped: list[Listed],
    note: dict[str, Any],
) -> dict[str, Any]:
    """One call's row: each labelled item with whether it was removed."""
    gone = {place for entry in dropped for place in entry.places}
    items = [
        {
            "label": item["label"],
            "why": item.get("why", ""),
            "in": item["in"],
            "chars": len(item["text"]),
            "removed": any(place in gone for place in places),
            "text": item["text"][:80],
        }
        for item, places in placed
    ]
    return {
        "id": case["id"],
        "kind": case.get("kind", ""),
        "lang": case.get("lang", "en"),
        "outcome": note.get("outcome", "off"),
        "ms": note.get("ms"),
        "listed": note.get("listed", 0),
        "items": items,
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The rates over every call: per label, per ``why``, per kind of case, and the calls."""
    items = [(row, item) for row in rows for item in row["items"]]

    def rate(label: str, removed: bool, **where: str) -> dict[str, Any]:
        chosen = [
            item
            for row, item in items
            if item["label"] == label
            and all((row | item).get(key) == value for key, value in where.items())
        ]
        hit = sum(item["removed"] is removed for item in chosen)
        return (
            {"n": len(chosen), "count": hit, "rate": round(hit / len(chosen), 4)} if chosen else {}
        )

    whys = sorted({item["why"] for _, item in items if item["why"]})
    kinds = sorted({row["kind"] for row in rows if row["kind"]})
    scored = [item for _, item in items]
    chars = sum(item["chars"] for item in scored)
    answered = [row["ms"] for row in rows if row["outcome"] == "checked" and row["ms"] is not None]
    outcomes = sorted({row["outcome"] for row in rows})
    return {
        "calls": len(rows),
        "outcomes": {o: sum(row["outcome"] == o for row in rows) for o in outcomes},
        # the error that costs an answer: something the request needs was removed
        "wrongly_removed": rate("keep", True),
        # what the check is for: what does not help and was still sent
        "noise_left": rate("drop", False),
        "chars_removed_share": round(
            sum(item["chars"] for item in scored if item["removed"]) / chars, 4
        )
        if chars
        else 0.0,
        "by_why": {
            why: rate("keep", True, why=why) or rate("drop", False, why=why) for why in whys
        },
        "by_kind": {
            kind: {
                "wrongly_removed": rate("keep", True, kind=kind),
                "noise_left": rate("drop", False, kind=kind),
            }
            for kind in kinds
        },
        "latency_ms": {
            "p50": round(statistics.median(answered), 1),
            "p95": round(sorted(answered)[max(0, int(len(answered) * 0.95) - 1)], 1),
            "max": round(max(answered), 1),
        }
        if answered
        else {},
    }


def _labelled_drop(
    bundle: ContextBundle, placed: list[tuple[dict[str, Any], list[Place]]]
) -> set[str]:
    """What the model is shown of each listed entry that states a ``drop`` item."""
    drop = [places for item, places in placed if item["label"] == "drop"]
    return {
        entry.text[:ITEM_CHARS]
        for entry in listed(bundle)
        if any(place in entry.places for places in drop for place in places)
    }


class Oracle:
    """The labels as a model that reads them: it names every ``drop`` item and nothing else.
    What the pipeline and the dataset give with a judgement that is never wrong."""

    def __init__(self) -> None:
        self.drop: set[str] = set()

    def wants(self, use: str) -> bool:
        return use == USE

    async def structured(self, use: str, *, user: str, **_: Any) -> dict[str, Any]:
        sent = json.loads(user)["ITEMS"]
        return {"drop": [item["n"] for item in sent if item["text"] in self.drop]}


def _assist(model: str | None) -> tuple[LLMAssist, ModelBinding]:
    """The service's model client as a deployment builds it, with its shipped limits (the
    check's deadline, the fast model, the answer's size are what is measured: the judged
    runs' longer ceilings are not used here), and the policy a tenant that allows only
    ``context_check`` resolves - naming ``model`` for it when given. No endpoint (no
    ``BENCH_LLM=on``): the check is off, as in a deployment without a model."""
    cfg = _settings().llm
    assist = (
        LLMAssist(EndpointLLM(cfg, tuning=LLM), cfg, None) if cfg.enabled else LLMAssist.disabled()
    )
    access = ModelAccess(frozenset({USE}), True, False, {USE: model} if model else {})
    return assist, ModelBinding(ModelIdentity(TENANT, "user:bench"), access)


async def run(
    *, model: str | None, runs: int, only: list[str], oracle: bool, deadline: float, extra: int = 0
) -> dict[str, Any]:
    cases = [c for c in load(extra=extra) if not only or c["id"] in only or c.get("kind") in only]
    today = datetime.now(UTC)
    for case in cases:
        if wrong := unlisted(case, today):
            raise SystemExit(
                f"{case['id']}: labelled items the check cannot be asked about: {wrong}"
            )
    assist, binding = _assist(model)
    judge = Oracle() if oracle else None
    check = ContextCheck(judge or assist, deadline_seconds=deadline)  # type: ignore[arg-type]
    rows: list[dict[str, Any]] = []
    try:
        with llm_accounting() as tokens, bind(binding):
            for case in cases:
                bundle, placed = bundle_of(case, today)
                if judge is not None:
                    judge.drop = _labelled_drop(bundle, placed)
                for _ in range(runs):
                    dropped, note = await check.read(case["request"], bundle)
                    rows.append(score(case, placed, dropped, note))
                print(f"[context_check] {case['id']}", file=sys.stderr, flush=True)
    finally:
        close = getattr(assist.provider, "close", None)
        if close is not None:
            await close()
    return {
        "model": "oracle" if oracle else (model or "the service's fast model"),
        "runs": runs,
        "extra_items": extra,
        "deadline_seconds": deadline,
        "summary": summarise(rows),
        "llm_tokens": {"input": tokens.input, "output": tokens.output},
        "records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="the model context_check calls")
    parser.add_argument("--runs", type=int, default=1, help="calls per case")
    parser.add_argument("--only", default="", help="case ids or kinds, comma-separated")
    parser.add_argument("--oracle", action="store_true", help="answer from the labels")
    parser.add_argument("--deadline", type=float, default=LLM.check_deadline_seconds)
    parser.add_argument(
        "--extra", type=int, default=0, help="unrelated memories added to every case (size)"
    )
    parser.add_argument("--out", default="context_check.json")
    args = parser.parse_args()
    out = asyncio.run(
        run(
            model=args.model or None,
            runs=args.runs,
            only=[s for s in args.only.split(",") if s],
            oracle=args.oracle,
            deadline=args.deadline,
            extra=args.extra,
        )
    )
    write_result(args.out, out)
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
