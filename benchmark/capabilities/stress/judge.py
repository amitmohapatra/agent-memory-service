"""Score one context bundle against one accumulated-store question.

The surface is the pushed context (``/v1/context``, ``format="full"``), read as the capability
eval reads it (``scoring.items_from_bundle``: profile, thread summary, window, memories,
knowledge, graph facts, summaries - in the order an agent's model reads them). An item the
service marks as no longer current (superseded, a fact with an end) is historical.

For an answerable question:

* ``missing``       a group of the final run's values found in no item;
* ``false_refusal`` the service said ``INSUFFICIENT``: an agent told so refuses, so this
                    fails even when the values are there;
* ``stale_leak``    an earlier value is served wrongly, either
  - ``stale_first``: the first current (non-historical) item holding any value of the
    question holds a stale one and no current one - the stale value comes first; or
  - ``stale_current``: a current item holds a stale value and no current value, and names what
    the question is about (its anchor: the store, the region) - or the question is about the
    user's own singleton (my manager, my landlord: no anchor), where any such item states the
    stale value as current.
  An item holding both ("Delia Tanaka has replaced Liora Moreau") states the current value and
  is not a leak.

For a never-stated question: pass when the service says ``INSUFFICIENT``; ``near_served`` (the
related facts the store does hold, in a current item) is recorded, not scored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from benchmark.capabilities.scoring import Item, Pattern, normalize
from benchmark.capabilities.stress.build import Expect

#: failure reasons, in precedence order
REASONS = ("error", "no_abstain", "false_refusal", "not_retrieved", "missing", "stale_leak")


def _pats(values: Sequence[str]) -> list[Pattern]:
    return [Pattern(v, literal=normalize(v)) for v in values]


def _has(pats: Sequence[Pattern], norm: str) -> bool:
    return any(p.matches(norm) for p in pats)


@dataclass
class Judged:
    passed: bool
    reason: str | None
    abstained: bool
    missing: list[str] = field(default_factory=list)
    stale_first: bool = False
    stale_current: bool = False
    stale_seen: bool = False
    stale_values: list[str] = field(default_factory=list)
    #: rank of the first current item holding the answer / a stale value (None: none)
    first_current_rank: int | None = None
    first_stale_rank: int | None = None
    near_served: bool = False
    items: int = 0

    @property
    def stale_leak(self) -> bool:
        return self.stale_first or self.stale_current

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "stale_leak": self.stale_leak}


def judge(e: Expect, items: Sequence[Item], evidence_status: str | None) -> Judged:
    abstained = evidence_status == "INSUFFICIENT"
    norms = [(i, i.norm) for i in items]
    current_items = [(i, n) for i, n in norms if not i.historical]
    j = Judged(passed=False, reason=None, abstained=abstained, items=len(items))
    if e.abstain:
        near = _pats(e.near)
        j.near_served = any(_has(near, n) for _, n in current_items)
        j.reason = None if abstained else "no_abstain"
        j.passed = j.reason is None
        return j

    for value in e.contains:
        p = Pattern(value, literal=normalize(value))
        if not any(p.matches(n) for _, n in norms):
            j.missing.append(value)
    cur = _pats(e.contains)
    stale = _pats(e.stale)
    anchor = _pats(e.anchor) if e.anchor else None
    seen: list[str] = []
    for rank, (item, n) in enumerate(norms):
        if item.historical:
            continue
        has_cur = _has(cur, n)
        hits = [p.label for p in stale if p.matches(n)]
        if hits:
            j.stale_seen = True
            seen += [h for h in hits if h not in seen]
        if has_cur and j.first_current_rank is None:
            j.first_current_rank = rank
        if hits and not has_cur:
            if j.first_stale_rank is None:
                j.first_stale_rank = rank
            if anchor is None or _has(anchor, n):
                j.stale_current = True
    j.stale_values = seen[:5]
    j.stale_first = j.first_stale_rank is not None and (
        j.first_current_rank is None or j.first_stale_rank < j.first_current_rank
    )
    if abstained:
        j.reason = "false_refusal"
    elif j.missing and not items:
        j.reason = "not_retrieved"
    elif j.missing:
        j.reason = "missing"
    elif j.stale_leak:
        j.reason = "stale_leak"
    j.passed = j.reason is None
    return j
