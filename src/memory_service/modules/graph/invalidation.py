"""Invalidation instead of deletion: the edge that records why a fact gave way to another,
the bitemporal filter both stores apply, and the deterministic adjudication of two facts
that occupy the same slot.

Status vocabulary for a closed relation:

- ``SUPERSEDED``  — it was true and stopped being true (``valid_to`` set); ``as_of`` inside
                    its interval still returns it
- ``INVALIDATED`` — it was never right (a re-extraction dropped it, a more reliable fact
                    contradicted it); ``as_of`` never returns it, ``valid_at`` before the
                    invalidation still does (that is what we believed then)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from memory_service.domain.graph import INVALIDATED_BY
from memory_service.domain.ids import stable_key
from memory_service.ports.intelligence import Relation

REASON_MEMORY_SUPERSEDED = "memory_superseded"
REASON_CONTRADICTION = "contradiction"
REASON_DOCUMENT_REINDEXED = "document_reindexed"


def invalidation_edge(
    loser: Relation,
    winner: Relation,
    *,
    reason: str,
    at: datetime,
    attributes: dict[str, Any] | None = None,
) -> Relation:
    """The relation -> relation edge that records why ``loser`` gave way to ``winner``."""
    return Relation(
        relation_id="inv_" + stable_key(loser.tenant_id, loser.relation_id, winner.relation_id),
        tenant_id=loser.tenant_id,
        subject_id=loser.relation_id,
        predicate=INVALIDATED_BY,
        object_id=winner.relation_id,
        layer="temporal",
        scope_key=loser.scope_key,
        visibility_keys=list(loser.visibility_keys),
        observed_at=at,
        status="CURRENT",
        confidence=1.0,
        evidence=list(winner.evidence[:3] or loser.evidence[:3]),
        fact_text=(
            f"'{loser.fact_text[:120]}' was invalidated by '{winner.fact_text[:120]}' ({reason})"
        ),
        attributes={
            "reason": reason,
            "invalidated_at": at.isoformat(),
            "winner": winner.relation_id,
            "loser": loser.relation_id,
            "loser_status": loser.status,
            **(attributes or {}),
        },
    )


def passes_time(r: Relation, *, as_of: datetime | None, valid_at: datetime | None) -> bool:
    """``as_of``: true at that instant (valid time). ``valid_at``: asserted by then and not
    yet invalidated (knowledge time). Neither: CURRENT only."""
    if as_of is None and valid_at is None:
        return r.status == "CURRENT"
    if as_of is not None:
        if r.status in ("RETRACTED", "INVALIDATED"):
            return False
        if (r.valid_from and r.valid_from > as_of) or (r.valid_to and r.valid_to <= as_of):
            return False
    if valid_at is not None:
        if r.observed_at > valid_at:
            return False
        if r.invalidated_at is not None and r.invalidated_at <= valid_at:
            return False
    return True


def adjudicate(a: Relation, b: Relation) -> tuple[Relation, Relation, str, str]:
    """Two CURRENT facts in the same slot (subject, predicate) with different objects:
    the more confident one wins; on a tie the more recently observed one. Returns
    ``(winner, loser, why, loser_status)`` where the loser is SUPERSEDED when it is the
    older statement (it was true, then replaced) and INVALIDATED when it is the newer one
    (a less reliable claim rejected against an established fact)."""
    if abs(a.confidence - b.confidence) > 1e-9:
        winner, loser = (a, b) if a.confidence > b.confidence else (b, a)
        why = f"higher confidence ({winner.confidence:.2f} vs {loser.confidence:.2f})"
    else:
        winner, loser = (a, b) if a.observed_at >= b.observed_at else (b, a)
        why = "more recently observed at equal confidence"
    loser_status = "SUPERSEDED" if loser.observed_at <= winner.observed_at else "INVALIDATED"
    return winner, loser, why, loser_status
