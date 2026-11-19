"""Invalidation instead of deletion: the bitemporal filter both graph stores apply.

Status vocabulary for a closed relation:

- ``SUPERSEDED``  — it was true and stopped being true (``valid_to`` set); ``as_of`` inside
                    its interval still returns it
- ``INVALIDATED`` — it was never right (a re-extraction dropped it, a more reliable fact
                    contradicted it); ``as_of`` never returns it, ``valid_at`` before the
                    invalidation still does (that is what we believed then)
"""

from __future__ import annotations

from datetime import datetime

from memory_service.ports.intelligence import Relation


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
