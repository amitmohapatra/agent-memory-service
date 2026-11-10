"""Persistence of beliefs, their sources and each person's consolidation progress."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from memory_service.domain.beliefs import Belief, BeliefProgress


@runtime_checkable
class BeliefRepository(Protocol):
    async def progress(
        self, tenant_id: str, scope_key: str, person_key: str, *, lock: bool = False
    ) -> BeliefProgress | None:
        """The person's progress in the scope; ``lock`` holds the row until commit."""
        ...

    async def save_progress(self, progress: BeliefProgress) -> None: ...

    async def mark_pending(
        self, tenant_id: str, scope_key: str, person_key: str, at: datetime
    ) -> datetime:
        """Note that new facts wait; returns when the oldest of them began waiting."""
        ...

    async def pending(self, before: datetime, *, limit: int) -> list[BeliefProgress]:
        """Units waiting since before ``before`` and not paused, oldest first."""
        ...

    async def of_unit(
        self, tenant_id: str, scope_key: str, person_key: str, *, current_only: bool = True
    ) -> list[Belief]:
        """The person's beliefs in the scope (current only, or with their history)."""
        ...

    async def add(self, belief: Belief) -> None: ...

    async def update(self, belief: Belief) -> None:
        """Write the belief's fields and sources; its revision moves."""
        ...

    async def delete_unit(
        self, tenant_id: str, scope_key: str, person_key: str, at: datetime
    ) -> list[Belief]:
        """Delete every belief of the unit (a full rebuild replaces them), words included."""
        ...

    async def forget_sources(
        self, tenant_id: str, memory_ids: Sequence[str], at: datetime
    ) -> list[Belief]:
        """Remove these memories as sources; a belief left without one is deleted with its
        words. Returns the beliefs that changed, as they are now."""
        ...

    async def visible(
        self,
        tenant_id: str,
        keys: Sequence[str],
        *,
        scope_keys: Sequence[str],
        people: int,
        history: int,
    ) -> list[Belief]:
        """The beliefs of these units a reader with ``keys`` may read, of the ``people``
        most-cited people: every current one, and the newest ``history`` earlier ones per
        person."""
        ...

    async def get_many(self, tenant_id: str, belief_ids: Sequence[str]) -> list[Belief]: ...

    async def mark_indexed(self, belief_ids: Sequence[str], *, fingerprint: str) -> None: ...
