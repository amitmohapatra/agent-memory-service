"""Persistence of agent-tool pulls and the prefetch counts learned from them."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from memory_service.domain.pulls import AgentPull


@runtime_checkable
class PullRepository(Protocol):
    async def add(self, pull: AgentPull) -> None: ...

    async def mark_used(self, tenant_id: str, run_id: str, item_ids: Sequence[str]) -> int:
        """Record that the run used these ids, on its pulls that returned any of them."""
        ...

    async def settled(self, *, before: datetime, limit: int) -> list[AgentPull]:
        """Pulls not folded into the counts yet, created before ``before``, oldest first."""
        ...

    async def fold(self, pulls: Sequence[AgentPull]) -> None:
        """Add each pull's returned and used ids to the counts; mark the pulls learned."""
        ...

    async def prefetch(
        self,
        tenant_id: str,
        scope_key: str,
        pattern: str,
        *,
        min_pulls: int,
        min_rate: float,
        limit: int,
    ) -> list[str]:
        """The items used often enough for this pattern, most used first."""
        ...
