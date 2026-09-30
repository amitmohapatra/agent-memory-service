"""Persistence of pinned profile blocks and durable thread summaries (indexed reads only)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from memory_service.domain.profile import ProfileBlock, ThreadSummary


@runtime_checkable
class ProfileRepository(Protocol):
    async def blocks(self, tenant_id: str, scope_keys: Sequence[str]) -> list[ProfileBlock]:
        """Every block of these scopes, by name."""
        ...

    async def get(self, tenant_id: str, scope_key: str, block: str) -> ProfileBlock | None: ...

    async def put(self, block: ProfileBlock) -> ProfileBlock:
        """Write the block's text and standing question; its version is the stored one plus
        one."""
        ...

    async def claim_due(
        self, now: datetime, *, limit: int, next_at: datetime
    ) -> list[ProfileBlock]:
        """Blocks whose standing question is due, moved to ``next_at``."""
        ...


@runtime_checkable
class ThreadSummaryRepository(Protocol):
    async def latest(self, tenant_id: str, thread_id: str) -> ThreadSummary | None: ...

    async def add(self, summary: ThreadSummary) -> bool:
        """Store a new version; False when that version already exists (a retried job)."""
        ...
