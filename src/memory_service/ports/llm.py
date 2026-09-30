"""Per-tenant generative-model policy and usage.

A tenant's policy says what the model may be used for, whether reads are assisted and which
model each use calls. Usage is one row per tenant, use and day, incremented by every
successful gateway call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True)
class StoredPolicy:
    tenant_id: str
    uses: frozenset[str]
    read_assist: bool
    models: Mapping[str, str]
    revision: int
    updated_at: datetime


@dataclass(frozen=True)
class UsageDay:
    day: date
    use: str
    tokens: int
    calls: int


class LLMPolicyRepository(Protocol):
    async def get(self, tenant_id: str) -> StoredPolicy | None: ...

    async def put(
        self,
        tenant_id: str,
        *,
        uses: Sequence[str],
        read_assist: bool,
        models: Mapping[str, str],
    ) -> StoredPolicy: ...


class LLMUsageRepository(Protocol):
    async def add(self, tenant_id: str, use: str, day: date, tokens: int) -> None:
        """Count one call and its tokens: an upsert on (tenant, use, day)."""
        ...

    async def between(self, tenant_id: str, since: date, until: date) -> list[UsageDay]:
        """The tenant's rows for ``since <= day <= until``, oldest first."""
        ...


class LLMUsageRecorder(Protocol):
    """What the gateway adapter reports each successful call to."""

    async def record(self, tenant_id: str, use: str, tokens: int) -> None: ...
