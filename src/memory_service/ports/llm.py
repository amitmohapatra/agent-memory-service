"""Per-tenant generative-model policy and usage (ADR 0023).

A policy row narrows what the model may be used for at one level of the key hierarchy -
the principal, its workspace, its tenant - and is resolved the way keys are: the most
specific row that exists wins. Usage is one row per tenant, use and day, incremented by
every successful gateway call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from memory_service.ports.credentials import ModelIdentity


@dataclass(frozen=True)
class StoredPolicy:
    identity: ModelIdentity
    uses: frozenset[str]
    read_assist: bool
    revision: int
    updated_at: datetime


@dataclass(frozen=True)
class UsageDay:
    day: date
    use: str
    tokens: int
    calls: int


class LLMPolicyRepository(Protocol):
    async def first(self, levels: Sequence[ModelIdentity]) -> StoredPolicy | None:
        """The row of the first level (most specific first) that has one: one indexed read."""
        ...

    async def put(
        self, identity: ModelIdentity, *, uses: Sequence[str], read_assist: bool
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
