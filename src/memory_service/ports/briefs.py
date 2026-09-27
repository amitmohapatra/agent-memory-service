"""Durable brief definitions and bounded refresh scheduling."""

from datetime import datetime
from typing import Protocol

from memory_service.domain.briefs import BriefInfo, BriefOutput, StoredBrief


class BriefRepository(Protocol):
    async def get(self, tenant_id: str, brief_id: str) -> StoredBrief | None: ...

    async def add(self, brief: StoredBrief) -> None: ...

    async def save_output(
        self,
        tenant_id: str,
        brief_id: str,
        generation: int,
        output: BriefOutput,
        next_refresh_at: datetime,
    ) -> bool: ...

    async def replace(self, brief: StoredBrief, *, expected_generation: int) -> bool: ...

    async def delete(self, tenant_id: str, brief_id: str) -> None: ...

    async def claim_due(self, now: datetime, *, limit: int) -> list[StoredBrief]: ...

    async def list_owned(
        self,
        tenant_id: str,
        scope_key: str,
        *,
        after: str,
        limit: int,
    ) -> list[BriefInfo]: ...
