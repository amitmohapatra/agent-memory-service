"""The read audit: who retrieved which records, when, under which scope.

Regulated buyers ask this question after the fact, and it cannot be answered over data that
was served without recording it. One entry per recall or context build, written off the
request path in batches; the loss window is the flush interval, and it is stated as such.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReadKind = Literal["recall", "context"]


class ReadAuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    credential: str = Field(
        description="the authenticated caller (key:<id>, platform, dev:<hash> or the JWT "
        "subject); ``principal`` below is who it acted for"
    )
    principal: str
    kind: ReadKind
    query_hash: str = Field(
        description="SHA-256 of the query text; the text itself is never stored"
    )
    scope_fingerprint: str
    record_ids: list[str] = Field(default_factory=list)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
