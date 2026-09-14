"""Claim-level provenance.

Every canonical memory / fact keeps ``EvidenceRef`` objects pointing at the raw source
it was derived from. OpenLineage tracks *processing* lineage and OpenTelemetry tracks
*execution* traces; ``EvidenceRef`` is the third, claim-level layer. They correlate by ID.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRef(BaseModel):
    """Pointer from a derived object back to the raw evidence that supports it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: str = Field(
        default=...,
        description="message | file | document_chunk | agent_result | tool_result | import",
    )
    source_id: str = Field(..., description="Primary identifier of the source object.")
    message_id: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    chunk_id: str | None = None
    node_id: str | None = Field(default=None, description="Document Context Graph node.")
    page: int | None = None
    span_start: int | None = Field(default=None, description="Character offset in source text.")
    span_end: int | None = None
    agent_id: str | None = None
    agent_run_id: str | None = None
    tool_run_id: str | None = None
    observed_at: datetime
    source_hash: str | None = Field(default=None, description="SHA-256 of the source bytes/text.")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    def citation_key(self) -> str:
        """Short, stable key for citations: prefers the most specific locator."""
        for name in ("chunk_id", "node_id", "message_id", "document_id", "source_id"):
            value = getattr(self, name)
            if value:
                return f"{name}:{value}"
        return f"source:{self.source_id}"


class EvidenceGroup(BaseModel):
    """A set of evidence any of which satisfies one required piece of a golden case.

    ``Evidence-Group Recall`` counts a group as recalled when at least one of its
    members is present in the retrieved set. All groups recalled => complete context.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    any_of: list[str] = Field(..., min_length=1, description="citation keys / chunk ids")
