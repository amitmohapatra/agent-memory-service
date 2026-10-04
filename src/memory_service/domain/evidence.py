"""Claim-level provenance.

Every canonical memory / fact keeps ``EvidenceRef`` objects pointing at the raw source
it was derived from. OpenLineage tracks *processing* lineage and OpenTelemetry tracks
*execution* traces; ``EvidenceRef`` is the third, claim-level layer. They correlate by ID.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EvidenceSource(StrEnum):
    """What kind of object a piece of evidence points at - every value the service writes."""

    #: a conversation message (``message_id``)
    MESSAGE = "message"
    #: an uploaded file before it was parsed into chunks
    FILE = "file"
    #: a passage of a parsed document (``document_id``, ``chunk_id``, ``page``)
    DOCUMENT_CHUNK = "document_chunk"
    #: what an agent run reported as its result
    AGENT_RESULT = "agent_result"
    #: what a tool call returned (``tool_run_id`` / the invocation)
    TOOL_RESULT = "tool_result"
    #: a record imported from another system
    IMPORT = "import"
    #: an observation of another kind (a decision, an event)
    OBSERVATION = "observation"
    #: a principal's own statement, stored verbatim (``POST /v1/memories``)
    STATEMENT = "statement"
    #: another memory this one was derived from (a consolidation, a derived slot)
    MEMORY = "memory"
    #: a fact of the knowledge graph
    GRAPH_FACT = "graph_fact"
    #: a document or thread summary
    SUMMARY = "summary"
    #: an earlier conversation of the user (one per thread)
    EPISODE = "episode"
    #: a feedback record that corrected or confirmed the memory
    FEEDBACK = "feedback"


EVIDENCE_SOURCE_DESCRIPTION = (
    "What the evidence points at: message, file, document_chunk, agent_result, tool_result, "
    "import, observation, statement (a principal's own words), memory (the memory it was "
    "derived from), graph_fact, summary, episode or feedback."
)


class EvidenceRef(BaseModel):
    """Pointer from a derived object back to the raw evidence that supports it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: EvidenceSource = Field(default=..., description=EVIDENCE_SOURCE_DESCRIPTION)
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
