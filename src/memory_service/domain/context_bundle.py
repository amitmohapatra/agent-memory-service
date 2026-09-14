"""ContextBundle: what an application receives to answer the current turn.

Built by the thin ContextBuilder from scope-filtered retrieval results. Follows the
context-engineering discipline: write / select / compress / isolate. It never contains
everything; it contains bounded, ranked, provenance-carrying evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.evidence import EvidenceRef


class ContextItem(BaseModel):
    """One ranked piece of context."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    representation: Representation
    text: str
    score: float = 0.0
    retrievers: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    citation: str = Field(..., description="stable citation key")
    document_id: str | None = None
    page: int | None = None
    section_path: str | None = Field(default=None, description="e.g. 'Financial Results > EBITDA'")
    expanded_from: str | None = Field(default=None, description="item_id this was expanded from")
    expansion_edge: str | None = Field(default=None, description="PARENT | NEXT | DEFINED_BY | ...")
    token_estimate: int = 0


class ConversationWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    thread_id: str | None = None
    message_ids: list[str] = Field(default_factory=list)
    rendered: str = ""
    token_estimate: int = 0
    summary: str | None = Field(default=None, description="rolling summary of older messages")


class EvidenceReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: EvidenceStatus
    required_groups: list[str] = Field(default_factory=list)
    satisfied_groups: list[str] = Field(default_factory=list)
    missing_groups: list[str] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list, description="strategies attempted")
    notes: list[str] = Field(default_factory=list)


class ContextBundle(BaseModel):
    """Bounded, ranked context for one query in one execution context."""

    model_config = ConfigDict(frozen=True)

    query: str
    query_type: QueryType
    conversation: ConversationWindow
    memories: list[ContextItem] = Field(default_factory=list)
    knowledge: list[ContextItem] = Field(default_factory=list)
    graph_facts: list[ContextItem] = Field(default_factory=list)
    summaries: list[ContextItem] = Field(default_factory=list)
    evidence: EvidenceReport
    token_budget: int
    token_estimate: int
    cache_hit: bool = False
    revision_fingerprint: str = Field(
        default="", description="revisions this bundle was built from"
    )
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        """Plain-text rendering suitable for a system prompt. Applications may ignore it."""
        parts: list[str] = []
        if self.conversation.summary:
            parts.append(f"## Conversation summary\n{self.conversation.summary}")
        if self.conversation.rendered:
            parts.append(f"## Recent conversation\n{self.conversation.rendered}")
        if self.memories:
            parts.append(
                "## Memories\n" + "\n".join(f"- [{m.citation}] {m.text}" for m in self.memories)
            )
        if self.graph_facts:
            parts.append(
                "## Facts\n" + "\n".join(f"- [{f.citation}] {f.text}" for f in self.graph_facts)
            )
        if self.summaries:
            parts.append(
                "## Summaries\n" + "\n".join(f"- [{s.citation}] {s.text}" for s in self.summaries)
            )
        if self.knowledge:
            parts.append(
                "## Knowledge\n"
                + "\n\n".join(
                    f"[{k.citation}]"
                    + (f" ({k.section_path})" if k.section_path else "")
                    + f"\n{k.text}"
                    for k in self.knowledge
                )
            )
        if self.evidence.status is not EvidenceStatus.COMPLETE:
            parts.append(f"## Evidence status\n{self.evidence.status}")
        return "\n\n".join(parts)
