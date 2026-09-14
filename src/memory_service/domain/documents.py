"""Document, structural node and chunk contracts (RAG is part of the Memory Service).

A raw file becomes a Document; parsing produces a DocumentVersion with a tree of
DocumentNodes (section > subsection > paragraph/table/code). Chunks are only created when
a natural unit exceeds limits. The Document Context Graph links nodes structurally and is
separate from the semantic Knowledge Graph.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import ArchiveStatus, ContextGraphEdge, Representation
from memory_service.domain.ids import new_id


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(default_factory=lambda: new_id("document"))
    tenant_id: str
    workspace_id: str | None = None
    owner_user_id: str | None = None
    thread_id: str | None = None
    title: str
    filename: str
    media_type: str
    size_bytes: int
    checksum: str = Field(..., description="SHA-256 of raw bytes; drives file-level dedup")
    current_version_id: str | None = None
    source_system: str | None = None
    source_id: str | None = None
    archive_status: ArchiveStatus = ArchiveStatus.STAGED
    system_metadata: dict[str, Any] = Field(default_factory=dict)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    revision: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = None


class DocumentVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_version_id: str = Field(default_factory=lambda: new_id("document_version"))
    document_id: str
    tenant_id: str
    version: int = 1
    parser: str = Field(..., description="docling | markdown | text | ...")
    parser_version: str | None = None
    page_count: int | None = None
    node_count: int = 0
    chunk_count: int = 0
    status: str = "PARSED"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DocumentNode(BaseModel):
    """A node in the document hierarchy. ``text`` is the *original* text."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(default_factory=lambda: new_id("node"))
    document_id: str
    document_version_id: str
    tenant_id: str
    representation: Representation
    parent_id: str | None = None
    ordinal: int = Field(..., ge=0, description="position among siblings")
    depth: int = Field(..., ge=0)
    title: str | None = Field(default=None, description="heading text for sections")
    section_path: str = Field(default="", description="'Doc > Section > Subsection'")
    page_start: int | None = None
    page_end: int | None = None
    text: str = ""
    text_hash: str = ""
    token_estimate: int = 0
    entities: list[str] = Field(default_factory=list)
    system_metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """An indexable unit. ``text`` is original; ``contextual_text`` is what gets indexed."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(default_factory=lambda: new_id("chunk"))
    node_id: str
    document_id: str
    document_version_id: str
    tenant_id: str
    ordinal: int = Field(..., ge=0, description="position within the node")
    text: str
    text_hash: str
    contextual_text: str = Field(..., description="deterministic context header + text")
    page: int | None = None
    section_path: str = ""
    token_estimate: int = 0
    entities: list[str] = Field(default_factory=list)


class ContextEdge(BaseModel):
    """Structural link between two nodes/chunks of the Document Context Graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    document_id: str
    source_id: str
    target_id: str
    edge: ContextGraphEdge
    weight: float = 1.0
    label: str | None = Field(default=None, description="entity name / reference label")
