"""SQLAlchemy 2 declarative tables for canonical state.

Only the persistence adapter and Alembic import this module. Application code works with
domain models; repositories translate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB, datetime: DateTime(timezone=True)}


def _now() -> Any:
    return text("now()")


# --------------------------------------------------------------------------
# Conversation
# --------------------------------------------------------------------------


class ThreadRow(Base):
    __tablename__ = "threads"

    thread_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    workspace_id: Mapped[str | None] = mapped_column(String(200))
    owner_user_id: Mapped[str | None] = mapped_column(String(200))
    title: Mapped[str | None] = mapped_column(Text)
    source_system: Mapped[str | None] = mapped_column(String(100))
    source_thread_id: Mapped[str | None] = mapped_column(String(400))
    system_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    updated_at: Mapped[datetime] = mapped_column(server_default=_now())
    archived_at: Mapped[datetime | None]
    deleted_at: Mapped[datetime | None]

    __table_args__ = (
        Index("ix_threads_tenant_owner_updated", "tenant_id", "owner_user_id", "updated_at"),
        Index("ix_threads_tenant_workspace", "tenant_id", "workspace_id"),
        Index("ix_threads_source", "tenant_id", "source_system", "source_thread_id"),
    )


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.thread_id", ondelete="CASCADE"))
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(200))
    client: Mapped[str | None] = mapped_column(String(50))
    started_at: Mapped[datetime] = mapped_column(server_default=_now())
    ended_at: Mapped[datetime | None]
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )

    __table_args__ = (Index("ix_sessions_thread", "thread_id", "started_at"),)


class TurnRow(Base):
    __tablename__ = "turns"

    turn_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"))
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.thread_id", ondelete="CASCADE"))
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(server_default=_now())
    completed_at: Mapped[datetime | None]
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )

    __table_args__ = (UniqueConstraint("thread_id", "sequence", name="uq_turns_thread_sequence"),)


class MessageRow(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.thread_id", ondelete="CASCADE"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"))
    turn_id: Mapped[str] = mapped_column(ForeignKey("turns.turn_id", ondelete="CASCADE"))
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="VISIBLE")
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    # staged raw payload; may be purged (set NULL) after archive verification + grace period
    content: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_bytes: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    author_principal: Mapped[str] = mapped_column(String(300), nullable=False)
    agent_run_id: Mapped[str | None] = mapped_column(String(200))
    parent_message_id: Mapped[str | None] = mapped_column(String(200))
    source_system: Mapped[str | None] = mapped_column(String(100))
    source_message_id: Mapped[str | None] = mapped_column(String(400))
    occurred_at: Mapped[datetime] = mapped_column(server_default=_now())
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    archive_status: Mapped[str] = mapped_column(
        String(20), default="STAGED", server_default="STAGED"
    )
    archived_at: Mapped[datetime | None]
    archive_segment_id: Mapped[str | None] = mapped_column(String(200))
    system_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    deleted_at: Mapped[datetime | None]

    __table_args__ = (
        UniqueConstraint("thread_id", "sequence", name="uq_messages_thread_sequence"),
        Index("ix_messages_tenant_thread_seq", "tenant_id", "thread_id", "sequence"),
        Index("ix_messages_turn", "turn_id"),
        Index(
            "ix_messages_staged",
            "tenant_id",
            "created_at",
            postgresql_where=text("archive_status = 'STAGED'"),
        ),
        Index("ix_messages_source", "tenant_id", "source_system", "source_message_id"),
    )


class MessageVersionRow(Base):
    __tablename__ = "message_versions"

    message_version_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.message_id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=_now())

    __table_args__ = (UniqueConstraint("message_id", "version", name="uq_message_versions"),)


class MessageAttachmentRow(Base):
    __tablename__ = "message_attachments"

    attachment_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.message_id", ondelete="CASCADE"))
    document_id: Mapped[str | None] = mapped_column(String(200))
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str] = mapped_column(String(200), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (Index("ix_message_attachments_message", "message_id"),)


class AgentRunRow(Base):
    __tablename__ = "agent_runs"

    agent_run_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    thread_id: Mapped[str | None] = mapped_column(String(200))
    session_id: Mapped[str | None] = mapped_column(String(200))
    turn_id: Mapped[str | None] = mapped_column(String(200))
    work_id: Mapped[str | None] = mapped_column(String(200))
    task_id: Mapped[str | None] = mapped_column(String(200))
    agent_id: Mapped[str] = mapped_column(String(200), nullable=False)
    agent_group_id: Mapped[str | None] = mapped_column(String(200))
    parent_agent_run_id: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="RUNNING", server_default="RUNNING")
    started_at: Mapped[datetime] = mapped_column(server_default=_now())
    completed_at: Mapped[datetime | None]
    trace_id: Mapped[str | None] = mapped_column(String(200))
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )

    __table_args__ = (
        Index("ix_agent_runs_turn", "tenant_id", "turn_id"),
        Index("ix_agent_runs_parent", "parent_agent_run_id"),
    )


class TurnRunLinkRow(Base):
    __tablename__ = "turn_run_links"

    turn_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(String(200), primary_key=True)


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


class ObservationRow(Base):
    __tablename__ = "observations"

    observation_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    workspace_id: Mapped[str | None] = mapped_column(String(200))
    user_id: Mapped[str | None] = mapped_column(String(200))
    thread_id: Mapped[str | None] = mapped_column(String(200))
    session_id: Mapped[str | None] = mapped_column(String(200))
    turn_id: Mapped[str | None] = mapped_column(String(200))
    work_id: Mapped[str | None] = mapped_column(String(200))
    task_id: Mapped[str | None] = mapped_column(String(200))
    agent_id: Mapped[str | None] = mapped_column(String(200))
    agent_group_id: Mapped[str | None] = mapped_column(String(200))
    agent_run_id: Mapped[str | None] = mapped_column(String(200))
    parent_agent_run_id: Mapped[str | None] = mapped_column(String(200))
    principal_id: Mapped[str] = mapped_column(String(300), nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(200))
    message_id: Mapped[str | None] = mapped_column(String(200))
    document_id: Mapped[str | None] = mapped_column(String(200))
    tool_run_id: Mapped[str | None] = mapped_column(String(200))
    source_system: Mapped[str | None] = mapped_column(String(100))
    source_id: Mapped[str | None] = mapped_column(String(400))
    hints: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    occurred_at: Mapped[datetime] = mapped_column(server_default=_now())
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    processed_at: Mapped[datetime | None]
    status: Mapped[str] = mapped_column(String(20), default="PENDING", server_default="PENDING")

    __table_args__ = (
        Index("ix_observations_tenant_created", "tenant_id", "created_at"),
        Index("ix_observations_thread", "tenant_id", "thread_id", "created_at"),
        Index(
            "ix_observations_pending",
            "created_at",
            postgresql_where=text("processed_at IS NULL"),
        ),
    )


# --------------------------------------------------------------------------
# Infrastructure: revisions, idempotency, outbox
# --------------------------------------------------------------------------


class RevisionRow(Base):
    __tablename__ = "revisions"

    tenant_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), primary_key=True)
    object_id: Mapped[str] = mapped_column(String(200), primary_key=True, default="")
    value: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(server_default=_now())


class IdempotencyRow(Base):
    __tablename__ = "idempotency_keys"

    tenant_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    key: Mapped[str] = mapped_column(String(300), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response_status: Mapped[int | None] = mapped_column(Integer)
    response_body: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    expires_at: Mapped[datetime] = mapped_column(nullable=False)

    __table_args__ = (Index("ix_idempotency_expires", "expires_at"),)


class OutboxRow(Base):
    """Transactional outbox: a job row committed with the source data.

    After commit the relay dispatches the row to the task queue; a periodic sweep re-dispatches
    rows whose dispatch never happened (process crash between COMMIT and defer).
    """

    __tablename__ = "job_outbox"

    outbox_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str | None] = mapped_column(String(200))
    task_name: Mapped[str] = mapped_column(String(200), nullable=False)
    queue: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    idempotency_key: Mapped[str | None] = mapped_column(String(300))
    lock: Mapped[str | None] = mapped_column(String(300))
    priority: Mapped[int | None] = mapped_column(Integer)
    schedule_in_seconds: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    dispatched_at: Mapped[datetime | None]
    job_id: Mapped[str | None] = mapped_column(String(200))
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text)
    dead: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    __table_args__ = (
        Index(
            "uq_job_outbox_idempotency",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
        Index(
            "ix_job_outbox_pending",
            "created_at",
            postgresql_where=text("dispatched_at IS NULL AND dead = false"),
        ),
    )


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------


class ArchiveSegmentRow(Base):
    __tablename__ = "archive_segments"

    segment_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="chat")
    thread_id: Mapped[str | None] = mapped_column(String(200))
    document_id: Mapped[str | None] = mapped_column(String(200))
    bucket: Mapped[str] = mapped_column(String(200), nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    generation: Mapped[str | None] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    raw_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    first_sequence: Mapped[int | None] = mapped_column(Integer)
    last_sequence: Mapped[int | None] = mapped_column(Integer)
    first_at: Mapped[datetime | None]
    last_at: Mapped[datetime | None]
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="UPLOADING")
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    verified_at: Mapped[datetime | None]
    last_error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_archive_segments_thread", "tenant_id", "thread_id", "first_sequence"),
        Index("ix_archive_segments_status", "status", "created_at"),
        Index("uq_archive_segments_key", "bucket", "key", unique=True),
    )


# --------------------------------------------------------------------------
# Documents / RAG
# --------------------------------------------------------------------------


class DocumentRow(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    workspace_id: Mapped[str | None] = mapped_column(String(200))
    owner_user_id: Mapped[str | None] = mapped_column(String(200))
    thread_id: Mapped[str | None] = mapped_column(String(200))
    message_id: Mapped[str | None] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str] = mapped_column(String(200), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    current_version_id: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="STAGED")
    source_system: Mapped[str | None] = mapped_column(String(100))
    source_id: Mapped[str | None] = mapped_column(String(400))
    archive_status: Mapped[str] = mapped_column(
        String(20), default="STAGED", server_default="STAGED"
    )
    archive_segment_id: Mapped[str | None] = mapped_column(String(200))
    archived_at: Mapped[datetime | None]
    visibility_keys: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=list, server_default="[]"
    )
    system_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    updated_at: Mapped[datetime] = mapped_column(server_default=_now())
    deleted_at: Mapped[datetime | None]

    __table_args__ = (
        Index("ix_documents_tenant_checksum", "tenant_id", "checksum"),
        Index("ix_documents_tenant_thread", "tenant_id", "thread_id"),
        Index("ix_documents_status", "status", "created_at"),
    )


class FileStagingRow(Base):
    """Raw bytes staged durably until the blob archive is verified (then purged)."""

    __tablename__ = "file_staging"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), primary_key=True
    )
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    data: Mapped[bytes | None] = mapped_column(LargeBinary)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    purged_at: Mapped[datetime | None]


class DocumentVersionRow(Base):
    __tablename__ = "document_versions"

    document_version_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE")
    )
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parser: Mapped[str] = mapped_column(String(50), nullable=False)
    parser_version: Mapped[str | None] = mapped_column(String(50))
    page_count: Mapped[int | None] = mapped_column(Integer)
    node_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="PARSED")
    created_at: Mapped[datetime] = mapped_column(server_default=_now())

    __table_args__ = (Index("ix_document_versions_document", "document_id", "version"),)


class DocumentNodeRow(Base):
    __tablename__ = "document_nodes"

    node_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE")
    )
    document_version_id: Mapped[str] = mapped_column(String(200), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    representation: Mapped[str] = mapped_column(String(30), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String(200))
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    section_path: Mapped[str] = mapped_column(Text, default="")
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    text_hash: Mapped[str] = mapped_column(String(64), default="")
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    entities: Mapped[dict[str, Any]] = mapped_column(JSONB, default=list, server_default="[]")
    system_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )

    __table_args__ = (
        Index("ix_document_nodes_version", "document_version_id", "depth", "ordinal"),
        Index("ix_document_nodes_parent", "parent_id"),
    )


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(200), nullable=False)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE")
    )
    document_version_id: Mapped[str] = mapped_column(String(200), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    contextual_text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[str] = mapped_column(Text, default="")
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    entities: Mapped[dict[str, Any]] = mapped_column(JSONB, default=list, server_default="[]")
    indexed_at: Mapped[datetime | None]
    index_fingerprint: Mapped[str | None] = mapped_column(String(100))

    __table_args__ = (
        Index("ix_chunks_version", "document_version_id", "node_id", "ordinal"),
        Index("ix_chunks_tenant_document", "tenant_id", "document_id"),
        Index("ix_chunks_text_hash", "tenant_id", "text_hash"),
    )


class ContextEdgeRow(Base):
    __tablename__ = "context_edges"

    edge_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE")
    )
    source_id: Mapped[str] = mapped_column(String(300), nullable=False)
    target_id: Mapped[str] = mapped_column(String(300), nullable=False)
    edge: Mapped[str] = mapped_column(String(30), nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    label: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_context_edges_source", "source_id", "edge"),
        Index("ix_context_edges_target", "target_id", "edge"),
        Index("ix_context_edges_document", "document_id"),
    )


# --------------------------------------------------------------------------
# Memory intelligence (M7): canonical memories derived from observations
# --------------------------------------------------------------------------


class MemoryRow(Base):
    __tablename__ = "memories"

    memory_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    scope_level: Mapped[str] = mapped_column(String(20), nullable=False)
    scope_key: Mapped[str] = mapped_column(String(600), nullable=False)
    workspace_id: Mapped[str | None] = mapped_column(String(200))
    user_id: Mapped[str | None] = mapped_column(String(200))
    group_id: Mapped[str | None] = mapped_column(String(200))
    thread_id: Mapped[str | None] = mapped_column(String(200))
    work_id: Mapped[str | None] = mapped_column(String(200))
    agent_id: Mapped[str | None] = mapped_column(String(200))
    agent_group_id: Mapped[str | None] = mapped_column(String(200))
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    visibility_keys: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=list, server_default="[]"
    )
    owner_principal: Mapped[str] = mapped_column(String(300), nullable=False)
    lifetime: Mapped[str] = mapped_column(String(20), nullable=False)
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False)
    custom_type: Mapped[str | None] = mapped_column(String(100))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(300))
    predicate: Mapped[str | None] = mapped_column(String(200))
    object: Mapped[str | None] = mapped_column(Text)
    temporal_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="CURRENT", server_default="CURRENT"
    )
    valid_from: Mapped[datetime | None]
    valid_to: Mapped[datetime | None]
    observed_at: Mapped[datetime] = mapped_column(server_default=_now())
    superseded_by: Mapped[str | None] = mapped_column(String(200))
    supersedes: Mapped[str | None] = mapped_column(String(200))
    contradicts: Mapped[dict[str, Any]] = mapped_column(JSONB, default=list, server_default="[]")
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, default=list, server_default="[]")
    confidence: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")
    importance: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")
    reinforcement_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    provider: Mapped[str] = mapped_column(String(40), default="native", server_default="native")
    system_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    expires_at: Mapped[datetime | None]
    indexed_at: Mapped[datetime | None]
    index_fingerprint: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(server_default=_now())
    updated_at: Mapped[datetime] = mapped_column(server_default=_now())
    revision: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    deleted_at: Mapped[datetime | None]

    __table_args__ = (
        Index("ix_memories_tenant_hash", "tenant_id", "normalized_hash"),
        Index("ix_memories_tenant_scope", "tenant_id", "scope_key", "temporal_status"),
        Index("ix_memories_tenant_subject", "tenant_id", "subject", "predicate"),
        Index("ix_memories_tenant_user", "tenant_id", "user_id", "created_at"),
        Index("ix_memories_tenant_thread", "tenant_id", "thread_id"),
        Index(
            "ix_memories_unindexed",
            "tenant_id",
            "created_at",
            postgresql_where=text("indexed_at IS NULL AND deleted_at IS NULL"),
        ),
        Index(
            "ix_memories_expiring",
            "expires_at",
            postgresql_where=text("expires_at IS NOT NULL AND deleted_at IS NULL"),
        ),
    )
