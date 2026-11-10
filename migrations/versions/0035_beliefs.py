"""Beliefs: what is held about one person, consolidated from their memories.

``beliefs`` (one facet of a person a line, with supersession as memories have it),
``belief_sources`` (the memories each cites) and ``belief_progress`` (where each person's
consolidation in a scope stands). Derived data: rebuildable from the memories, so nothing is
backfilled.

Indexes: a unit's live beliefs (consolidation's read), a reader's beliefs by audience,
current and earlier (the context's read, GIN like ``graph_entities``), the sources of a memory (the delete
cascade), and the waiting units (the lag metric and the retry sweep).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0035_beliefs"
down_revision = "0034_tool_permissions"
branch_labels = None
depends_on = None

_LIVE = "superseded_by IS NULL AND deleted_at IS NULL"
_TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "beliefs",
        sa.Column("belief_id", sa.String(200), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("scope_key", sa.String(600), nullable=False),
        sa.Column("visibility_keys", JSONB(), server_default="[]", nullable=False),
        sa.Column("person_key", sa.String(300), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("facet", sa.String(100), server_default="", nullable=False),
        sa.Column("first_seen_at", _TS, nullable=False),
        sa.Column("last_seen_at", _TS, nullable=False),
        sa.Column("proof_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("supersedes", sa.String(200)),
        sa.Column("superseded_by", sa.String(200)),
        sa.Column("valid_from", _TS),
        sa.Column("valid_to", _TS),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("index_fingerprint", sa.String(200)),
        sa.Column("revision", sa.Integer(), server_default="1", nullable=False),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", _TS),
    )
    op.create_index(
        "ix_beliefs_unit",
        "beliefs",
        ["tenant_id", "scope_key", "person_key"],
        postgresql_where=sa.text(_LIVE),
    )
    op.create_index(
        "ix_beliefs_keys",
        "beliefs",
        ["visibility_keys"],
        postgresql_using="gin",
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_table(
        "belief_sources",
        sa.Column(
            "belief_id",
            sa.String(200),
            sa.ForeignKey("beliefs.belief_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "memory_id",
            sa.String(200),
            sa.ForeignKey("memories.memory_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("added_at", _TS, server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_belief_sources_memory", "belief_sources", ["tenant_id", "memory_id"])
    op.create_table(
        "belief_progress",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("scope_key", sa.String(600), primary_key=True),
        sa.Column("person_key", sa.String(300), primary_key=True),
        sa.Column("source_fingerprint", sa.String(64), server_default="", nullable=False),
        sa.Column("last_consolidated_at", _TS),
        sa.Column("pending_since", _TS),
        sa.Column("failures", sa.Integer(), server_default="0", nullable=False),
        sa.Column("paused_at", _TS),
    )
    op.create_index(
        "ix_belief_progress_pending",
        "belief_progress",
        ["pending_since"],
        postgresql_where=sa.text("pending_since IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("belief_progress")
    op.drop_table("belief_sources")
    op.drop_table("beliefs")
