"""Derived-memory source lifecycle and unique live consolidation slots."""

import sqlalchemy as sa
from alembic import op

revision = "0009_memory_dependencies"
down_revision = "0008_drop_entity_alias"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("derived_slot", sa.String(64), nullable=True))
    op.create_index(
        "uq_memories_live_derived_slot",
        "memories",
        ["tenant_id", "derived_slot"],
        unique=True,
        postgresql_where=sa.text(
            "derived_slot IS NOT NULL AND temporal_status = 'CURRENT' AND deleted_at IS NULL"
        ),
    )
    op.create_table(
        "memory_dependencies",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column(
            "derived_id",
            sa.String(200),
            sa.ForeignKey("memories.memory_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "source_id",
            sa.String(200),
            sa.ForeignKey("memories.memory_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("source_revision", sa.Integer(), nullable=False),
    )
    op.create_index(
        "ix_memory_dependencies_source", "memory_dependencies", ["tenant_id", "source_id"]
    )

    # Legacy generated records have no captured source revisions. Rebuilding is safer
    # than asserting their old text still follows today's sources. Reconcile removes
    # the index copies; the retrieval guard also rejects them before that finishes.
    op.execute("""
        UPDATE memories SET temporal_status = 'RETRACTED', indexed_at = NULL,
            revision = revision + 1, updated_at = now()
        WHERE deleted_at IS NULL AND temporal_status = 'CURRENT'
          AND (memory_type IN ('BELIEF', 'ENTITY_SUMMARY')
               OR system_metadata->>'category' = 'reflection')
    """)


def downgrade() -> None:
    op.drop_table("memory_dependencies")
    op.drop_index("uq_memories_live_derived_slot", table_name="memories")
    op.drop_column("memories", "derived_slot")
