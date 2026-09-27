"""Durable, revision-aware progress for bounded background consolidation."""

import sqlalchemy as sa
from alembic import op

revision = "0011_reflection_progress"
down_revision = "0010_memory_recent_updates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_reflection_progress",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column(
            "memory_id",
            sa.String(200),
            sa.ForeignKey("memories.memory_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("source_revision", sa.Integer(), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("memory_reflection_progress")
