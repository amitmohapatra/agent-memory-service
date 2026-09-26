"""Bounded recent-change discovery for continuous memory consolidation."""

import sqlalchemy as sa
from alembic import op

revision = "0010_memory_recent_updates"
down_revision = "0009_memory_dependencies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_memories_recent_updates",
        "memories",
        ["updated_at", "memory_id"],
        postgresql_where=sa.text("temporal_status = 'CURRENT' AND deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_memories_recent_updates", table_name="memories")
