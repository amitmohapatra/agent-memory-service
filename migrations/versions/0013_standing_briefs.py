"""Native standing questions and knowledge pages; no source corpus mutation."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0013_standing_briefs"
down_revision = "0012_agent_credentials"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "standing_briefs",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("brief_id", sa.String(200), primary_key=True),
        sa.Column("scope_key", sa.String(200), nullable=False),
        sa.Column("context", postgresql.JSONB(), nullable=False),
        sa.Column("spec", postgresql.JSONB(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("output", postgresql.JSONB(), nullable=True),
        sa.Column("next_refresh_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_briefs_due", "standing_briefs", ["next_refresh_at", "brief_id"])
    op.create_index("ix_briefs_scope", "standing_briefs", ["tenant_id", "scope_key", "brief_id"])


def downgrade() -> None:
    op.drop_table("standing_briefs")
