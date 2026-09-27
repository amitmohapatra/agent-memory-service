"""Agent-scoped encrypted model keys and durable revocation tombstones."""

import sqlalchemy as sa
from alembic import op

revision = "0012_agent_credentials"
down_revision = "0011_reflection_progress"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_credentials",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("principal_id", sa.String(512), primary_key=True),
        sa.Column("key_id", sa.String(100), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("agent_credentials")
