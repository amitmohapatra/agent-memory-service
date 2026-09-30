"""The overhaul's schema step.

- Drop the write-only conversation tables and columns (message versions, turn/run links,
  session end and turn completion): nothing ever read them.
- Entity search and summaries: a prefix index on the canonical name, a most-mentioned index,
  and the fingerprint of the facts an entity summary was written from.
"""

import sqlalchemy as sa
from alembic import op

revision = "0016_overhaul"
down_revision = "0015_feedback_and_webhooks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("message_versions")
    op.drop_table("turn_run_links")
    op.drop_column("sessions", "ended_at")
    op.drop_column("turns", "completed_at")
    op.add_column(
        "graph_entities",
        sa.Column("summary_source", sa.String(64), nullable=False, server_default=""),
    )
    op.create_index(
        "ix_graph_entities_name_prefix",
        "graph_entities",
        ["tenant_id", sa.text("canonical_name text_pattern_ops")],
    )
    op.create_index(
        "ix_graph_entities_tenant_mentions",
        "graph_entities",
        ["tenant_id", sa.text("mention_count DESC"), "entity_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_graph_entities_tenant_mentions", table_name="graph_entities")
    op.drop_index("ix_graph_entities_name_prefix", table_name="graph_entities")
    op.drop_column("graph_entities", "summary_source")
    op.add_column("turns", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("sessions", sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True))
    op.create_table(
        "turn_run_links",
        sa.Column("turn_id", sa.String(200), nullable=False),
        sa.Column("agent_run_id", sa.String(200), nullable=False),
        sa.PrimaryKeyConstraint("turn_id", "agent_run_id"),
    )
    op.create_table(
        "message_versions",
        sa.Column("message_version_id", sa.String(200), nullable=False),
        sa.Column("message_id", sa.String(200), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["message_id"], ["messages.message_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("message_version_id"),
        sa.UniqueConstraint("message_id", "version", name="uq_message_versions"),
    )
