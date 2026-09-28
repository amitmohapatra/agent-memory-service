"""Feedback records, webhook subscriptions and their deliveries (ADR 0023)."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0015_feedback_and_webhooks"
down_revision = "0014_tenancy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("feedback_id", sa.String(200), primary_key=True),
        sa.Column("workspace_id", sa.String(200), nullable=True),
        sa.Column("user_id", sa.String(200), nullable=True),
        sa.Column("agent_id", sa.String(200), nullable=True),
        sa.Column("agent_run_id", sa.String(200), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("target_kind", sa.String(20), nullable=False),
        sa.Column("target_id", sa.String(200), nullable=False),
        sa.Column("verdict", sa.String(20), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("correction", postgresql.JSONB(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("comment", sa.String(4000), nullable=True),
        sa.Column("reviewer", sa.String(200), nullable=True),
        sa.Column("evidence_refs", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("projection", postgresql.JSONB(), nullable=True),
        sa.Column("projected_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_feedback_target",
        "feedback",
        [
            "tenant_id",
            "target_kind",
            "target_id",
            sa.text("created_at DESC"),
            sa.text("feedback_id DESC"),
        ],
    )
    op.create_table(
        "webhook_subscriptions",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("subscription_id", sa.String(200), primary_key=True),
        sa.Column("workspace_id", sa.String(200), nullable=True),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("events", postgresql.JSONB(), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("secret_key_id", sa.String(100), nullable=False),
        sa.Column("secret_ciphertext", sa.LargeBinary(), nullable=False),
    )
    op.create_table(
        "webhook_deliveries",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("delivery_id", sa.String(200), primary_key=True),
        sa.Column("subscription_id", sa.String(200), nullable=False),
        sa.Column("event_id", sa.String(200), nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_webhook_deliveries_subscription",
        "webhook_deliveries",
        ["tenant_id", "subscription_id", sa.text("created_at DESC"), sa.text("delivery_id DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_webhook_deliveries_subscription", table_name="webhook_deliveries")
    op.drop_table("webhook_deliveries")
    op.drop_table("webhook_subscriptions")
    op.drop_index("ix_feedback_target", table_name="feedback")
    op.drop_table("feedback")
