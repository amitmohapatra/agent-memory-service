"""tool_memory: tool descriptors, invocation records and run outcomes

Revision ID: 0007_tool_memory
Revises: 0006_kg_arch
Create Date: 2026-09-15 16:00:00.000000+00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0007_tool_memory"
down_revision = "0006_kg_arch"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tools",
        sa.Column("tool_id", sa.String(200), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("workspace_id", sa.String(200)),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("input_schema", JSONB()),
        sa.Column("output_schema", JSONB()),
        sa.Column("tags", JSONB(), nullable=False, server_default="[]"),
        sa.Column("source", sa.String(20), nullable=False, server_default="manual"),
        sa.Column("server", sa.String(200)),
        sa.Column("policy", JSONB(), nullable=False, server_default="{}"),
        sa.Column("schema_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "name", "schema_hash", name="uq_tools_tenant_name_schema"),
    )
    op.create_index("ix_tools_tenant_name", "tools", ["tenant_id", "name"])

    op.create_table(
        "tool_invocations",
        sa.Column("invocation_id", sa.String(200), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("tool_id", sa.String(200), nullable=False),
        sa.Column("tool_name", sa.String(200), nullable=False),
        sa.Column("tool_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("run_id", sa.String(200)),
        sa.Column("thread_id", sa.String(200)),
        sa.Column("turn_id", sa.String(200)),
        sa.Column("workspace_id", sa.String(200)),
        sa.Column("user_id", sa.String(200)),
        sa.Column("agent_id", sa.String(200)),
        sa.Column("principal_id", sa.String(300)),
        sa.Column("step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("args_redacted", JSONB(), nullable=False, server_default="{}"),
        sa.Column("args_hash", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("output_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("output_digest", sa.String(64)),
        sa.Column("output_blob_ref", sa.String(600)),
        sa.Column("output_fields", JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(20), nullable=False, server_default="ok"),
        sa.Column("error_class", sa.String(200)),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("cost", sa.Float()),
        sa.Column("task", sa.Text(), nullable=False, server_default=""),
        sa.Column("task_pattern", sa.Text()),
        sa.Column("sub_calls", JSONB(), nullable=False, server_default="[]"),
        sa.Column("visibility_keys", JSONB(), nullable=False, server_default="[]"),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_tool_invocations_idempotent"
        ),
    )
    op.create_index(
        "ix_tool_invocations_run", "tool_invocations", ["tenant_id", "run_id", "step"]
    )
    op.create_index(
        "ix_tool_invocations_tool", "tool_invocations", ["tenant_id", "tool_id", "occurred_at"]
    )
    op.create_index("ix_tool_invocations_pattern", "tool_invocations", ["tenant_id", "task_pattern"])
    op.create_index(
        "ix_tool_invocations_unindexed", "tool_invocations", ["tenant_id", "indexed_at"]
    )

    op.create_table(
        "run_outcomes",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("run_id", sa.String(200), primary_key=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("note", sa.Text()),
        sa.Column("source", sa.String(20), nullable=False, server_default="explicit"),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_run_outcomes_tenant_success", "run_outcomes", ["tenant_id", "success"])


def downgrade() -> None:
    op.drop_table("run_outcomes")
    op.drop_table("tool_invocations")
    op.drop_table("tools")
