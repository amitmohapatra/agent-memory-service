"""A person's tool permissions: "Always allow" for a tool, per tenant and user.

One row per (tenant, user, tool, assignee): the harness writes it when the person answers an
approval with "Always allow", and reads the user's rows at the start of a run, so the tool's
later calls (asked of the same person or role, ``assignee``) run without asking in any of the
user's conversations until the row is deleted. ``assignee`` is null for an approval anyone may
answer; nulls are not distinct in the unique constraint, so allowing twice keeps one row.
"""

import sqlalchemy as sa
from alembic import op

revision = "0034_tool_permissions"
down_revision = "0033_drop_agent_runs_turn_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_permissions",
        sa.Column("permission_id", sa.String(40), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("user_id", sa.String(200), nullable=False),
        sa.Column("tool", sa.String(200), nullable=False),
        sa.Column("assignee", sa.String(200), nullable=True),
        sa.Column("allowed_by", sa.String(512), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "user_id",
            "tool",
            "assignee",
            name="uq_tool_permissions",
            postgresql_nulls_not_distinct=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("tool_permissions")
