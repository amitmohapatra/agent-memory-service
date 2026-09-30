"""The final surface (OVERHAUL-SPEC-2 section C).

- Removed with their routes: memory webhooks (webhook_subscriptions, webhook_deliveries),
  groups (user_groups, user_group_members), standing briefs (standing_briefs), and the
  memories.group_id column nothing wrote.
- Model keys and policy: keys at the agent (``agent:<agent_id>``, whichever user it acts for)
  and tenant levels only - workspace keys go, a user-bound agent key becomes its agent's
  (the newest wins); the policy is the tenant's alone and names the model per use; the
  ``briefs`` use is gone.
- Tools: the catalog carries MCP annotations and ``approve_when``.
- Profile blocks may carry a standing question the profile job answers (source_query, the
  context it was set in, when it is due), claimed through a partial index.
- API keys say whom they may act for (``may_act_as``; every existing key: anyone).
- Feedback: an ANSWER verdict is a RUN verdict (on the run that answered), BRIEF verdicts go;
  a run outcome's source is the feedback source that decided it (explicit -> human,
  feedback -> system).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0021_final_surface"
down_revision = "0020_drop_query_decomposition"
branch_labels = None
depends_on = None

_JSON_OBJECT = sa.text("'{}'::jsonb")


def upgrade() -> None:
    op.drop_table("webhook_deliveries")
    op.drop_table("webhook_subscriptions")
    op.drop_table("user_group_members")
    op.drop_table("user_groups")
    op.drop_table("standing_briefs")
    op.drop_column("memories", "group_id")

    op.execute("DELETE FROM agent_credentials WHERE principal_id LIKE 'workspace:%'")
    op.execute(
        """
        DELETE FROM agent_credentials a
        USING agent_credentials b
        WHERE a.tenant_id = b.tenant_id
          AND a.principal_id LIKE 'agent:%/%'
          AND split_part(a.principal_id, '/', 2) <> ''
          AND (
            b.principal_id = 'agent:' || split_part(a.principal_id, '/', 2)
            OR (
              b.principal_id LIKE 'agent:%/%'
              AND split_part(b.principal_id, '/', 2) = split_part(a.principal_id, '/', 2)
              AND (b.updated_at, b.principal_id) > (a.updated_at, a.principal_id)
            )
          )
        """
    )
    op.execute(
        "UPDATE agent_credentials SET principal_id = 'agent:' || split_part(principal_id, '/', 2) "
        "WHERE principal_id LIKE 'agent:%/%'"
    )
    op.execute("DELETE FROM agent_credentials WHERE principal_id LIKE 'user:%'")

    op.execute("DELETE FROM llm_policies WHERE principal_id <> 'tenant'")
    op.drop_constraint("llm_policies_pkey", "llm_policies", type_="primary")
    op.drop_column("llm_policies", "principal_id")
    op.create_primary_key("llm_policies_pkey", "llm_policies", ["tenant_id"])
    op.add_column(
        "llm_policies",
        sa.Column("models", postgresql.JSONB(), nullable=False, server_default=_JSON_OBJECT),
    )
    op.execute("UPDATE llm_policies SET uses = array_remove(uses, 'briefs')")

    op.add_column(
        "tools",
        sa.Column("annotations", postgresql.JSONB(), nullable=False, server_default=_JSON_OBJECT),
    )
    op.add_column("tools", sa.Column("approve_when", sa.Text(), nullable=True))

    op.add_column("profile_blocks", sa.Column("source_query", sa.Text(), nullable=True))
    op.add_column("profile_blocks", sa.Column("source_context", postgresql.JSONB(), nullable=True))
    op.add_column(
        "profile_blocks", sa.Column("refresh_due_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index(
        "ix_profile_blocks_due",
        "profile_blocks",
        ["refresh_due_at"],
        postgresql_where=sa.text("source_query IS NOT NULL"),
    )

    op.add_column(
        "api_keys",
        sa.Column(
            "may_act_as",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[\"*\"]'::jsonb"),
        ),
    )

    op.execute(
        "UPDATE feedback SET target_kind = 'run', target_id = agent_run_id "
        "WHERE target_kind = 'answer' AND agent_run_id IS NOT NULL"
    )
    op.execute("DELETE FROM feedback WHERE target_kind IN ('answer', 'brief')")
    op.execute("UPDATE run_outcomes SET source = 'human' WHERE source = 'explicit'")
    op.execute("UPDATE run_outcomes SET source = 'system' WHERE source = 'feedback'")
    op.alter_column("run_outcomes", "source", server_default="system")


def downgrade() -> None:
    """The previous schema, with the removed features' tables empty (their rows are gone)."""
    now = sa.text("now()")
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
    op.create_table(
        "user_groups",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("group_id", sa.String(200), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "user_group_members",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("group_id", sa.String(200), primary_key=True),
        sa.Column("user_id", sa.String(200), primary_key=True),
        sa.Column("added_by", sa.String(512), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
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
    op.alter_column("run_outcomes", "source", server_default="explicit")
    op.execute("UPDATE run_outcomes SET source = 'explicit' WHERE source IN ('human', 'judge')")
    op.execute("UPDATE run_outcomes SET source = 'feedback' WHERE source = 'system'")
    op.drop_column("api_keys", "may_act_as")
    op.drop_index("ix_profile_blocks_due", table_name="profile_blocks")
    op.drop_column("profile_blocks", "refresh_due_at")
    op.drop_column("profile_blocks", "source_context")
    op.drop_column("profile_blocks", "source_query")
    op.drop_column("tools", "approve_when")
    op.drop_column("tools", "annotations")
    op.drop_column("llm_policies", "models")
    op.drop_constraint("llm_policies_pkey", "llm_policies", type_="primary")
    op.add_column(
        "llm_policies",
        sa.Column("principal_id", sa.String(512), nullable=False, server_default="tenant"),
    )
    op.create_primary_key("llm_policies_pkey", "llm_policies", ["tenant_id", "principal_id"])
    op.add_column("memories", sa.Column("group_id", sa.String(200), nullable=True))
