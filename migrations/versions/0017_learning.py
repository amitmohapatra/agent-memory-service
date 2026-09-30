"""What the service learns from use.

- The tool catalog: one entry per (tenant, workspace, name) with its required arguments,
  the entity type each argument names, its side effects, examples and redacted paths; the
  never-read policy, tags and output schema go.
- tool_stats: running counts per tool (calls, successes, latency, approvals, rejections,
  edits), one upsert per recorded call or tool-call verdict.
- approval_patterns: approve / reject / edit decisions per (agent, tool, argument shape).
- procedures: the procedure learned per (tenant, audience, task pattern).
- tool_invocations.learned_at (was the never-written indexed_at): the learning job's
  cursor, behind a partial index on the calls not learned yet.
- profile_blocks: pinned text per (scope, block); thread_summaries: a thread's durable
  summary, one row per version (the primary key reads the newest).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0017_learning"
down_revision = "0016_overhaul"
branch_labels = None
depends_on = None

_JSON_LIST = sa.text("'[]'::jsonb")
_JSON_OBJECT = sa.text("'{}'::jsonb")


def _catalog() -> None:
    # One entry per name: keep the newest version of each, fold NULL workspaces into "".
    op.execute(
        "DELETE FROM tools t USING tools newer "
        "WHERE t.tenant_id = newer.tenant_id AND t.name = newer.name "
        "AND coalesce(t.workspace_id, '') = coalesce(newer.workspace_id, '') "
        "AND (t.version, t.tool_id) < (newer.version, newer.tool_id)"
    )
    op.execute("UPDATE tools SET workspace_id = '' WHERE workspace_id IS NULL")
    op.alter_column("tools", "workspace_id", nullable=False, server_default="")
    op.drop_constraint("uq_tools_tenant_name_schema", "tools", type_="unique")
    op.drop_index("ix_tools_tenant_name", table_name="tools")
    for column in ("output_schema", "tags", "policy"):
        op.drop_column("tools", column)
    op.alter_column("tools", "source", type_=sa.String(50))
    op.add_column(
        "tools",
        sa.Column("required", postgresql.JSONB(), nullable=False, server_default=_JSON_LIST),
    )
    op.add_column(
        "tools",
        sa.Column(
            "argument_entity_types",
            postgresql.JSONB(),
            nullable=False,
            server_default=_JSON_OBJECT,
        ),
    )
    op.add_column("tools", sa.Column("side_effects", sa.String(20), nullable=True))
    op.add_column(
        "tools",
        sa.Column("examples", postgresql.JSONB(), nullable=False, server_default=_JSON_LIST),
    )
    op.add_column(
        "tools", sa.Column("redact", postgresql.JSONB(), nullable=False, server_default=_JSON_LIST)
    )
    op.create_unique_constraint(
        "uq_tools_catalog_name", "tools", ["tenant_id", "workspace_id", "name"]
    )


def _counters() -> None:
    op.create_table(
        "tool_stats",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("tool_name", sa.String(200), primary_key=True),
        sa.Column("calls", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("successes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("latency_ms_total", sa.Float(), nullable=False, server_default="0"),
        sa.Column("latency_calls", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("approvals", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("rejections", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("edits", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "approval_patterns",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("agent_id", sa.String(200), primary_key=True),
        sa.Column("tool_name", sa.String(200), primary_key=True),
        sa.Column("arg_shape", sa.String(1000), primary_key=True),
        sa.Column("approvals", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rejections", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("edits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def _procedures() -> None:
    op.create_table(
        "procedures",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("procedure_id", sa.String(200), primary_key=True),
        sa.Column("scope_key", sa.String(600), nullable=False),
        sa.Column("pattern", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("strategy", sa.Text(), nullable=False, server_default=""),
        sa.Column("steps", postgresql.JSONB(), nullable=False, server_default=_JSON_LIST),
        sa.Column("bindings", postgresql.JSONB(), nullable=False, server_default=_JSON_LIST),
        sa.Column("success_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("support", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="candidate"),
        sa.Column("steps_hash", sa.String(64), nullable=False, server_default=""),
        sa.Column("distilled", sa.String(64), nullable=False, server_default=""),
        sa.Column("owner_principal", sa.String(512), nullable=True),
        sa.Column("workspace_id", sa.String(200), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("tenant_id", "scope_key", "pattern", name="uq_procedures_pattern"),
    )
    op.create_index(
        "ix_procedures_scope", "procedures", ["tenant_id", "scope_key", "status", "updated_at"]
    )


def _profile_and_summaries() -> None:
    op.create_table(
        "profile_blocks",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("scope_key", sa.String(600), primary_key=True),
        sa.Column("block", sa.String(60), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False, server_default=""),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source", sa.String(20), nullable=False, server_default="learned"),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_table(
        "thread_summaries",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("thread_id", sa.String(200), primary_key=True),
        sa.Column("version", sa.Integer(), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("covers_to_sequence", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def upgrade() -> None:
    _profile_and_summaries()
    _catalog()
    _counters()
    _procedures()
    op.drop_index("ix_tool_invocations_unindexed", table_name="tool_invocations")
    op.alter_column("tool_invocations", "indexed_at", new_column_name="learned_at")
    op.create_index(
        "ix_tool_invocations_unlearned",
        "tool_invocations",
        ["occurred_at"],
        postgresql_where=sa.text("learned_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_table("thread_summaries")
    op.drop_table("profile_blocks")
    op.drop_index("ix_tool_invocations_unlearned", table_name="tool_invocations")
    op.alter_column("tool_invocations", "learned_at", new_column_name="indexed_at")
    op.create_index(
        "ix_tool_invocations_unindexed", "tool_invocations", ["tenant_id", "indexed_at"]
    )
    op.drop_index("ix_procedures_scope", table_name="procedures")
    op.drop_table("procedures")
    op.drop_table("approval_patterns")
    op.drop_table("tool_stats")
    op.drop_constraint("uq_tools_catalog_name", "tools", type_="unique")
    for column in ("redact", "examples", "side_effects", "argument_entity_types", "required"):
        op.drop_column("tools", column)
    op.alter_column("tools", "source", type_=sa.String(20))
    op.add_column(
        "tools", sa.Column("policy", postgresql.JSONB(), nullable=False, server_default="{}")
    )
    op.add_column(
        "tools", sa.Column("tags", postgresql.JSONB(), nullable=False, server_default="[]")
    )
    op.add_column("tools", sa.Column("output_schema", postgresql.JSONB(), nullable=True))
    op.alter_column("tools", "workspace_id", nullable=True, server_default=None)
    op.create_index("ix_tools_tenant_name", "tools", ["tenant_id", "name"])
    op.create_unique_constraint(
        "uq_tools_tenant_name_schema", "tools", ["tenant_id", "name", "schema_hash"]
    )
