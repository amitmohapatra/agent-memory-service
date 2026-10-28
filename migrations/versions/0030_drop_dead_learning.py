"""Two learning features that never fed anything are removed with what they kept.

- Prefetch: the context pre-included memories an agent's earlier tool calls (``agent_tool_pulls``,
  every memory agent-tool call) kept using for requests of the same pattern, from counts the
  ``periodic.prefetch_learn`` job folded into ``prefetch_stats``. Pulls were keyed by the model's
  own query wording, which differs every run, so no item ever reached the threshold: both
  tables go, and nothing else read the pull log.
- Strategy distillation: the tool-learning job wrote a procedure's ``title``, ``strategy`` and
  ``distilled`` (with the ``procedure_abstraction`` model use, under ``owner_principal``'s key);
  no read path showed them. The four columns go, and no stored tenant policy may keep naming
  the use (as 0020 did for ``query_decomposition``): it leaves the policy's uses and models, and
  its usage rows, which the usage read could no longer name, are dropped.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0030_drop_dead_learning"
down_revision = "0029_message_tool_calls"
branch_labels = None
depends_on = None

_JSON_LIST = sa.text("'[]'::jsonb")
_JSON_OBJECT = sa.text("'{}'::jsonb")


def upgrade() -> None:
    op.drop_table("prefetch_stats")
    op.drop_table("agent_tool_pulls")
    for column in ("title", "strategy", "distilled", "owner_principal"):
        op.drop_column("procedures", column)
    op.execute(
        "UPDATE llm_policies SET uses = array_remove(uses, 'procedure_abstraction'), "
        "models = models - 'procedure_abstraction'"
    )
    op.execute("DELETE FROM llm_usage_daily WHERE use = 'procedure_abstraction'")


def downgrade() -> None:
    """The previous schema, with the removed features' tables empty and their columns at
    their defaults (their rows are gone)."""
    now = sa.text("now()")
    op.add_column("procedures", sa.Column("owner_principal", sa.String(512), nullable=True))
    op.add_column(
        "procedures", sa.Column("distilled", sa.String(64), nullable=False, server_default="")
    )
    op.add_column("procedures", sa.Column("strategy", sa.Text(), nullable=False, server_default=""))
    op.add_column("procedures", sa.Column("title", sa.Text(), nullable=False, server_default=""))
    op.create_table(
        "agent_tool_pulls",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("pull_id", sa.String(200), primary_key=True),
        sa.Column("scope_key", sa.String(600), nullable=False),
        sa.Column("run_id", sa.String(200), nullable=True),
        sa.Column("pattern", sa.Text(), nullable=False, server_default=""),
        sa.Column("tool", sa.String(60), nullable=False),
        sa.Column("args", postgresql.JSONB(), nullable=False, server_default=_JSON_OBJECT),
        sa.Column("result_ids", postgresql.JSONB(), nullable=False, server_default=_JSON_LIST),
        sa.Column("used_ids", postgresql.JSONB(), nullable=False, server_default=_JSON_LIST),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
        sa.Column("learned_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_agent_tool_pulls_run", "agent_tool_pulls", ["tenant_id", "run_id"])
    op.create_index(
        "ix_agent_tool_pulls_unlearned",
        "agent_tool_pulls",
        ["created_at"],
        postgresql_where=sa.text("learned_at IS NULL"),
    )
    op.create_table(
        "prefetch_stats",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("scope_key", sa.String(600), primary_key=True),
        sa.Column("pattern", sa.Text(), primary_key=True),
        sa.Column("item_id", sa.String(200), primary_key=True),
        sa.Column("pulls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("uses", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
    )
    op.create_index(
        "ix_prefetch_stats_pattern",
        "prefetch_stats",
        ["tenant_id", "scope_key", "pattern", "uses"],
    )
