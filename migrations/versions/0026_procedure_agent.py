"""An agent learns from all its users: procedures carry the agent and who produced them.

Learned skills (docs/api/skills.md): an agent's own tool calls, whichever user it ran for, are
learned under one audience (``agent:<tenant>/<agent>``) instead of one per (user, agent) pair,
and reach the agent's other users once two users produced them (``users``, ``sole_user``). The
skill-draft decision column of 0025 goes: a learned skill is offered on its own, and dismissing
it is the ``rejected`` status procedures already had.

The agent's calls are marked unlearned and their per-user procedures dropped, so the learning
job re-mines them under the agent's audience on its next run (records are kept; only what was
derived from them is rebuilt).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0026_procedure_agent"
down_revision = "0025_procedure_skill"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("procedures", sa.Column("agent_id", sa.String(200), nullable=True))
    op.add_column(
        "procedures", sa.Column("users", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column("procedures", sa.Column("sole_user", sa.String(200), nullable=True))
    op.drop_column("procedures", "skill")
    op.create_index("ix_procedures_agent", "procedures", ["tenant_id", "agent_id", "support"])
    # an agent's PRIVATE records were learned per (user, agent): re-learn them per agent
    op.execute("DELETE FROM procedures WHERE scope_key LIKE 'principal:%/agent:%'")
    op.execute("UPDATE tool_invocations SET learned_at = NULL WHERE agent_id IS NOT NULL")


def downgrade() -> None:
    op.drop_index("ix_procedures_agent", table_name="procedures")
    op.execute("DELETE FROM procedures WHERE agent_id IS NOT NULL")
    op.drop_column("procedures", "sole_user")
    op.drop_column("procedures", "users")
    op.drop_column("procedures", "agent_id")
    op.add_column("procedures", sa.Column("skill", postgresql.JSONB(), nullable=True))
