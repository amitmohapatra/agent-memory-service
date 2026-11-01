"""The index of a lookup nothing makes is dropped.

``ix_agent_runs_turn`` (``tenant_id``, ``turn_id``) served ``AgentRunRepository.list_for_turn``,
the agent runs of one turn. Nothing in the service called it (an end-to-end test did): agent
runs are read by their id, which the primary key serves. The method is removed, and its index
with it, a write saved on every agent run.
"""

from alembic import op

revision = "0033_drop_agent_runs_turn_index"
down_revision = "0032_drop_memberships_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_agent_runs_turn", table_name="agent_runs")


def downgrade() -> None:
    op.create_index("ix_agent_runs_turn", "agent_runs", ["tenant_id", "turn_id"])
