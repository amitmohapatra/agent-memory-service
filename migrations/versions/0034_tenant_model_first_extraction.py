"""Model-first extraction becomes a tenant's choice (ADR 0045).

Off for every tenant: turning it on changes which memories a message yields.
"""

import sqlalchemy as sa
from alembic import op

revision = "0034_tenant_model_first"
down_revision = "0033_drop_agent_runs_turn_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column(
            "model_first_extraction", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )


def downgrade() -> None:
    op.drop_column("tenants", "model_first_extraction")
