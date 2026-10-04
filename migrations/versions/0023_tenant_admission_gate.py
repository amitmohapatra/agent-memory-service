"""The admission gate becomes a tenant's choice (ADR 0032).

``modules/memory/admission.py`` was built and never wired. It is wired now, behind this
per-tenant switch, off for every tenant: turning it on changes which extracted candidates are
stored, and the retrieval gates were measured with every candidate kept.
"""

import sqlalchemy as sa
from alembic import op

revision = "0023_tenant_admission_gate"
down_revision = "0022_feedback_review"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("admission_gate", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("tenants", "admission_gate")
