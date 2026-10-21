"""Learned approval rules are accepted by the service on its own (ADR 0042).

``approval_patterns`` records the rule the service accepted from a pattern's decisions
(``auto_accepted``: ``auto_approve`` or ``always_ask``) and when. It is written once: a rule a
person revoked through the catalog is not accepted again on its own, and the listing of the
rules the service accepted (``GET /v1/tools/approval-suggestions/auto-accepted``) reads it,
newest first (``ix_approval_patterns_auto_accepted``, partial on the rows that carry one).
"""

import sqlalchemy as sa
from alembic import op

revision = "0029_approval_auto_accepted"
down_revision = "0028_user_statement_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("approval_patterns", sa.Column("auto_accepted", sa.String(20), nullable=True))
    op.add_column(
        "approval_patterns",
        sa.Column("auto_accepted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_approval_patterns_auto_accepted",
        "approval_patterns",
        ["tenant_id", sa.text("auto_accepted_at DESC")],
        postgresql_where=sa.text("auto_accepted_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_approval_patterns_auto_accepted", table_name="approval_patterns")
    op.drop_column("approval_patterns", "auto_accepted_at")
    op.drop_column("approval_patterns", "auto_accepted")
