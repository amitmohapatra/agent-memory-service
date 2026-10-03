"""Feedback review (ADR 0028): a verdict that would change what the platform learned waits
for a tenant admin.

``review_state`` is null for a record applied as it arrived (every existing row), else
``pending``, ``approved`` or ``dismissed``; who decided, when and why ride beside it. The
review queue is read newest first through a partial index on the pending rows alone.
"""

import sqlalchemy as sa
from alembic import op

revision = "0022_feedback_review"
down_revision = "0021_final_surface"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("feedback", sa.Column("review_state", sa.String(20), nullable=True))
    op.add_column("feedback", sa.Column("reviewed_by", sa.String(300), nullable=True))
    op.add_column("feedback", sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("feedback", sa.Column("review_note", sa.String(4000), nullable=True))
    op.create_index(
        "ix_feedback_pending",
        "feedback",
        ["tenant_id", sa.text("created_at DESC"), sa.text("feedback_id DESC")],
        postgresql_where=sa.text("review_state = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_pending", table_name="feedback")
    for column in ("review_note", "reviewed_at", "reviewed_by", "review_state"):
        op.drop_column("feedback", column)
