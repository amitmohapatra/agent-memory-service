"""A line the profile job appends to an edited ``user`` block records the memory it came from.

``profile_blocks.appended_from`` maps each such line to its memory, so the line leaves the
block when that memory is forgotten, replaced or archived, as a learned block's line does
(ADR 0032: derived data follows its source), while the editor's own lines stay.

A block written before keeps an empty map: which of its lines the job appended was never
recorded, so every line of an edited block reads as the editor's, as it did until now.
Downgrade drops the column, and with it the record of which lines the job appended.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0040_profile_line_sources"
down_revision = "0039_redact_stored_decisions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "profile_blocks",
        sa.Column("appended_from", JSONB(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("profile_blocks", "appended_from")
