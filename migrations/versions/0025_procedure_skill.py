"""A procedure remembers the reviewer's decision about its skill draft.

Learned skills (docs/api/tools.md#learned-skills): an active procedure is offered to the
tenant's administrator as a draft Agent Skill; publishing it (to ``SKILLS_DIR`` or the Bifrost
gateway's skills repository) or dismissing it is recorded here, with the steps it had, so the
draft comes back only when the steps change. A nullable column: no rewrite, no lock beyond the
brief one ``ADD COLUMN`` takes.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0025_procedure_skill"
down_revision = "0024_online_candidate_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("procedures", sa.Column("skill", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("procedures", "skill")
