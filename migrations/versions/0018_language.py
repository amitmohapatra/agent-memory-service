"""The language of every observation, memory and chunk (``domain.language``), stored at write.

Nullable: a row written before this revision has none, and its model derives it from the
text when it is read (the detector is deterministic, so both give the same answer). Nothing
filters on it in SQL, so it carries no index.
"""

import sqlalchemy as sa
from alembic import op

revision = "0018_language"
down_revision = "0017_learning"
branch_labels = None
depends_on = None

_TABLES = ("observations", "memories", "chunks")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("lang", sa.String(8), nullable=True))


def downgrade() -> None:
    for table in _TABLES:
        op.drop_column(table, "lang")
