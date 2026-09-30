"""chunks.index_fingerprint holds the ensemble's fingerprint.

With two dense spaces the index fingerprint is ~150 characters
(``dense_en=...+dense_ml=...|bm25-...``); the chunk column was 100 wide, so every document
index job failed on the receipt UPDATE after writing its vectors and was retried until it
gave up. memories.index_fingerprint was already 200.
"""

import sqlalchemy as sa
from alembic import op

revision = "0019_chunk_index_fingerprint"
down_revision = "0018_language"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "chunks", "index_fingerprint", type_=sa.String(200), existing_type=sa.String(100)
    )


def downgrade() -> None:
    op.alter_column(
        "chunks", "index_fingerprint", type_=sa.String(100), existing_type=sa.String(200)
    )
