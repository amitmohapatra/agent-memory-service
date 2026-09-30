"""The query_decomposition model use is removed (docs/MEASUREMENTS.md, section 8): no stored
tenant policy may keep naming it."""

from alembic import op

revision = "0020_drop_query_decomposition"
down_revision = "0019_chunk_index_fingerprint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE llm_policies SET uses = array_remove(uses, 'query_decomposition')")


def downgrade() -> None:
    """Nothing to restore: a policy that named the use named it for a read path that no
    longer exists."""
