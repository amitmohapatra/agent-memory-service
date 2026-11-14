"""The user's inventory finds what was said in their conversations by its audience.

``GET /v1/memories`` lists "what do we hold about this user", and a statement said in a
conversation is anchored in its thread (ADR 0036) with the user as its audience
(``user:<tenant>/<user>`` in ``visibility_keys``). ``SqlMemoryRepository.list_scope`` finds
those by containment (``visibility_keys @> '["user:<t>/<u>"]'``) beside the anchors' range;
``ix_memories_visibility_keys`` is a GIN index on the live rows' audience (``jsonb_path_ops``:
containment only, the one operator the read uses).

Online like migration 0028: ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` in an autocommit
block, an INVALID index from a failed build dropped first.
"""

from alembic import op

revision = "0037_memory_audience_index"
down_revision = "0036_said_by_user"
branch_labels = None
depends_on = None

NAME = "ix_memories_visibility_keys"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            f"""
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                WHERE c.relname = '{NAME}' AND NOT i.indisvalid
              ) THEN
                EXECUTE 'DROP INDEX {NAME}';
              END IF;
            END $$
            """
        )
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {NAME} ON memories"
            " USING gin (visibility_keys jsonb_path_ops) WHERE deleted_at IS NULL"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {NAME}")
