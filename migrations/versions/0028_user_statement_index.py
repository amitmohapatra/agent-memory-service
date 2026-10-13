"""Two indexes for cross-conversation replacement (ADR 0038), built without blocking writes.

``SqlMemoryRepository.statements_of`` runs for every user statement the pipeline stores: the
user's live, CURRENT, ``USER``-visible memories, newest first, up to 64. No index led with the
owner, so it read every live row of the tenant. ``ix_memories_owner_statements`` is that range
in ``observed_at DESC`` order, partial on the query's predicate.

``replaced_by_any`` runs when a statement is forgotten or retracted, to reinstate what it
replaced: ``ix_memories_replaced_by`` is the newer statement's id inside
``system_metadata->'replaced_by'``, partial on the rows that carry one (few).

Online like migration 0024: ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` in an autocommit
block, an INVALID index from a failed build dropped first.
"""

from alembic import op

revision = "0028_user_statement_index"
down_revision = "0027_votes_replaced_by_subject"
branch_labels = None
depends_on = None

INDEXES = {
    "ix_memories_owner_statements": (
        "(tenant_id, owner_principal, observed_at DESC) "
        "WHERE deleted_at IS NULL AND temporal_status = 'CURRENT' AND visibility = 'USER'"
    ),
    "ix_memories_replaced_by": (
        "(tenant_id, ((system_metadata -> 'replaced_by') ->> 'memory_id')) "
        "WHERE system_metadata ? 'replaced_by'"
    ),
}


def upgrade() -> None:
    with op.get_context().autocommit_block():
        for name, definition in INDEXES.items():
            op.execute(
                f"""
                DO $$
                BEGIN
                  IF EXISTS (
                    SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                    WHERE c.relname = '{name}' AND NOT i.indisvalid
                  ) THEN
                    EXECUTE 'DROP INDEX {name}';
                  END IF;
                END $$
                """
            )
            op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON memories {definition}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name in INDEXES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
