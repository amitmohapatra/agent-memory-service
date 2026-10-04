"""The consolidation candidate read gets its own index, built without blocking writes.

``SqlMemoryRepository.candidates`` runs for every candidate of every processed observation:
a scope's live, CURRENT rows newest first, twice (exact match, then recent). The index it had
(``tenant_id, scope_key, temporal_status``) found the scope and then sorted every live row of
it by ``updated_at`` to take twenty; a busy user's scope is thousands of rows. This one is
that range already in ``updated_at DESC`` order, partial on ``deleted_at IS NULL`` like the
query.

The first online migration (docs/deploy/database.md, ADR 0031): ``CREATE INDEX
CONCURRENTLY`` takes no lock that blocks writes, cannot run inside a transaction, so it runs
in an autocommit block, and is ``IF NOT EXISTS`` so a migration retried after a lock timeout
is harmless. A concurrent build that fails leaves an INVALID index behind; the upgrade drops
one first.

``list_idle`` (the forgetting sweep) needs nothing new: its cross-tenant scan is
``ix_memories_recent_updates`` (``updated_at, memory_id`` partial on CURRENT and live), the
same predicate and order.
"""

from alembic import op

revision = "0024_online_candidate_index"
down_revision = "0023_tenant_admission_gate"
branch_labels = None
depends_on = None

INDEX = "ix_memories_scope_candidates"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            f"""
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                WHERE c.relname = '{INDEX}' AND NOT i.indisvalid
              ) THEN
                EXECUTE 'DROP INDEX {INDEX}';
              END IF;
            END $$
            """
        )
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX} ON memories "
            "(tenant_id, scope_key, temporal_status, updated_at DESC) "
            "WHERE deleted_at IS NULL"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX}")
