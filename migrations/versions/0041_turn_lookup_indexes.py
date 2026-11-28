"""A found turn's neighbours, and the turn a model's reading was read from, by index.

``SqlMemoryRepository.turns_around`` and ``turns_of`` look turns kept as said up by the
message they keep (a search's neighbours; what a reading finds, ADR 0052) and by the turn
they reply to (``preceding_turn``). Both read inside JSON no index covered, so each lookup
read every live turn of the tenant.

- ``ix_memories_turn_messages``: GIN over the message and observation ids a live turn keeps
  (``evidence[*].message_id`` and ``evidence[*].source_id``); a turn can keep more than one.
- ``ix_memories_turn_replies``: the turn a live turn replies to
  (``system_metadata.preceding_turn.source_id``), with the tenant.

Online like migration 0028: ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` in an autocommit
block, an INVALID index from a failed build dropped first.
"""

from alembic import op

revision = "0041_turn_lookup_indexes"
down_revision = "0040_profile_line_sources"
branch_labels = None
depends_on = None

#: What a live turn kept as said is; the lookups name the same predicate.
TURNS = "deleted_at IS NULL AND (system_metadata ->> 'category') = 'verbatim_turn'"
INDEXES = {
    "ix_memories_turn_messages": (
        "USING gin ((jsonb_path_query_array(evidence, '$[*].message_id')"
        " || jsonb_path_query_array(evidence, '$[*].source_id')))"
    ),
    "ix_memories_turn_replies": (
        "(tenant_id, ((system_metadata -> 'preceding_turn') ->> 'source_id'))"
    ),
}


def upgrade() -> None:
    with op.get_context().autocommit_block():
        for name, spec in INDEXES.items():
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
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON memories {spec} WHERE {TURNS}"
            )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name in INDEXES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
