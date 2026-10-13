"""A replaced vote names its replacement; subjects are looked up whatever their case; review
reports read an index.

``feedback.review_replaced_by`` (ADR 0039): a vote dismissed because its voter voted again
on the same target names the newer vote. It is not a review state of its own, so a client
that knows pending, approved and dismissed reads every record; ``author_record`` and the
grouped queue leave these out (an author changing their mind is not a verdict on them).

``ix_memories_scope_subject_lower``: consolidation's candidate read matches a subject's
spellings case-insensitively (``lower(subject)`` against ``lower(:spelling)``, both folded by
PostgreSQL), so "Forklift 4" stored through the API is found by "forklift-4" said in a turn.

``feedback.is_vote``: whether a record is a person's vote, decided when it is stored by who
sent it - a client may claim any ``source`` - so the one-vote rule cannot be dodged by
writing ``source=judge`` on a thumb. Null on older rows, which are classified by source.

``ix_feedback_review`` and ``ix_feedback_review_target``: ``GET /v1/feedback?review=<state>``
(newest first) and its ``group=target`` form read one review state of one tenant, not the
tenant's whole history (``ix_feedback_pending`` served the pending queue only).

Every index is built like 0024: ``CREATE INDEX CONCURRENTLY`` in an autocommit block, ``IF
NOT EXISTS``, an INVALID leftover dropped first. On a large ``memories`` table the subject
index may need more than the migration session's 15-minute ``statement_timeout``
(docs/deploy/database.md): raise ``MEMORY_MIGRATION_STATEMENT_TIMEOUT`` for that run.
"""

import sqlalchemy as sa
from alembic import op

#: the file's name less "_lower": alembic_version.version_num holds at most 32 characters
revision = "0027_votes_replaced_by_subject"
down_revision = "0026_procedure_agent"
branch_labels = None
depends_on = None

INDEXES = {
    "ix_memories_scope_subject_lower": (
        "memories (tenant_id, scope_key, lower(subject)) WHERE deleted_at IS NULL"
    ),
    "ix_feedback_review": (
        "feedback (tenant_id, review_state, created_at DESC, feedback_id DESC) "
        "WHERE review_state IS NOT NULL"
    ),
    "ix_feedback_review_target": (
        "feedback (tenant_id, review_state, target_kind, target_id) "
        "INCLUDE (created_at, verdict, user_id, agent_id) "
        "WHERE review_state IS NOT NULL AND review_replaced_by IS NULL"
    ),
}


def upgrade() -> None:
    op.add_column("feedback", sa.Column("review_replaced_by", sa.String(200), nullable=True))
    op.add_column("feedback", sa.Column("is_vote", sa.Boolean(), nullable=True))
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
            op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {definition}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name in INDEXES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
    op.drop_column("feedback", "is_vote")
    op.drop_column("feedback", "review_replaced_by")
