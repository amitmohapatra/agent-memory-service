"""A thread's deletion asks whether another live thread still attaches a document.

``SqlDocumentRepository.release_thread`` marks deleted the documents a deleted thread held,
unless a live message of another live thread attaches them: ``message_attachments`` by
``document_id``, which no index led with. ``ix_message_attachments_document`` is that lookup,
partial on the attachments that name an ingested document.

Online like migration 0028: ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` in an autocommit
block, an INVALID index from a failed build dropped first.
"""

from alembic import op

revision = "0038_attachment_document_index"
down_revision = "0037_memory_audience_index"
branch_labels = None
depends_on = None

NAME = "ix_message_attachments_document"


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
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {NAME} ON message_attachments"
            " (document_id) WHERE document_id IS NOT NULL"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {NAME}")
