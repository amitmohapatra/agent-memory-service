"""What was read out of a person's words says so: ``memories.system_metadata.said_by_user``.

A memory read out of a user's turn that an agent harness relayed is owned by the agent
(``agent:<user>/<agent>``) and is no ``said`` turn, so nothing told it apart from the agent's
own note, and an agent could retract it without asking the person. From this version every
memory the pipeline or a statement writes records ``said_by_user``.

A row stored before carries no key and reads as before: not a person's words, unless its
owner or its ``said`` turn says so. This backfills the rows whose origin the store still
holds: an agent-owned memory read out of a conversation message a user sent (``role =
'USER'``, a ``VISIBLE`` message: the turns ``said_by_user`` reads as a person's own words) is
marked a person's words, so an agent asks before retracting it. A row read out of a
decision or a feedback record keeps no message to tell from, and stays as it was.

One statement over the agent-owned rows, each looked up by the message's primary key.
"""

from alembic import op

revision = "0036_said_by_user"
down_revision = "0035_beliefs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE memories AS m
        SET system_metadata = m.system_metadata || '{"said_by_user": true}'::jsonb
        WHERE m.owner_principal LIKE 'agent:%/%'
          AND NOT (m.system_metadata ? 'said_by_user')
          AND EXISTS (
            SELECT 1
            FROM jsonb_array_elements(m.evidence) AS ev
            JOIN messages AS msg
              ON msg.message_id = ev->>'message_id' AND msg.tenant_id = m.tenant_id
            WHERE ev->>'source_type' = 'message'
              AND msg.role = 'USER'
              AND msg.kind = 'VISIBLE'
          )
        """
    )


def downgrade() -> None:
    op.execute(
        "UPDATE memories SET system_metadata = system_metadata - 'said_by_user'"
        " WHERE system_metadata ? 'said_by_user'"
    )
