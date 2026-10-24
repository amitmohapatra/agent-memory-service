"""A message's tool calls, as the agent's framework said them.

An ASSISTANT message carries the calls it made (``tool_calls``: ``[{id, name, arguments}]``)
and a TOOL message the call it answers (``tool_call_id``), so the conversation window can give
a thread back the way a framework's own checkpoint replays it. Both nullable, no default: adding
them rewrites no row.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0029_message_tool_calls"
down_revision = "0028_user_statement_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("tool_calls", JSONB(), nullable=True))
    op.add_column("messages", sa.Column("tool_call_id", sa.String(200), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "tool_call_id")
    op.drop_column("messages", "tool_calls")
