"""Drop graph_entity_aliases: a table nothing ever read or wrote.

Migration 0006 created it for cross-document entity resolution, and the resolution that
shipped keeps aliases in the ``aliases`` JSONB column on ``graph_entities`` instead - which
is what the store reads, merges and now seeds principal nodes with. The table was never
queried from any code path: no repository, no service, no adapter. A schema that carries an
object no code knows about is a claim that something uses it.

The downgrade recreates it exactly as 0006 did, so this is reversible; there is no data to
preserve because nothing ever inserted any.

Revision ID: 0008_drop_entity_alias
Revises: 0007_graph_alias
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_drop_entity_alias"
down_revision = "0007_graph_alias"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_graph_entity_aliases_alias", table_name="graph_entity_aliases")
    op.drop_table("graph_entity_aliases")


def downgrade() -> None:
    op.create_table(
        "graph_entity_aliases",
        sa.Column("alias_id", sa.String(length=200), nullable=False),
        sa.Column("tenant_id", sa.String(length=200), nullable=False),
        sa.Column("alias", sa.String(length=300), nullable=False),
        sa.Column("entity_id", sa.String(length=200), nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("source", sa.String(length=40), server_default="canonical", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("alias_id"),
        sa.UniqueConstraint("tenant_id", "alias", "entity_id", name="uq_graph_entity_alias"),
    )
    op.create_index(
        "ix_graph_entity_aliases_alias",
        "graph_entity_aliases",
        ["tenant_id", "alias"],
        unique=False,
    )
