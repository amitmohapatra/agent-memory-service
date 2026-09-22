"""graph_alias_index: the alias lookup every entity question makes gets an index

``GraphStore.find_entities`` matches a question's candidate names against
``graph_entities.canonical_name`` *or* ``graph_entities.aliases`` with the JSONB ``?|``
operator (adapters/graph/postgres_store.py:292). The name half has had a btree index since
0005 (``ix_graph_entities_tenant_name``); the alias half has had nothing, so every entity,
temporal and multi-hop question ended in a sequential scan of the tenant's entities - on the
query path, underneath a wall budget, and getting slower with every document ingested.

``jsonb_ops`` is what ``?|`` needs (``jsonb_path_ops`` does not support the key-existence
operators), which is also how 0005 built the ``visibility_keys`` GIN on this same table.

Revision ID: 0007_graph_alias
Revises: 0007_tool_memory
Create Date: 2026-09-23 02:10:00.000000+00:00
"""

from __future__ import annotations

from alembic import op

revision = "0007_graph_alias"
down_revision = "0007_tool_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_graph_entities_aliases",
        "graph_entities",
        ["aliases"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_graph_entities_aliases", table_name="graph_entities", postgresql_using="gin"
    )
