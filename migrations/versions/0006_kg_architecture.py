"""kg_architecture: relation layers, invalidation instead of deletion, entity summaries and
the tenant-wide alias table

Revision ID: 0006_kg_arch
Revises: 0005
Create Date: 2026-09-15 12:00:00.000000+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = '0006_kg_arch'
down_revision = '0005'
branch_labels = None
depends_on = None

_CAUSAL = (
    "driven_by", "caused_by", "causes", "leads_to", "results_in", "due_to", "reflects",
    "offset_by", "impacted_by", "attributable_to", "supported_by", "helped_by", "hurt_by",
    "enables", "contributed_to", "triggered_by", "depends_on", "because_of",
)
_TEMPORAL = (
    "valid_from", "valid_to", "as_of", "supersedes", "superseded_by", "invalidated_by",
    "closed_on", "founded_in", "adopted_in", "occurred_on", "announced_on", "effective_from",
    "started_on", "ended_on", "dated", "scheduled_for", "expires_on", "preceded_by",
    "followed_by",
)
_STRUCTURAL = (
    "mentions", "mentioned_in", "co_occurs_with", "discusses", "defined_in", "refers_to",
    "appears_in", "cites", "contained_in", "links",
)


def _in(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def upgrade() -> None:
    op.add_column(
        'graph_relations',
        sa.Column('layer', sa.String(length=20), server_default='entity', nullable=False),
    )
    op.add_column(
        'graph_relations', sa.Column('invalidated_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.execute(
        f"UPDATE graph_relations SET layer = CASE "
        f"WHEN predicate IN ({_in(_CAUSAL)}) THEN 'causal' "
        f"WHEN predicate IN ({_in(_TEMPORAL)}) THEN 'temporal' "
        f"WHEN predicate IN ({_in(_STRUCTURAL)}) THEN 'structural' "
        f"ELSE 'entity' END"
    )
    op.execute(
        "UPDATE graph_relations SET invalidated_at = valid_to "
        "WHERE status <> 'CURRENT' AND invalidated_at IS NULL AND valid_to IS NOT NULL"
    )
    op.create_index(
        'ix_graph_relations_layer', 'graph_relations', ['tenant_id', 'layer', 'status'], unique=False
    )
    op.add_column(
        'graph_entities', sa.Column('summary', sa.Text(), server_default='', nullable=False)
    )
    op.create_table(
        'graph_entity_aliases',
        sa.Column('alias_id', sa.String(length=200), nullable=False),
        sa.Column('tenant_id', sa.String(length=200), nullable=False),
        sa.Column('alias', sa.String(length=300), nullable=False),
        sa.Column('entity_id', sa.String(length=200), nullable=False),
        sa.Column('confidence', sa.Float(), server_default='1.0', nullable=False),
        sa.Column('source', sa.String(length=40), server_default='canonical', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('alias_id'),
        sa.UniqueConstraint('tenant_id', 'alias', 'entity_id', name='uq_graph_entity_alias'),
    )
    op.create_index(
        'ix_graph_entity_aliases_alias', 'graph_entity_aliases', ['tenant_id', 'alias'], unique=False
    )
    op.create_index(
        'ix_graph_entity_aliases_entity', 'graph_entity_aliases', ['tenant_id', 'entity_id'], unique=False
    )
    op.execute(
        "INSERT INTO graph_entity_aliases (alias_id, tenant_id, alias, entity_id, confidence, source) "
        "SELECT 'als_' || md5(tenant_id || chr(31) || canonical_name || chr(31) || entity_id), "
        "tenant_id, canonical_name, entity_id, 1.0, 'canonical' FROM graph_entities "
        "ON CONFLICT DO NOTHING"
    )


def downgrade() -> None:
    op.drop_index('ix_graph_entity_aliases_entity', table_name='graph_entity_aliases')
    op.drop_index('ix_graph_entity_aliases_alias', table_name='graph_entity_aliases')
    op.drop_table('graph_entity_aliases')
    op.drop_column('graph_entities', 'summary')
    op.drop_index('ix_graph_relations_layer', table_name='graph_relations')
    op.drop_column('graph_relations', 'invalidated_at')
    op.drop_column('graph_relations', 'layer')
