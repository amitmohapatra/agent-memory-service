"""m14_memory_architecture: access tracking for the forgetting policy

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-15 12:00:00.000000+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'memories',
        sa.Column('access_count', sa.Integer(), server_default='0', nullable=False),
    )
    op.add_column('memories', sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        'ix_memories_forgetting',
        'memories',
        ['tenant_id', 'updated_at'],
        unique=False,
        postgresql_where=sa.text("temporal_status = 'CURRENT' AND deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        'ix_memories_forgetting',
        table_name='memories',
        postgresql_where=sa.text("temporal_status = 'CURRENT' AND deleted_at IS NULL"),
    )
    op.drop_column('memories', 'last_accessed_at')
    op.drop_column('memories', 'access_count')
