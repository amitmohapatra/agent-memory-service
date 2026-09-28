"""Tenants, API keys, workspaces, groups and the read audit; no source corpus mutation."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0014_tenancy"
down_revision = "0013_standing_briefs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    now = sa.text("now()")
    op.create_table(
        "tenants",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("retention_days", sa.Integer(), nullable=True),
        sa.Column("rate_limit_per_minute", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
    )
    op.create_table(
        "api_keys",
        sa.Column("key_id", sa.String(32), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("workspace_id", sa.String(200), nullable=True),
        sa.Column("secret_hash", sa.String(64), nullable=False),
        sa.Column("created_by", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_api_keys_tenant", "api_keys", ["tenant_id"])
    op.create_index(
        "ix_api_keys_live", "api_keys", ["key_id"], postgresql_where=sa.text("revoked_at IS NULL")
    )
    op.create_table(
        "workspaces",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("workspace_id", sa.String(200), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "workspace_members",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("workspace_id", sa.String(200), primary_key=True),
        sa.Column("principal", sa.String(512), primary_key=True),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("added_by", sa.String(512), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
    )
    op.create_index(
        "ix_workspace_members_principal", "workspace_members", ["tenant_id", "principal"]
    )
    op.create_table(
        "user_groups",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("group_id", sa.String(200), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "user_group_members",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("group_id", sa.String(200), primary_key=True),
        sa.Column("user_id", sa.String(200), primary_key=True),
        sa.Column("added_by", sa.String(512), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=now),
    )
    op.create_table(
        "memory_reads",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("credential", sa.String(512), nullable=False, server_default=""),
        sa.Column("principal", sa.String(512), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=False),
        sa.Column("scope_fingerprint", sa.String(64), nullable=False),
        sa.Column(
            "record_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False, server_default=now),
    )
    op.create_index("ix_memory_reads_tenant_at", "memory_reads", ["tenant_id", "at"])
    op.create_index("ix_memory_reads_at", "memory_reads", ["at"])
    # The retention sweep reads a tenant's live memories oldest first; without this every
    # run scanned the tenant. Partial, so it costs deleted rows nothing.
    op.create_index(
        "ix_memories_tenant_created_live",
        "memories",
        ["tenant_id", "created_at"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_memories_tenant_created_live", table_name="memories")
    op.drop_index("ix_memory_reads_at", table_name="memory_reads")
    op.drop_index("ix_memory_reads_tenant_at", table_name="memory_reads")
    op.drop_table("memory_reads")
    op.drop_table("user_group_members")
    op.drop_table("user_groups")
    op.drop_index("ix_workspace_members_principal", table_name="workspace_members")
    op.drop_table("workspace_members")
    op.drop_table("workspaces")
    op.drop_index("ix_api_keys_live", table_name="api_keys")
    op.drop_index("ix_api_keys_tenant", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_table("tenants")
