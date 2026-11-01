"""The index of a lookup nothing makes is dropped.

``ix_workspace_members_principal`` (``tenant_id``, ``principal``) served
``WorkspaceRepository.memberships_of``, the workspaces a principal was admitted to. Nothing in
the service ever called it: every membership read names its workspace, which the primary key
(``tenant_id``, ``workspace_id``, ``principal``) serves. The method is removed, and its index
with it, a write saved on every membership change.
"""

from alembic import op

revision = "0032_drop_memberships_index"
down_revision = "0031_endpoint_credentials"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_workspace_members_principal", table_name="workspace_members")


def downgrade() -> None:
    op.create_index(
        "ix_workspace_members_principal", "workspace_members", ["tenant_id", "principal"]
    )
