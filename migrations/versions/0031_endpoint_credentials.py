"""A registered model credential is a whole endpoint now, not a bare key.

``agent_credentials.ciphertext`` sealed a Bifrost virtual key; it now seals a
``trellis.contracts.Endpoint`` as JSON (base URL, route, auth, model, key, profile). The column
is unchanged, but a key alone says nowhere to send it, and the envelope cannot be opened here,
so the rows that hold one are deleted: the harness registers its agent's endpoint again at its
next start, and a tenant administrator registers the tenant's again
(``PUT /v1/model-key``). Until then those calls go to the service's own endpoint (``LLM_*``),
or nowhere. Revocations (no ciphertext) are kept: a revoked agent stays refused. The tenants
that lose a row have their revision bumped, as a rotation would, so cached assisted output is
recomputed.
"""

from alembic import op

revision = "0031_endpoint_credentials"
down_revision = "0030_drop_dead_learning"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "INSERT INTO revisions (tenant_id, kind, object_id, value) "
        "SELECT DISTINCT tenant_id, 'tenant', '', 1 FROM agent_credentials "
        "WHERE ciphertext IS NOT NULL "
        "ON CONFLICT (tenant_id, kind, object_id) "
        "DO UPDATE SET value = revisions.value + 1, updated_at = now()"
    )
    op.execute("DELETE FROM agent_credentials WHERE ciphertext IS NOT NULL")


def downgrade() -> None:
    """Nothing to restore: the endpoints registered since are unreadable to the previous
    build (it read the sealed value as a bare key), so they go too, and the harness registers
    a key again at its next start."""
    op.execute("DELETE FROM agent_credentials WHERE ciphertext IS NOT NULL")
