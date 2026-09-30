"""One encrypted key or revocation tombstone per tenant and owner principal."""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import AgentCredentialRow
from memory_service.ports.credentials import ModelIdentity, StoredCredential


def _record(row: AgentCredentialRow) -> StoredCredential:
    return StoredCredential(
        ModelIdentity(row.tenant_id, row.principal_id),
        row.key_id,
        row.ciphertext,
        row.revision,
        row.updated_at,
    )


class SqlCredentialRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, identity: ModelIdentity) -> StoredCredential | None:
        row = await self.session.get(
            AgentCredentialRow, (identity.tenant_id, identity.principal_id)
        )
        return _record(row) if row is not None else None

    async def first(self, levels: Sequence[ModelIdentity]) -> StoredCredential | None:
        if not levels:
            return None
        order = [level.principal_id for level in levels]
        rows = (
            await self.session.scalars(
                select(AgentCredentialRow).where(
                    AgentCredentialRow.tenant_id == levels[0].tenant_id,
                    AgentCredentialRow.principal_id.in_(order),
                )
            )
        ).all()
        by_level = {row.principal_id: row for row in rows}
        found = next((by_level[p] for p in order if p in by_level), None)
        return _record(found) if found is not None else None

    async def tenants_with_keys(self) -> list[str]:
        return list(
            (
                await self.session.scalars(
                    select(AgentCredentialRow.tenant_id)
                    .where(AgentCredentialRow.ciphertext.is_not(None))
                    .distinct()
                    .order_by(AgentCredentialRow.tenant_id)
                )
            ).all()
        )

    async def put(
        self, identity: ModelIdentity, *, key_id: str, ciphertext: bytes | None
    ) -> StoredCredential:
        statement = insert(AgentCredentialRow).values(
            tenant_id=identity.tenant_id,
            principal_id=identity.principal_id,
            key_id=key_id,
            ciphertext=ciphertext,
            revision=1,
            updated_at=datetime.now(UTC),
        )
        statement = statement.on_conflict_do_update(
            index_elements=["tenant_id", "principal_id"],
            set_={
                "key_id": statement.excluded.key_id,
                "ciphertext": statement.excluded.ciphertext,
                "revision": AgentCredentialRow.revision + 1,
                "updated_at": statement.excluded.updated_at,
            },
        ).returning(AgentCredentialRow)
        row = (
            await self.session.scalars(statement, execution_options={"populate_existing": True})
        ).one()
        return _record(row)
