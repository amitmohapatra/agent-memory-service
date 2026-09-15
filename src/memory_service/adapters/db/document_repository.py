"""PostgreSQL DocumentRepository."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import (
    ChunkRow,
    ContextEdgeRow,
    DocumentNodeRow,
    DocumentRow,
    DocumentVersionRow,
    FileStagingRow,
)
from memory_service.domain.documents import (
    Chunk,
    ContextEdge,
    Document,
    DocumentNode,
    DocumentVersion,
)
from memory_service.domain.enums import ArchiveStatus, ContextGraphEdge, Representation


def _doc(r: DocumentRow) -> Document:
    return Document(
        document_id=r.document_id,
        tenant_id=r.tenant_id,
        workspace_id=r.workspace_id,
        owner_user_id=r.owner_user_id,
        thread_id=r.thread_id,
        title=r.title,
        filename=r.filename,
        media_type=r.media_type,
        size_bytes=r.size_bytes,
        checksum=r.checksum,
        current_version_id=r.current_version_id,
        source_system=r.source_system,
        source_id=r.source_id,
        archive_status=ArchiveStatus(r.archive_status),
        system_metadata={
            **(r.system_metadata or {}),
            "status": r.status,
            "message_id": r.message_id,
            "last_error": r.last_error,
        },
        custom_metadata=r.custom_metadata or {},
        revision=r.revision,
        created_at=r.created_at,
        updated_at=r.updated_at,
        deleted_at=r.deleted_at,
    )


def _node(r: DocumentNodeRow) -> DocumentNode:
    return DocumentNode(
        node_id=r.node_id,
        document_id=r.document_id,
        document_version_id=r.document_version_id,
        tenant_id=r.tenant_id,
        representation=Representation(r.representation),
        parent_id=r.parent_id,
        ordinal=r.ordinal,
        depth=r.depth,
        title=r.title,
        section_path=r.section_path,
        page_start=r.page_start,
        page_end=r.page_end,
        text=r.text,
        text_hash=r.text_hash,
        token_estimate=r.token_estimate,
        entities=list(r.entities or []),
        system_metadata=r.system_metadata or {},
    )


def _chunk(r: ChunkRow) -> Chunk:
    return Chunk(
        chunk_id=r.chunk_id,
        node_id=r.node_id,
        document_id=r.document_id,
        document_version_id=r.document_version_id,
        tenant_id=r.tenant_id,
        ordinal=r.ordinal,
        text=r.text,
        text_hash=r.text_hash,
        contextual_text=r.contextual_text,
        page=r.page,
        section_path=r.section_path,
        token_estimate=r.token_estimate,
        entities=list(r.entities or []),
        indexed_at=r.indexed_at,
        index_fingerprint=r.index_fingerprint,
    )


def _edge(r: ContextEdgeRow) -> ContextEdge:
    return ContextEdge(
        tenant_id=r.tenant_id,
        document_id=r.document_id,
        source_id=r.source_id,
        target_id=r.target_id,
        edge=ContextGraphEdge(r.edge),
        weight=r.weight,
        label=r.label,
    )


class SqlDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(
        self, document: Document, *, visibility_keys: Sequence[str], message_id: str | None = None
    ) -> None:
        self.s.add(
            DocumentRow(
                document_id=document.document_id,
                tenant_id=document.tenant_id,
                workspace_id=document.workspace_id,
                owner_user_id=document.owner_user_id,
                thread_id=document.thread_id,
                message_id=message_id,
                title=document.title,
                filename=document.filename,
                media_type=document.media_type,
                size_bytes=document.size_bytes,
                checksum=document.checksum,
                status="STAGED",
                source_system=document.source_system,
                source_id=document.source_id,
                visibility_keys=list(visibility_keys),
                system_metadata=document.system_metadata,
                custom_metadata=document.custom_metadata,
            )
        )
        await self.s.flush()

    async def get(self, tenant_id: str, document_id: str) -> Document | None:
        r = await self.s.get(DocumentRow, document_id)
        if r is None or r.tenant_id != tenant_id or r.deleted_at is not None:
            return None
        return _doc(r)

    async def find_by_checksum(self, tenant_id: str, checksum: str) -> Document | None:
        r = await self.s.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.checksum == checksum,
                DocumentRow.deleted_at.is_(None),
            )
            .order_by(DocumentRow.created_at)
            .limit(1)
        )
        return _doc(r) if r is not None else None

    async def set_status(
        self,
        tenant_id: str,
        document_id: str,
        *,
        status: str,
        current_version_id: str | None = None,
        error: str | None = None,
    ) -> None:
        values = {
            "status": status,
            "updated_at": func.now(),
            "revision": DocumentRow.revision + 1,
            "last_error": error,
        }
        if current_version_id is not None:
            values["current_version_id"] = current_version_id
        await self.s.execute(
            update(DocumentRow)
            .where(DocumentRow.document_id == document_id, DocumentRow.tenant_id == tenant_id)
            .values(**values)
        )

    async def stage_bytes(
        self, tenant_id: str, document_id: str, data: bytes, *, checksum: str
    ) -> None:
        self.s.add(
            FileStagingRow(
                document_id=document_id,
                tenant_id=tenant_id,
                data=data,
                size_bytes=len(data),
                checksum=checksum,
            )
        )
        await self.s.flush()

    async def staged_bytes(self, tenant_id: str, document_id: str) -> bytes | None:
        r = await self.s.get(FileStagingRow, document_id)
        if r is None or r.tenant_id != tenant_id:
            return None
        return bytes(r.data) if r.data is not None else None

    async def purge_staged_bytes(self, tenant_id: str, document_id: str) -> None:
        await self.s.execute(
            update(FileStagingRow)
            .where(FileStagingRow.document_id == document_id, FileStagingRow.tenant_id == tenant_id)
            .values(data=None, purged_at=func.now())
        )

    async def mark_archived(
        self, tenant_id: str, document_id: str, *, segment_id: str, archived_at: datetime
    ) -> None:
        await self.s.execute(
            update(DocumentRow)
            .where(DocumentRow.document_id == document_id, DocumentRow.tenant_id == tenant_id)
            .values(
                archive_status=ArchiveStatus.ARCHIVED.value,
                archive_segment_id=segment_id,
                archived_at=archived_at,
            )
        )

    async def list_staged_archive(
        self, *, older_than: datetime | None = None, limit: int = 200
    ) -> list[Document]:
        stmt = (
            select(DocumentRow)
            .where(
                DocumentRow.archive_status == ArchiveStatus.STAGED.value,
                DocumentRow.deleted_at.is_(None),
            )
            .order_by(DocumentRow.created_at)
            .limit(limit)
        )
        if older_than is not None:
            stmt = stmt.where(DocumentRow.created_at <= older_than)
        return [_doc(r) for r in (await self.s.execute(stmt)).scalars().all()]

    async def add_version(self, version: DocumentVersion) -> None:
        self.s.add(
            DocumentVersionRow(
                document_version_id=version.document_version_id,
                document_id=version.document_id,
                tenant_id=version.tenant_id,
                version=version.version,
                parser=version.parser,
                parser_version=version.parser_version,
                page_count=version.page_count,
                node_count=version.node_count,
                chunk_count=version.chunk_count,
                status=version.status,
            )
        )
        await self.s.flush()

    async def get_version(self, tenant_id: str, document_version_id: str) -> DocumentVersion | None:
        r = await self.s.get(DocumentVersionRow, document_version_id)
        if r is None or r.tenant_id != tenant_id:
            return None
        return DocumentVersion(
            document_version_id=r.document_version_id,
            document_id=r.document_id,
            tenant_id=r.tenant_id,
            version=r.version,
            parser=r.parser,
            parser_version=r.parser_version,
            page_count=r.page_count,
            node_count=r.node_count,
            chunk_count=r.chunk_count,
            status=r.status,
        )

    async def add_nodes(self, nodes: Sequence[DocumentNode]) -> None:
        self.s.add_all(
            [
                DocumentNodeRow(
                    node_id=n.node_id,
                    document_id=n.document_id,
                    document_version_id=n.document_version_id,
                    tenant_id=n.tenant_id,
                    representation=n.representation.value,
                    parent_id=n.parent_id,
                    ordinal=n.ordinal,
                    depth=n.depth,
                    title=n.title,
                    section_path=n.section_path,
                    page_start=n.page_start,
                    page_end=n.page_end,
                    text=n.text,
                    text_hash=n.text_hash,
                    token_estimate=n.token_estimate,
                    entities=list(n.entities),
                    system_metadata=n.system_metadata,
                )
                for n in nodes
            ]
        )
        await self.s.flush()

    async def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.s.add_all(
            [
                ChunkRow(
                    chunk_id=c.chunk_id,
                    node_id=c.node_id,
                    document_id=c.document_id,
                    document_version_id=c.document_version_id,
                    tenant_id=c.tenant_id,
                    ordinal=c.ordinal,
                    text=c.text,
                    text_hash=c.text_hash,
                    contextual_text=c.contextual_text,
                    page=c.page,
                    section_path=c.section_path,
                    token_estimate=c.token_estimate,
                    entities=list(c.entities),
                )
                for c in chunks
            ]
        )
        await self.s.flush()

    async def add_edges(self, edges: Sequence[ContextEdge]) -> None:
        self.s.add_all(
            [
                ContextEdgeRow(
                    tenant_id=e.tenant_id,
                    document_id=e.document_id,
                    source_id=e.source_id,
                    target_id=e.target_id,
                    edge=e.edge.value,
                    weight=e.weight,
                    label=e.label,
                )
                for e in edges
            ]
        )
        await self.s.flush()

    async def replace_version_content(self, tenant_id: str, document_id: str) -> None:
        for table in (ChunkRow, DocumentNodeRow, ContextEdgeRow):
            await self.s.execute(
                delete(table).where(table.document_id == document_id, table.tenant_id == tenant_id)
            )

    async def list_nodes(
        self, tenant_id: str, document_id: str, *, version_id: str | None = None
    ) -> list[DocumentNode]:
        stmt = (
            select(DocumentNodeRow)
            .where(
                DocumentNodeRow.tenant_id == tenant_id, DocumentNodeRow.document_id == document_id
            )
            .order_by(DocumentNodeRow.depth, DocumentNodeRow.ordinal)
        )
        if version_id:
            stmt = stmt.where(DocumentNodeRow.document_version_id == version_id)
        return [_node(r) for r in (await self.s.execute(stmt)).scalars().all()]

    async def get_nodes(self, tenant_id: str, node_ids: Sequence[str]) -> list[DocumentNode]:
        if not node_ids:
            return []
        rows = (
            (
                await self.s.execute(
                    select(DocumentNodeRow).where(
                        DocumentNodeRow.tenant_id == tenant_id,
                        DocumentNodeRow.node_id.in_(list(node_ids)),
                    )
                )
            )
            .scalars()
            .all()
        )
        return [_node(r) for r in rows]

    async def list_chunks(
        self, tenant_id: str, document_id: str, *, unindexed_only: bool = False, limit: int = 5000
    ) -> list[Chunk]:
        stmt = (
            select(ChunkRow)
            .where(ChunkRow.tenant_id == tenant_id, ChunkRow.document_id == document_id)
            .order_by(ChunkRow.node_id, ChunkRow.ordinal)
            .limit(limit)
        )
        if unindexed_only:
            stmt = stmt.where(ChunkRow.indexed_at.is_(None))
        return [_chunk(r) for r in (await self.s.execute(stmt)).scalars().all()]

    async def get_chunks(self, tenant_id: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        rows = (
            (
                await self.s.execute(
                    select(ChunkRow).where(
                        ChunkRow.tenant_id == tenant_id, ChunkRow.chunk_id.in_(list(chunk_ids))
                    )
                )
            )
            .scalars()
            .all()
        )
        return [_chunk(r) for r in rows]

    async def chunks_for_nodes(self, tenant_id: str, node_ids: Sequence[str]) -> list[Chunk]:
        if not node_ids:
            return []
        rows = (
            await self.s.execute(
                select(ChunkRow)
                .where(ChunkRow.tenant_id == tenant_id, ChunkRow.node_id.in_(list(node_ids)))
                .order_by(ChunkRow.node_id, ChunkRow.ordinal)
            )
        ).scalars()
        return [_chunk(r) for r in rows]

    async def set_node_summaries(self, tenant_id: str, summaries: dict[str, str]) -> None:
        if not summaries:
            return
        rows = (
            await self.s.execute(
                select(DocumentNodeRow).where(
                    DocumentNodeRow.tenant_id == tenant_id,
                    DocumentNodeRow.node_id.in_(list(summaries)),
                )
            )
        ).scalars()
        for r in rows:
            r.system_metadata = {**(r.system_metadata or {}), "summary": summaries[r.node_id]}
        await self.s.flush()

    async def node_summaries(self, tenant_id: str, node_ids: Sequence[str]) -> dict[str, str]:
        if not node_ids:
            return {}
        rows = (
            await self.s.execute(
                select(DocumentNodeRow).where(
                    DocumentNodeRow.tenant_id == tenant_id,
                    DocumentNodeRow.node_id.in_(list(node_ids)),
                )
            )
        ).scalars()
        return {
            r.node_id: str((r.system_metadata or {}).get("summary"))
            for r in rows
            if (r.system_metadata or {}).get("summary")
        }

    async def mark_chunks_indexed(
        self, chunk_ids: Sequence[str], *, fingerprint: str, indexed_at: datetime
    ) -> int:
        if not chunk_ids:
            return 0
        result = await self.s.execute(
            update(ChunkRow)
            .where(ChunkRow.chunk_id.in_(list(chunk_ids)))
            .values(indexed_at=indexed_at, index_fingerprint=fingerprint)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def edges_from(
        self,
        tenant_id: str,
        source_ids: Sequence[str],
        *,
        kinds: Sequence[ContextGraphEdge] | None = None,
    ) -> list[ContextEdge]:
        if not source_ids:
            return []
        stmt = select(ContextEdgeRow).where(
            ContextEdgeRow.tenant_id == tenant_id, ContextEdgeRow.source_id.in_(list(source_ids))
        )
        if kinds:
            stmt = stmt.where(ContextEdgeRow.edge.in_([k.value for k in kinds]))
        return [_edge(r) for r in (await self.s.execute(stmt)).scalars().all()]

    async def edges_to(
        self,
        tenant_id: str,
        target_ids: Sequence[str],
        *,
        kinds: Sequence[ContextGraphEdge] | None = None,
    ) -> list[ContextEdge]:
        if not target_ids:
            return []
        stmt = select(ContextEdgeRow).where(
            ContextEdgeRow.tenant_id == tenant_id, ContextEdgeRow.target_id.in_(list(target_ids))
        )
        if kinds:
            stmt = stmt.where(ContextEdgeRow.edge.in_([k.value for k in kinds]))
        return [_edge(r) for r in (await self.s.execute(stmt)).scalars().all()]

    async def visibility_keys(self, tenant_id: str, document_id: str) -> list[str]:
        r = await self.s.get(DocumentRow, document_id)
        if r is None or r.tenant_id != tenant_id:
            return []
        return list(r.visibility_keys or [])
