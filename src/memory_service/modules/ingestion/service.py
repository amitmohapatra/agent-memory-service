"""IngestionService: durable file acceptance -> async parse -> hierarchy/chunks/context graph.

Accept path (synchronous, one transaction):
    checksum -> dedup by (tenant, checksum) -> document row + staged bytes + FILE observation
    -> outbox job document.parse -> COMMIT -> 202 FileHandle

Parse job (async): parser -> nodes/chunks/edges persisted (replacing older versions)
-> document READY -> outbox job document.index (M6) -> raw file archived to the file bucket
(immutable + verified) -> staged bytes purged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from memory_service.config.constants import DocumentSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Document
from memory_service.domain.enums import ArchiveStatus, ObservationKind, Visibility
from memory_service.domain.errors import (
    CorruptSource,
    DependencyUnavailable,
    NotFound,
    ValidationFailed,
)
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import visibility_keys
from memory_service.modules.ingestion.chunking import chunk_nodes, situate_chunks
from memory_service.modules.llm.assist import LLMAssist
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import archive_bytes_total, stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.blob import BlobStore
from memory_service.ports.intelligence import DocumentParser
from memory_service.ports.repositories import ArchiveSegment
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory

log = get_logger(__name__)

TASK_DOCUMENT_PARSE = "document.parse"
TASK_DOCUMENT_INDEX = "document.index"


@dataclass(frozen=True)
class FileAck:
    document_id: str
    filename: str
    checksum: str
    size_bytes: int
    job_ids: list[str]
    deduplicated: bool = False
    observation_id: str | None = None


class IngestionService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        authz: AuthorizationService,
        parser: DocumentParser,
        blob: BlobStore | None,
        *,
        settings: DocumentSettings,
        file_bucket: str,
        tenant_shards: int = 64,
        fallback_parser: DocumentParser | None = None,
        assist: LLMAssist | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.authz = authz
        self.parser = parser
        self.fallback = fallback_parser
        self.blob = blob
        self.cfg = settings
        self.file_bucket = file_bucket
        self.shards = tenant_shards
        self.assist = assist or LLMAssist.disabled()

    # -- accept -----------------------------------------------------------------
    async def accept_file(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        filename: str,
        media_type: str,
        data: bytes,
        message_id: str | None = None,
        title: str | None = None,
        visibility: Visibility | None = None,
        custom_metadata: dict[str, Any] | None = None,
        source_system: str | None = None,
        source_id: str | None = None,
    ) -> FileAck:
        if not data:
            raise ValidationFailed("empty file")
        if len(data) > self.cfg.max_file_bytes:
            raise ValidationFailed(f"file exceeds {self.cfg.max_file_bytes} bytes")
        if media_type not in self.parser.supported_media_types and not (
            self.fallback and media_type in self.fallback.supported_media_types
        ):
            raise ValidationFailed(
                f"unsupported media type {media_type}",
                details={"supported": sorted(self.parser.supported_media_types)},
            )
        if ctx.thread_id:
            thread = await uow.threads.get(ctx.tenant_id, ctx.thread_id)
            if thread is not None:
                await self.authz.require(ctx, "can_write", "thread", ctx.thread_id)
        checksum = content_hash(data)
        existing = await uow.documents.find_by_checksum(ctx.tenant_id, checksum)
        if existing is not None and await self.authz.allowed(
            ctx, "can_read", "document", existing.document_id
        ):
            return FileAck(
                existing.document_id,
                existing.filename,
                checksum,
                existing.size_bytes,
                [],
                deduplicated=True,
            )

        with (
            span("ingest.accept", tenant_id=ctx.tenant_id),
            stage_seconds.labels("ingest.accept").time(),
        ):
            vis = visibility or (
                Visibility.THREAD
                if ctx.thread_id
                else Visibility.USER
                if ctx.user_id
                else Visibility.WORKSPACE
                if ctx.workspace_id
                else Visibility.TENANT
            )
            keys = visibility_keys(
                ctx.tenant_id,
                vis,
                owner_principal=ctx.principal_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                thread_id=ctx.thread_id,
                work_id=ctx.work_id,
                agent_group_id=ctx.agent_group_id,
            )
            document = Document(
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                owner_user_id=ctx.user_id,
                thread_id=ctx.thread_id,
                title=title or filename,
                filename=filename,
                media_type=media_type,
                size_bytes=len(data),
                checksum=checksum,
                source_system=source_system,
                source_id=source_id,
                custom_metadata=custom_metadata or {},
                system_metadata={"visibility": vis.value, "trace_id": ctx.trace_id},
            )
            await uow.documents.add(document, visibility_keys=keys, message_id=message_id)
            await uow.documents.stage_bytes(
                ctx.tenant_id, document.document_id, data, checksum=checksum
            )
            await self.authz.grant_document(
                ctx,
                document.document_id,
                thread_id=ctx.thread_id,
                workspace_id=ctx.workspace_id,
                revisions=uow.revisions,
            )
            observation = Observation(
                tenant_id=ctx.tenant_id,
                kind=ObservationKind.FILE,
                content=f"file:{filename}",
                content_hash=checksum,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                thread_id=ctx.thread_id,
                session_id=ctx.session_id,
                turn_id=ctx.turn_id,
                work_id=ctx.work_id,
                task_id=ctx.task_id,
                agent_id=ctx.agent_id,
                agent_group_id=ctx.agent_group_id,
                agent_run_id=ctx.agent_run_id,
                parent_agent_run_id=ctx.parent_agent_run_id,
                principal_id=ctx.principal_id,
                trace_id=ctx.trace_id,
                message_id=message_id,
                document_id=document.document_id,
                source_system=source_system,
                source_id=source_id,
                custom_metadata={"media_type": media_type, "size_bytes": len(data)},
            )
            await uow.observations.add(observation)
            job_ids: list[str] = []
            outbox_id = await uow.enqueue(
                JobSpec(
                    task_name=TASK_DOCUMENT_PARSE,
                    queue=Queue.DOCUMENT_PARSE,
                    payload={
                        "tenant_id": ctx.tenant_id,
                        "document_id": document.document_id,
                        "trace_id": ctx.trace_id,
                        "observation_id": observation.observation_id,
                    },
                    idempotency_key=f"parse:{document.document_id}",
                    lock=f"document:{document.document_id}",
                    tenant_id=ctx.tenant_id,
                )
            )
            if outbox_id is not None:
                job_ids.append(f"obx_{outbox_id}")
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.DOCUMENT, document.document_id)
            if ctx.user_id:
                await uow.revisions.bump(ctx.tenant_id, RevisionKind.USER, ctx.user_id)
            return FileAck(
                document.document_id,
                filename,
                checksum,
                len(data),
                job_ids,
                observation_id=observation.observation_id,
            )

    async def get_document(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, document_id: str
    ) -> Document:
        document = await uow.documents.get(ctx.tenant_id, document_id)
        if document is None:
            raise NotFound("Document not found")
        await self.authz.require(ctx, "can_read", "document", document_id)
        return document

    # -- parse job --------------------------------------------------------------
    async def parse_document(self, tenant_id: str, document_id: str) -> dict[str, int]:
        async with self.uow_factory() as uow:
            document = await uow.documents.get(tenant_id, document_id)
            if document is None:
                raise NotFound(f"document {document_id} not found")
            data = await uow.documents.staged_bytes(tenant_id, document_id)
        if data is None:
            data = await self._load_archived_bytes(document)
        if content_hash(data) != document.checksum:
            raise CorruptSource("staged bytes do not match the document checksum")

        with span("ingest.parse", tenant_id=tenant_id), stage_seconds.labels("ingest.parse").time():
            parser = (
                self.parser
                if document.media_type in self.parser.supported_media_types
                else self.fallback
            )
            if parser is None:
                raise ValidationFailed(f"no parser for {document.media_type}")
            try:
                parsed = await parser.parse(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    filename=document.filename,
                    media_type=document.media_type,
                    data=data,
                )
            except (CorruptSource, ValidationFailed):
                async with self.uow_factory() as uow:
                    await uow.documents.set_status(
                        tenant_id, document_id, status="FAILED", error="parse failed"
                    )
                    await uow.commit()
                raise
            except DependencyUnavailable:
                raise
            except Exception as exc:
                async with self.uow_factory() as uow:
                    await uow.documents.set_status(
                        tenant_id,
                        document_id,
                        status="FAILED",
                        error=f"{type(exc).__name__}: {exc}"[:500],
                    )
                    await uow.commit()
                raise CorruptSource(f"parser error: {type(exc).__name__}") from exc
            chunks = chunk_nodes(
                parsed.nodes,
                document_title=parsed.title,
                max_tokens=self.cfg.max_chunk_tokens,
                min_tokens=self.cfg.min_chunk_tokens,
                overlap_tokens=self.cfg.chunk_overlap_tokens,
                keep_tables_intact=self.cfg.keep_tables_intact,
                keep_code_intact=self.cfg.keep_code_intact,
                contextual=self.cfg.contextual_chunks,
            )
            if self.assist.wants("chunk_context"):
                chunks = await situate_chunks(
                    self.assist, chunks, parsed.nodes, document_title=parsed.title
                )
            version = parsed.version.model_copy(
                update={
                    "node_count": len(parsed.nodes),
                    "chunk_count": len(chunks),
                    "page_count": parsed.page_count,
                }
            )
            async with self.uow_factory() as uow:
                await uow.documents.replace_version_content(tenant_id, document_id)
                await uow.documents.add_version(version)
                await uow.documents.add_nodes(parsed.nodes)
                await uow.documents.add_chunks(chunks)
                await uow.documents.add_edges(parsed.edges)
                await uow.documents.set_status(
                    tenant_id,
                    document_id,
                    status="READY",
                    current_version_id=version.document_version_id,
                )
                await uow.revisions.bump(tenant_id, RevisionKind.DOCUMENT, document_id)
                await uow.enqueue(
                    JobSpec(
                        task_name=TASK_DOCUMENT_INDEX,
                        queue=Queue.EMBEDDING,
                        payload={
                            "tenant_id": tenant_id,
                            "document_id": document_id,
                            "document_version_id": version.document_version_id,
                        },
                        idempotency_key=f"index:{version.document_version_id}",
                        lock=f"document:{document_id}",
                        tenant_id=tenant_id,
                    )
                )
                await uow.commit()
        log.info(
            "document.parsed",
            tenant_id=tenant_id,
            document_id=document_id,
            nodes=len(parsed.nodes),
            chunks=len(chunks),
            edges=len(parsed.edges),
            parser=version.parser,
        )
        await self.archive_raw_file(tenant_id, document_id)
        return {"nodes": len(parsed.nodes), "chunks": len(chunks), "edges": len(parsed.edges)}

    # -- raw file archive -------------------------------------------------------
    async def archive_raw_file(self, tenant_id: str, document_id: str) -> str | None:
        if self.blob is None:
            return None
        async with self.uow_factory() as uow:
            document = await uow.documents.get(tenant_id, document_id)
            if document is None or document.archive_status is not ArchiveStatus.STAGED:
                return None
            data = await uow.documents.staged_bytes(tenant_id, document_id)
        if data is None:
            return None
        from memory_service.modules.archive.segments import tenant_shard

        created = document.created_at
        key = (
            f"tenant-shard={tenant_shard(tenant_id, self.shards):03d}/tenant={tenant_id}/"
            f"year={created:%Y}/month={created:%m}/document={document_id}/{document.checksum}.bin"
        )
        segment_id = f"seg_file_{document_id}"
        async with self.uow_factory() as uow:
            if await uow.archive.get(segment_id) is None:
                await uow.archive.add(
                    ArchiveSegment(
                        segment_id=segment_id,
                        tenant_id=tenant_id,
                        kind="file",
                        document_id=document_id,
                        bucket=self.file_bucket,
                        key=key,
                        size_bytes=len(data),
                        raw_bytes=len(data),
                        checksum_sha256=document.checksum,
                        message_count=0,
                        status="UPLOADING",
                        manifest={"filename": document.filename, "media_type": document.media_type},
                    )
                )
                await uow.commit()
        try:
            try:
                ref = await self.blob.put(
                    self.file_bucket,
                    key,
                    data,
                    content_type=document.media_type,
                    checksum_sha256=document.checksum,
                    if_generation_match=0,
                    metadata={"tenant": tenant_id, "document": document_id},
                )
            except Exception as exc:
                from memory_service.adapters.blob.filesystem import BlobAlreadyExists

                if isinstance(exc, BlobAlreadyExists):
                    ref = await self.blob.head(self.file_bucket, key)
                else:
                    raise
            verified = await self.blob.verify(ref) and ref.checksum_sha256 == document.checksum
        except Exception as exc:
            async with self.uow_factory() as uow:
                await uow.archive.mark_failed(segment_id, error=f"{type(exc).__name__}: {exc}")
                await uow.commit()
            raise DependencyUnavailable(f"file archive failed: {type(exc).__name__}") from exc
        if not verified:
            async with self.uow_factory() as uow:
                await uow.archive.mark_failed(segment_id, error="verification failed")
                await uow.commit()
            raise CorruptSource(
                f"raw file {document_id} failed archive verification; staged bytes kept"
            )
        now = datetime.now(UTC)
        async with self.uow_factory() as uow:
            await uow.archive.mark_verified(segment_id, generation=ref.generation, verified_at=now)
            await uow.documents.mark_archived(
                tenant_id, document_id, segment_id=segment_id, archived_at=now
            )
            await uow.documents.purge_staged_bytes(tenant_id, document_id)
            await uow.commit()
        archive_bytes_total.labels("file", "raw").inc(len(data))
        return segment_id

    async def _load_archived_bytes(self, document: Document) -> bytes:
        if self.blob is None:
            raise NotFound("raw bytes unavailable: not staged and no blob store")
        async with self.uow_factory() as uow:
            seg = await uow.archive.get(f"seg_file_{document.document_id}")
        if seg is None or seg.status != "VERIFIED":
            raise NotFound("raw bytes unavailable: archive segment missing")
        data = await self.blob.get(seg.bucket, seg.key, generation=seg.generation)
        if content_hash(data) != document.checksum:
            raise CorruptSource("archived file checksum mismatch")
        return data
