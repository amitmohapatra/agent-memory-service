"""Public /v1 routes: file ingestion and document status."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import first_job, run_idempotent
from memory_service.api.params import DocumentIdPath
from memory_service.api.validation import CustomMetadata
from memory_service.domain.enums import ArchiveStatus, DocumentStatus, Visibility
from memory_service.domain.errors import PayloadTooLarge, ValidationFailed
from memory_service.domain.ids import content_hash
from memory_service.modules.ingestion.service import IngestionService

router = APIRouter()

_WRITE_ERRORS = error_responses(401, 403, 409, 422, 503)
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)
_METADATA: TypeAdapter[dict[str, Any]] = TypeAdapter(CustomMetadata)


class FileAckResponse(BaseModel):
    """Durable acknowledgement: bytes and the parse job are committed; parsing is asynchronous."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "doc_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "filename": "acme-fy26-annual-report.pdf",
                    "checksum": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                    "size_bytes": 204800,
                    "job_ids": ["obx_1051"],
                    "deduplicated": False,
                    "observation_id": "obs_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                }
            ]
        }
    )

    document_id: str = Field(
        description="The document the upload created (doc_...), or the existing one when "
        "deduplicated."
    )
    filename: str = Field(description="The uploaded file's name.")
    checksum: str = Field(
        description="SHA-256 of the bytes, hex: identical bytes in a tenant are one document."
    )
    size_bytes: int = Field(description="The file's size in bytes.")
    job_ids: list[str] = Field(
        default_factory=list,
        description="The parse job queued for it (obx_...); poll GET /v1/jobs/{job_id}.",
    )
    deduplicated: bool = Field(
        default=False,
        description="true: the same bytes were already a document the caller may read, and "
        "its id is returned (no job is queued).",
    )
    observation_id: str | None = Field(
        default=None, description="The FILE observation the upload recorded, when one was."
    )


class DocumentResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "doc_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "tenant_id": "acme",
                    "title": "ACME FY26 Annual Report",
                    "filename": "acme-fy26-annual-report.pdf",
                    "media_type": "application/pdf",
                    "size_bytes": 204800,
                    "checksum": "9f86…",
                    "status": "READY",
                    "archive_status": "ARCHIVED",
                    "current_version_id": "dcv_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "thread_id": None,
                    "created_at": "2026-09-14T10:00:00Z",
                    "custom_metadata": {},
                }
            ]
        }
    )

    document_id: str = Field(description="The document's id (doc_...).")
    tenant_id: str = Field(description="The tenant the record belongs to.")
    title: str = Field(description="The title given at upload, else the filename.")
    filename: str = Field(description="The uploaded file's name.")
    media_type: str = Field(
        description="The file's media type (e.g. application/pdf), as uploaded or guessed "
        "from the name."
    )
    size_bytes: int = Field(description="The file's size in bytes.")
    checksum: str = Field(description="SHA-256 of the bytes, hex.")
    status: DocumentStatus = Field(
        ...,
        description="STAGED: accepted, parse job queued or running; READY: parsed and "
        "indexed, retrievable; FAILED: the parse job gave up, see last_error.",
    )
    archive_status: ArchiveStatus = Field(
        ...,
        description="Where the raw bytes live: STAGED (PostgreSQL only), ARCHIVING, ARCHIVED "
        "(blob store, verified) or PURGED (removed from the hot database after the grace "
        "period).",
    )
    current_version_id: str | None = Field(
        default=None,
        description="The parsed version retrieval reads (dcv_...); null until the first parse.",
    )
    thread_id: str | None = Field(
        default=None, description="The thread the document was attached to, if any."
    )
    created_at: datetime = Field(description="When the record was created (ISO 8601, UTC).")
    last_error: str | None = Field(
        default=None, description="Why the parse job gave up, when status is FAILED."
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict, description="The caller-defined JSON given at upload."
    )


_FILE = "The file's bytes (at most the ingestion limit, and 25 MB per request)."
_FORM_SCOPE = (
    "The scope as a JSON object, the same shape as a message's (thread_id, agent_id, ...): the "
    "document's lineage. A thread_id nobody has written to yet is created for the caller, as "
    "a message creates it. Omitted: the headers' scope alone."
)
_MESSAGE = "The message (msg_...) the file was attached to; the document joins its thread."
_TITLE = "The document's title (shown in citations); omitted: the filename."
_FORM_METADATA = (
    "Caller-defined JSON object (at most 32 keys, 2 levels, 8192 bytes), kept on the document."
)

_UPLOAD_ROUTE: dict[str, Any] = {
    "response_model": FileAckResponse,
    "status_code": 202,
    "summary": "Ingest a file into RAG memory (multipart)",
    "description": (
        "Fields: `file` (binary), `scope` (JSON object, same shape as message scope), optional "
        "`message_id`, `title`, `visibility` "
        "(PRIVATE|RUN|THREAD|AGENT_GROUP|USER|WORKSPACE|TENANT), "
        "`custom_metadata` (JSON). Identical bytes within a tenant are deduplicated by SHA-256."
    ),
    "responses": {
        **_WRITE_ERRORS,
        202: {"model": FileAckResponse, "description": "Accepted (durable)"},
    },
    "openapi_extra": {
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": _FILE,
                            },
                            "scope": {
                                "type": "string",
                                "description": _FORM_SCOPE,
                                "example": '{"thread_id":"thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"}',
                            },
                            "message_id": {
                                "type": "string",
                                "description": _MESSAGE,
                                "example": "msg_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                            },
                            "title": {
                                "type": "string",
                                "description": _TITLE,
                                "example": "ACME FY26 Annual Report",
                            },
                            "visibility": {
                                "type": "string",
                                "enum": [v.value for v in Visibility],
                                "description": "Who may retrieve the document, narrowest "
                                "first; omitted: the thread, else the user.",
                                "example": "THREAD",
                            },
                            "custom_metadata": {
                                "type": "string",
                                "description": _FORM_METADATA,
                                "example": '{"source":"upload"}',
                            },
                        },
                    },
                    "example": {
                        "scope": '{"thread_id":"thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"}',
                        "title": "ACME FY26 Annual Report",
                    },
                }
            }
        }
    },
}


#: How much of an upload is read at a time while it is counted against the file limit.
UPLOAD_CHUNK_BYTES = 1024 * 1024


async def read_bounded(file: UploadFile, limit: int) -> bytes:
    """The upload's bytes, read a chunk at a time and refused the moment they pass ``limit``
    - never the whole file first and the size after. A size the parser already knows is
    refused before anything is read."""
    if file.size is not None and file.size > limit:
        raise PayloadTooLarge(f"file exceeds {limit} bytes", details={"size_bytes": file.size})
    chunks: list[bytes] = []
    seen = 0
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        seen += len(chunk)
        if seen > limit:
            raise PayloadTooLarge(f"file exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/documents", tags=["documents"], name="upload_document", **_UPLOAD_ROUTE)
async def upload_document(
    request: Request,
    container: ContainerDep,
    _: ServicePrincipalDep,
    file: Annotated[UploadFile, File(description=_FILE)],
    scope: Annotated[str | None, Form(description=_FORM_SCOPE)] = None,
    message_id: Annotated[str | None, Form(description=_MESSAGE)] = None,
    title: Annotated[str | None, Form(description=_TITLE)] = None,
    visibility: Annotated[
        Visibility | None,
        Form(
            description=(
                "Who may retrieve the document, narrowest first: PRIVATE, RUN, THREAD, "
                "AGENT_GROUP, USER, WORKSPACE (the team named in X-Trellis-Workspace; members "
                "only), TENANT. Omit for the thread, else the user."
            )
        ),
    ] = None,
    custom_metadata: Annotated[str | None, Form(description=_FORM_METADATA)] = None,
) -> Response:
    try:
        scope_body = ScopeBody.model_validate(json.loads(scope)) if scope else ScopeBody()
        metadata = _METADATA.validate_python(json.loads(custom_metadata)) if custom_metadata else {}
    except (ValueError, ValidationError) as exc:
        raise ValidationFailed(
            "invalid scope/custom_metadata", details={"error": str(exc)[:300]}
        ) from exc
    ctx = build_context(request, container, scope_body)
    service: IngestionService = container.services["ingestion"]
    data = await read_bounded(file, service.cfg.max_file_bytes)
    filename = file.filename or "upload.bin"
    media_type = file.content_type or "application/octet-stream"
    if media_type == "application/octet-stream":
        import mimetypes

        media_type = mimetypes.guess_type(filename)[0] or media_type
    checksum = content_hash(data)
    key = request.state.idempotency_key or f"file-{ctx.tenant_id}-{checksum}-{message_id or ''}"

    async def handler(uow):  # type: ignore[no-untyped-def]
        ack = await service.accept_file(
            uow,
            ctx,
            filename=filename,
            media_type=media_type,
            data=data,
            message_id=message_id,
            title=title,
            visibility=visibility,
            custom_metadata=metadata,
        )
        return 202, FileAckResponse(**ack.__dict__).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=key,
        payload={
            "checksum": checksum,
            "filename": filename,
            "message_id": message_id,
            "title": title,
        },
        handler=handler,
        location=first_job,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["documents"],
    summary="Document status",
    responses=_READ_ERRORS,
)
async def get_document(
    document_id: DocumentIdPath, ctx: HeaderContextDep, container: ContainerDep
) -> DocumentResponse:
    service: IngestionService = container.services["ingestion"]
    async with container.services["uow_factory"]() as uow:
        d = await service.get_document(uow, ctx, document_id)
    return DocumentResponse(
        document_id=d.document_id,
        tenant_id=d.tenant_id,
        title=d.title,
        filename=d.filename,
        media_type=d.media_type,
        size_bytes=d.size_bytes,
        checksum=d.checksum,
        status=DocumentStatus(str(d.system_metadata.get("status", "STAGED"))),
        archive_status=d.archive_status,
        current_version_id=d.current_version_id,
        thread_id=d.thread_id,
        created_at=d.created_at,
        last_error=d.system_metadata.get("last_error"),
        custom_metadata=d.custom_metadata,
    )
