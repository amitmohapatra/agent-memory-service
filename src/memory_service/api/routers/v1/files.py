"""Public /v1 routes: file ingestion and document status."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import run_idempotent
from memory_service.api.validation import CustomMetadata
from memory_service.domain.enums import ArchiveStatus, DocumentStatus, Visibility
from memory_service.domain.errors import ValidationFailed
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

    document_id: str
    filename: str
    checksum: str
    size_bytes: int
    job_ids: list[str] = Field(default_factory=list)
    deduplicated: bool = False
    observation_id: str | None = None


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

    document_id: str
    tenant_id: str
    title: str
    filename: str
    media_type: str
    size_bytes: int
    checksum: str
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
    current_version_id: str | None = None
    thread_id: str | None = None
    created_at: datetime
    last_error: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/files",
    response_model=FileAckResponse,
    status_code=202,
    tags=["files"],
    summary="Ingest a file into RAG memory (multipart)",
    description=(
        "Fields: `file` (binary), `scope` (JSON object, same shape as message scope), optional "
        "`message_id`, `title`, `visibility` (PRIVATE|USER|GROUP|THREAD|WORK|WORKSPACE|TENANT), "
        "`custom_metadata` (JSON). Identical bytes within a tenant are deduplicated by SHA-256."
    ),
    responses={
        **_WRITE_ERRORS,
        202: {"model": FileAckResponse, "description": "Accepted (durable)"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "scope": {
                                "type": "string",
                                "description": "JSON-encoded scope",
                                "example": '{"thread_id":"thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"}',
                            },
                            "message_id": {
                                "type": "string",
                                "example": "msg_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                            },
                            "title": {"type": "string", "example": "ACME FY26 Annual Report"},
                            "visibility": {
                                "type": "string",
                                "enum": [v.value for v in Visibility],
                                "example": "THREAD",
                            },
                            "custom_metadata": {"type": "string", "example": '{"source":"upload"}'},
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
)
async def upload_file(
    request: Request,
    container: ContainerDep,
    _: ServicePrincipalDep,
    file: Annotated[UploadFile, File()],
    scope: Annotated[str | None, Form()] = None,
    message_id: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
    visibility: Annotated[
        Visibility | None,
        Form(
            description=(
                "Who may retrieve the document, narrowest first: PRIVATE, RUN, THREAD, WORK, "
                "AGENT_GROUP, GROUP, USER, WORKSPACE, TENANT, GLOBAL. Omit for the thread, "
                "else the user."
            )
        ),
    ] = None,
    custom_metadata: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    try:
        scope_body = ScopeBody.model_validate(json.loads(scope)) if scope else ScopeBody()
        metadata = _METADATA.validate_python(json.loads(custom_metadata)) if custom_metadata else {}
    except (ValueError, ValidationError) as exc:
        raise ValidationFailed(
            "invalid scope/custom_metadata", details={"error": str(exc)[:300]}
        ) from exc
    ctx = build_context(request, container, scope_body)
    data = await file.read()
    filename = file.filename or "upload.bin"
    media_type = file.content_type or "application/octet-stream"
    if media_type == "application/octet-stream":
        import mimetypes

        media_type = mimetypes.guess_type(filename)[0] or media_type
    checksum = content_hash(data)
    key = request.state.idempotency_key or f"file-{ctx.tenant_id}-{checksum}-{message_id or ''}"
    service: IngestionService = container.services["ingestion"]

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
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["files"],
    summary="Document status",
    responses=_READ_ERRORS,
)
async def get_document(
    document_id: str, ctx: HeaderContextDep, container: ContainerDep
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
