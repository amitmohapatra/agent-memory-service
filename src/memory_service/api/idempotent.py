"""Idempotent write helper for routes: replay -> reserve -> handler -> complete -> warm.

Every persistent write goes through :func:`run_idempotent`, so the ``Idempotency-Key`` the
OpenAPI document advertises on a write is honoured by it: a retry with the same key and body
gets the first response (status, body, ``Location``) with ``Idempotent-Replayed: true``,
never a second effect - and never the 404 a second ``DELETE`` would otherwise earn.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from memory_service.api.headers import IDEMPOTENT_REPLAYED_HEADER, LOCATION_HEADER
from memory_service.application.container import Container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.ids import stable_key
from memory_service.modules.idempotency.service import IdempotencyService
from memory_service.ports.uow import UnitOfWork

Handler = Callable[
    [UnitOfWork], Awaitable[tuple[int, dict[str, Any], Callable[[], Awaitable[None]] | None]]
]
#: The path (under the API's root, e.g. ``/v1/memories/mem_1``) a response body names:
#: the created resource of a 201, the job of a 202. ``None`` when the body names none.
Locate = Callable[[dict[str, Any]], str | None]

NO_CONTENT = 204


def resource_at(template: str, field: str) -> Locate:
    """``Location`` of a created resource: ``template`` filled with the body's ``field``
    (``resource_at("/v1/memories/{}", "memory_id")``)."""

    def locate(body: dict[str, Any]) -> str | None:
        value = body.get(field)
        return template.format(value) if value else None

    return locate


def first_job(body: dict[str, Any]) -> str | None:
    """``Location`` of an accepted write: the status of the first job it queued."""
    jobs = body.get("job_ids") or []
    return f"/v1/jobs/{jobs[0]}" if jobs else None


def _respond(
    request: Request,
    status: int,
    body: dict[str, Any],
    *,
    replayed: bool,
    location: Locate | None,
) -> Response:
    headers: dict[str, str] = {}
    if replayed:
        headers[IDEMPOTENT_REPLAYED_HEADER] = "true"
    where = location(body) if location is not None else None
    if where:
        headers[LOCATION_HEADER] = request.scope.get("root_path", "") + where
    if status == NO_CONTENT:
        return Response(status_code=NO_CONTENT, headers=headers)
    return JSONResponse(status_code=status, content=body, headers=headers)


def default_idempotency_key(ctx: MemoryExecutionContext, *parts: str) -> str:
    """Server-side default when the client sends no Idempotency-Key: lineage + content."""
    return "auto-" + stable_key(
        ctx.tenant_id,
        ctx.thread_id or "",
        ctx.session_id or "",
        ctx.turn_id or "",
        ctx.agent_run_id or "",
        *parts,
    )


def derived_or_body(request: Request, body: Any, identity: Sequence[str]) -> Any:
    """What a replay of this request is compared against: the whole body when the *client*
    sent an Idempotency-Key, and only ``identity`` when the service derived one.

    A client's key is a promise that two requests are the same, so a body that differs is a
    mistake worth a 409. A derived key is a de-duplication convenience the caller never asked
    for, and comparing it against the whole body contradicts the key itself: ``POST /v1/threads``
    derives its key from (thread, title) and ``POST /v1/messages`` from (role, kind, content),
    so a second call that differed only in ``custom_metadata`` hashed differently and was
    refused with "Idempotency-Key reused with a different payload" - a conflict for a header
    nobody sent, on routes that document the opposite ("retries with ... the same lineage +
    content when the header is absent return the original acknowledgement"). Found by the
    agent suite: an adapter reconnecting to a thread it had already created was answered 409.
    """
    return body.model_dump(mode="json") if request.state.idempotency_key else list(identity)


async def run_idempotent(
    request: Request,
    container: Container,
    ctx: MemoryExecutionContext,
    *,
    key: str | None,
    payload: Any,
    handler: Handler,
    stored_body: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    location: Locate | None = None,
) -> Response:
    """Execute ``handler`` inside a Unit of Work exactly once per (tenant, key, payload).

    ``handler`` returns ``(status, body, after_commit)``; the body is what later retries will
    receive verbatim. ``after_commit`` runs only after a successful commit.

    ``key=None`` runs the handler once and records nothing: the route has no natural key for
    the request (a create with a server-generated id, a key issued under a name two keys may
    share) and the client sent no ``Idempotency-Key``, so two calls are two resources.

    ``stored_body`` narrows what a replay may return: a secret that is shown once (an issued
    key's token) goes out on the first response and is never written to the idempotency
    table or the cache, so a retried request gets the record and ``Idempotent-Replayed``,
    not a second look at the secret.

    ``location`` names the resource (201) or the job (202) the body identifies; a replay
    carries the same ``Location``. A 204 is written without a body and replayed as one.
    """
    uow_factory = container.services["uow_factory"]
    if key is None:
        async with uow_factory() as uow:
            status, body, after_commit = await handler(uow)
            await uow.commit()
        if after_commit is not None:
            await after_commit()
        return _respond(request, status, body, replayed=False, location=location)
    idem: IdempotencyService = container.services["idempotency"]
    request_hash = idem.request_hash(payload)
    cached = await idem.lookup_cached(ctx.tenant_id, key, request_hash)
    if cached is not None:
        return _respond(request, cached.status, cached.body, replayed=True, location=location)

    async with uow_factory() as uow:
        replay = await idem.begin(uow.idempotency, ctx.tenant_id, key, request_hash)
        if replay is not None:
            return _respond(request, replay.status, replay.body, replayed=True, location=location)
        status, body, after_commit = await handler(uow)
        kept = stored_body(body) if stored_body is not None else body
        await idem.complete(uow.idempotency, ctx.tenant_id, key, status=status, body=kept)
        await uow.commit()
    await idem.warm_cache(ctx.tenant_id, key, request_hash, status=status, body=kept)
    if after_commit is not None:
        await after_commit()
    return _respond(request, status, body, replayed=False, location=location)
