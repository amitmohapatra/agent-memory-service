"""Idempotent write helper for routes: replay -> reserve -> handler -> complete -> warm."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from memory_service.application.container import Container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.ids import stable_key
from memory_service.modules.idempotency.service import IdempotencyService
from memory_service.ports.uow import UnitOfWork

Handler = Callable[
    [UnitOfWork], Awaitable[tuple[int, dict[str, Any], Callable[[], Awaitable[None]] | None]]
]


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


async def run_idempotent(
    request: Request,
    container: Container,
    ctx: MemoryExecutionContext,
    *,
    key: str,
    payload: Any,
    handler: Handler,
) -> JSONResponse:
    """Execute ``handler`` inside a Unit of Work exactly once per (tenant, key, payload).

    ``handler`` returns ``(status, body, after_commit)``; the body is what later retries will
    receive verbatim. ``after_commit`` runs only after a successful commit.
    """
    idem: IdempotencyService = container.services["idempotency"]
    request_hash = idem.request_hash(payload)
    cached = await idem.lookup_cached(ctx.tenant_id, key, request_hash)
    if cached is not None:
        return JSONResponse(
            status_code=cached.status, content=cached.body, headers={"Idempotent-Replayed": "true"}
        )

    uow_factory = container.services["uow_factory"]
    async with uow_factory() as uow:
        replay = await idem.begin(uow.idempotency, ctx.tenant_id, key, request_hash)
        if replay is not None:
            return JSONResponse(
                status_code=replay.status,
                content=replay.body,
                headers={"Idempotent-Replayed": "true"},
            )
        status, body, after_commit = await handler(uow)
        await idem.complete(uow.idempotency, ctx.tenant_id, key, status=status, body=body)
        await uow.commit()
    await idem.warm_cache(ctx.tenant_id, key, request_hash, status=status, body=body)
    if after_commit is not None:
        await after_commit()
    return JSONResponse(status_code=status, content=body)
