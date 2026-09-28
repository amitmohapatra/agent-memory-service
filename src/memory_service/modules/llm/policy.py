"""Request-local permission to consult a generative model, independent of ingestion."""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar

from memory_service.domain.context import MemoryExecutionContext
from memory_service.ports.credentials import ModelIdentity

_ALLOWED: ContextVar[bool] = ContextVar("memory_model_calls_allowed", default=True)
_IDENTITY: ContextVar[ModelIdentity | None] = ContextVar("memory_model_identity", default=None)


def model_calls_allowed() -> bool:
    return _ALLOWED.get()


def current_model_identity() -> ModelIdentity | None:
    return _IDENTITY.get()


@contextmanager
def model_identity(
    tenant_id: str, principal_id: str, *, workspace_id: str | None = None
) -> Iterator[None]:
    """Bind an authenticated request or persisted source owner, never free-form metadata.
    ``workspace_id`` lets a call fall back to the team's key (ADR 0023)."""
    token = _IDENTITY.set(ModelIdentity(tenant_id, principal_id, workspace_id))
    try:
        yield
    finally:
        _IDENTITY.reset(token)


def model_identity_of(ctx: MemoryExecutionContext) -> AbstractContextManager[None]:
    """The binding for an authenticated request: its principal, falling back to its team."""
    return model_identity(ctx.tenant_id, ctx.principal_id, workspace_id=ctx.workspace_id)


@contextmanager
def model_call_policy(allow: bool) -> Iterator[None]:
    """A nested operation may narrow permission, never override an outer prohibition."""
    token = _ALLOWED.set(allow and _ALLOWED.get())
    try:
        yield
    finally:
        _ALLOWED.reset(token)
