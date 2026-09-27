"""Request-local permission to consult a generative model, independent of ingestion."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from memory_service.ports.credentials import ModelIdentity

_ALLOWED: ContextVar[bool] = ContextVar("memory_model_calls_allowed", default=True)
_IDENTITY: ContextVar[ModelIdentity | None] = ContextVar("memory_model_identity", default=None)


def model_calls_allowed() -> bool:
    return _ALLOWED.get()


def current_model_identity() -> ModelIdentity | None:
    return _IDENTITY.get()


@contextmanager
def model_identity(tenant_id: str, principal_id: str) -> Iterator[None]:
    """Bind an authenticated request or persisted source owner, never free-form metadata."""
    token = _IDENTITY.set(ModelIdentity(tenant_id, principal_id))
    try:
        yield
    finally:
        _IDENTITY.reset(token)


@contextmanager
def model_call_policy(allow: bool) -> Iterator[None]:
    """A nested operation may narrow permission, never override an outer prohibition."""
    token = _ALLOWED.set(allow and _ALLOWED.get())
    try:
        yield
    finally:
        _ALLOWED.reset(token)
