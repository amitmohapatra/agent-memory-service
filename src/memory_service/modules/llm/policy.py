"""Request- and job-local model binding: whose key pays, what its policy allows, and whether
this operation may consult a model at all.

``LLMAssist.bound`` resolves the binding once per request or job (one indexed read of the key
and policy hierarchy) and sets it here; ``LLMAssist.wants`` reads it synchronously.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from memory_service.config.constants import LLM, LLMTuning
from memory_service.config.settings import ALL_LLM_USES, OPT_IN_LLM_USES
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Document
from memory_service.ports.credentials import ModelIdentity


@dataclass(frozen=True)
class ModelAccess:
    """What the tenant's policy allows, whether a registered key can pay for it, and the model
    the policy names for a use (the service's default for the use otherwise)."""

    uses: frozenset[str]
    read_assist: bool
    has_key: bool
    models: Mapping[str, str] = field(default_factory=dict)


#: No policy row at any level: every use but the opt-in ones, and reads assisted. Without a
#: key this allows nothing in automatic mode, because nothing can pay (``LLMAssist.wants``).
DEFAULT_ACCESS = ModelAccess(
    uses=frozenset(ALL_LLM_USES) - OPT_IN_LLM_USES, read_assist=True, has_key=False
)


@dataclass(frozen=True)
class ModelBinding:
    identity: ModelIdentity
    access: ModelAccess


_ALLOWED: ContextVar[bool] = ContextVar("memory_model_calls_allowed", default=True)
_BINDING: ContextVar[ModelBinding | None] = ContextVar("memory_model_binding", default=None)


def model_calls_allowed() -> bool:
    return _ALLOWED.get()


def current_binding() -> ModelBinding | None:
    return _BINDING.get()


def model_for(use: str, tuning: LLMTuning = LLM) -> str:
    """The model ``use`` calls: the one the bound tenant's policy names for it, else the
    service's (``tuning``; ``auto`` is discovered through the gateway per call)."""
    binding = _BINDING.get()
    named = binding.access.models.get(use) if binding is not None else None
    return named or (tuning.fast_model if use in tuning.fast_uses else tuning.model)


def current_model_identity() -> ModelIdentity | None:
    binding = _BINDING.get()
    return binding.identity if binding is not None else None


def identity_of(ctx: MemoryExecutionContext) -> ModelIdentity:
    """An authenticated request's owner: its principal (resolved through its agent, then its
    tenant)."""
    return ModelIdentity(ctx.tenant_id, ctx.principal_id)


def document_identity(document: Document) -> ModelIdentity:
    """A document's model work is owned by whoever uploaded it."""
    return ModelIdentity(document.tenant_id, document.model_principal)


@contextmanager
def bind(binding: ModelBinding) -> Iterator[None]:
    """Set a resolved binding (``LLMAssist.bound`` is the way to obtain one)."""
    token = _BINDING.set(binding)
    try:
        yield
    finally:
        _BINDING.reset(token)


@contextmanager
def model_call_policy(allow: bool) -> Iterator[None]:
    """A nested operation may narrow permission, never override an outer prohibition."""
    token = _ALLOWED.set(allow and _ALLOWED.get())
    try:
        yield
    finally:
        _ALLOWED.reset(token)
