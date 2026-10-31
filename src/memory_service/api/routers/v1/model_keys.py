"""Register/rotate/revoke model endpoints at agent and tenant level; never return a key.
A call resolves the acting agent's endpoint, else the tenant's, else the service's own.

An endpoint is ``trellis.contracts.Endpoint`` as JSON (its ``api_key`` a plain string on the
wire): where the model is, the route and auth by name, the model and what it accepts. It is
probed before it is stored (``GET /models`` with its own auth) and stored encrypted. The status
says which endpoint a call at that level would use, and why.

The tenant's model policy (which uses may run, whether reads are assisted, which model each
use calls) and its usage ledger are read and set here too."""

import hashlib
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, cast

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_service.api.deps import (
    AdministeredTenantDep,
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
    request_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import NO_CONTENT, default_idempotency_key, run_idempotent
from memory_service.config.settings import ALL_LLM_USES, LLMUse
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.provenance import require_permitted_model
from memory_service.modules.llm.credentials import ModelCredentials, agent_identity, sealed
from memory_service.modules.llm.endpoints import EndpointResolver, EndpointSource
from memory_service.modules.llm.policies import LLMUsage, ModelPolicies
from memory_service.modules.llm.policy import DEFAULT_ACCESS
from memory_service.ports.credentials import (
    EndpointProbe,
    ModelIdentity,
    ResolvedCredential,
    StoredCredential,
    tenant_identity,
)
from memory_service.ports.llm import StoredPolicy
from trellis.contracts import Endpoint, ModelProfile

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 409, 422, 503)
#: reads never conflict
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)
#: the usage ledger's default window and the widest one a read may ask for
USAGE_DEFAULT_DAYS = 30
USAGE_MAX_DAYS = 366

_ENDPOINT_EXAMPLE = {
    "base_url": Endpoint.model_fields["base_url"].examples[0],  # type: ignore[index]
    "gateway": "openai-compatible",
    "auth": "bearer",
    "model": "gpt-4.1-mini",
    "api_key": "sk-example",
    "profile": {"reasoning_output": False, "temperature": True},
}
_ENDPOINT = (
    "The endpoint (trellis.contracts.Endpoint): base_url (http or https, the API root), "
    "gateway (openai-compatible or bifrost), auth (bearer, google or aws), model, api_key "
    "(the bearer key; stored encrypted and never returned) and profile (what the model "
    "accepts). The service must reach base_url itself: it is probed before it is stored."
)


def _model_named(endpoint: Endpoint) -> Endpoint:
    if endpoint.model is not None:
        require_permitted_model(endpoint.model)
    return endpoint


class AgentKeyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"scope": {"agent_id": "research"}, "endpoint": _ENDPOINT_EXAMPLE}]
        },
    )

    scope: ScopeBody = Field(description="The agent the endpoint is for: agent_id is required.")
    endpoint: Endpoint = Field(description=_ENDPOINT)

    _permitted = field_validator("endpoint")(_model_named)


class ModelKeyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"endpoint": _ENDPOINT_EXAMPLE}]}
    )

    endpoint: Endpoint = Field(description=_ENDPOINT)

    _permitted = field_validator("endpoint")(_model_named)


class EndpointInfo(BaseModel):
    """A stored endpoint without its key."""

    base_url: str = Field(description="The API root the service calls.")
    gateway: str = Field(description="The route: openai-compatible or bifrost.")
    auth: str = Field(description="The auth strategy: bearer, google or aws.")
    model: str | None = Field(description="The model it names; null: the policy's or auto.")
    has_api_key: bool = Field(description="Whether a bearer key is stored (never returned).")
    profile: ModelProfile = Field(description="What the model accepts.")


class AgentKeyStatus(BaseModel):
    registered: bool = Field(
        description="true: an endpoint (or a revocation of one) is stored at this level."
    )
    revoked: bool = Field(
        description="true: the stored endpoint was revoked; calls at this level are refused "
        "rather than falling back to another level's endpoint or the service's."
    )
    revision: int = Field(
        description="How many times the endpoint was set or revoked (0: never); it changes "
        "on every rotation."
    )
    updated_at: datetime | None = Field(
        default=None, description="When it was last set or revoked; null when never."
    )
    endpoint: EndpointInfo | None = Field(
        default=None, description="The endpoint stored at this level, without its key."
    )
    source: EndpointSource = Field(
        description="Which endpoint a model call at this level uses now: agent (the agent's "
        "registration), tenant (the tenant's), service (the service's own LLM_* endpoint) "
        "or none (no model call is made)."
    )
    reason: str = Field(description="Why that source, in a sentence.")


def _info(endpoint: Endpoint | None) -> EndpointInfo | None:
    if endpoint is None:
        return None
    return EndpointInfo(
        base_url=endpoint.base_url,
        gateway=endpoint.gateway,
        auth=endpoint.auth,
        model=endpoint.model,
        has_api_key=endpoint.api_key is not None,
        profile=endpoint.profile,
    )


async def _status(
    container,  # type: ignore[no-untyped-def]
    identity: ModelIdentity,
    record: StoredCredential | None,
    *,
    stored: Endpoint | None = None,
) -> AgentKeyStatus:
    """``record``'s status and where a call at ``identity`` goes. ``stored``: the endpoint
    just written in this unit of work, which a fresh read would not see yet."""
    if stored is not None and record is not None:
        written = ResolvedCredential(stored, record.revision, record.identity)
        choice = await EndpointResolver(container.settings.llm, _Written(written)).resolve(identity)
        endpoint: Endpoint | None = stored
    else:
        resolver: EndpointResolver = container.services["llm_endpoints"]
        choice = await resolver.resolve(identity)
        endpoint = _service(container).endpoint_of(record) if record is not None else None
    return AgentKeyStatus(
        registered=record is not None,
        revoked=record is not None and record.ciphertext is None,
        revision=record.revision if record else 0,
        updated_at=record.updated_at if record else None,
        endpoint=_info(endpoint),
        source=choice.source,
        reason=choice.reason,
    )


class _Written:
    """The endpoint just registered, as the resolver's registered source reads it."""

    def __init__(self, written: ResolvedCredential) -> None:
        self.written = written

    async def resolve(self, identity: ModelIdentity) -> ResolvedCredential | None:
        return self.written

    async def confirm(self, identity: ModelIdentity, resolved: ResolvedCredential | None) -> None:
        return None


def _service(container) -> ModelCredentials:  # type: ignore[no-untyped-def]
    return container.services["model_credentials"]


async def _probe(container, endpoint: Endpoint) -> None:  # type: ignore[no-untyped-def]
    probe: EndpointProbe = container.services["endpoint_probe"]
    await probe(endpoint)


def _digest(endpoint: Endpoint) -> str:
    return hashlib.sha256(sealed(endpoint).get_secret_value().encode()).hexdigest()


# --------------------------------------------------------------------------- the agent's own


@router.get(
    "/agents/model-key",
    response_model=AgentKeyStatus,
    tags=["agents"],
    responses=_READ_ERRORS,
    summary="Read the acting agent's model endpoint and which endpoint its calls use, "
    "without its key",
)
async def key_status(container: ContainerDep, ctx: HeaderContextDep) -> AgentKeyStatus:
    identity = agent_identity(ctx)
    return await _status(container, identity, await _service(container).metadata_for(identity))


@router.put(
    "/agents/model-key",
    response_model=AgentKeyStatus,
    tags=["agents"],
    responses=_ERRORS,
    summary="Register or rotate the acting agent's model endpoint (stored encrypted)",
    description="The harness registers the endpoint its agent's model is on, so the service "
    "calls the same model for that agent's memory. 422 when the service cannot reach the "
    "endpoint (`GET {base_url}/models` with its auth), e.g. a loopback address only the "
    "harness's machine resolves.",
)
async def set_key(
    request: Request, body: AgentKeyRequest, container: ContainerDep, _: ServicePrincipalDep
):
    ctx = build_context(request, container, body.scope)
    identity = agent_identity(ctx)
    await _probe(container, body.endpoint)

    async def write(uow):  # type: ignore[no-untyped-def]
        record = await _service(container).set(uow, ctx, body.endpoint)
        status = await _status(container, identity, record, stored=body.endpoint)
        return 200, status.model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key
        or default_idempotency_key(ctx, "agent-key-put", ctx.request_id),
        payload={
            "principal": ctx.principal_id,
            "action": "rotate",
            "digest": _digest(body.endpoint),
        },
        handler=write,
    )


@router.delete(
    "/agents/model-key",
    status_code=204,
    tags=["agents"],
    responses=_ERRORS,
    summary="Revoke the acting agent's model endpoint and invalidate assisted read caches",
    description="204: the endpoint is revoked (a revocation tombstone, so the agent never "
    "borrows the tenant's or the service's); `GET /v1/agents/model-key` reads the status "
    "afterwards.",
)
async def revoke_key(request: Request, container: ContainerDep, ctx: HeaderContextDep):
    async def write(uow):  # type: ignore[no-untyped-def]
        await _service(container).set(uow, ctx, None)
        return NO_CONTENT, {}, None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key
        or default_idempotency_key(ctx, "agent-key-delete", ctx.request_id),
        payload={"principal": ctx.principal_id, "action": "revoke"},
        handler=write,
    )


# --------------------------------------------------------------------------- the tenant's


async def _put_level(
    request: Request,
    container,
    tenant_id: str,
    identity: ModelIdentity,
    endpoint: Endpoint | None,
    action: str,
):  # type: ignore[no-untyped-def]
    ctx = request_context(request, tenant_id)
    if endpoint is not None:
        await _probe(container, endpoint)

    async def write(uow):  # type: ignore[no-untyped-def]
        record = await _service(container).set_for(uow, identity, endpoint)
        if endpoint is None:
            return NO_CONTENT, {}, None
        status = await _status(container, identity, record, stored=endpoint)
        return 200, status.model_dump(mode="json"), None

    payload = {"principal": identity.principal_id, "action": action}
    if endpoint is not None:
        payload["digest"] = _digest(endpoint)
    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key
        or default_idempotency_key(ctx, f"model-key-{action}", ctx.request_id),
        payload=payload,
        handler=write,
    )


@router.get(
    "/model-key",
    response_model=AgentKeyStatus,
    tags=["tenancy"],
    responses=_READ_ERRORS,
    summary="Read the tenant's model endpoint (used by agents without one of their own) and "
    "which endpoint its calls use, without its key",
)
async def tenant_key_status(
    container: ContainerDep, tenant_id: AdministeredTenantDep
) -> AgentKeyStatus:
    identity = tenant_identity(tenant_id)
    return await _status(container, identity, await _service(container).metadata_for(identity))


@router.put(
    "/model-key",
    response_model=AgentKeyStatus,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Register or rotate the tenant's model endpoint (stored encrypted)",
    description="422 when the service cannot reach the endpoint (`GET {base_url}/models` "
    "with its auth).",
)
async def set_tenant_key(
    request: Request,
    body: ModelKeyRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
):
    return await _put_level(
        request, container, tenant_id, tenant_identity(tenant_id), body.endpoint, "rotate"
    )


@router.delete(
    "/model-key",
    status_code=204,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Revoke the tenant's model endpoint",
    description="204: the endpoint is revoked; `GET /v1/model-key` reads the status afterwards.",
)
async def revoke_tenant_key(
    request: Request, container: ContainerDep, tenant_id: AdministeredTenantDep
):
    return await _put_level(
        request, container, tenant_id, tenant_identity(tenant_id), None, "revoke"
    )


# --------------------------------------------------------------------------- policies


class ModelPolicyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "uses": ["contextual_extraction", "summaries", "grounding_judge"],
                    "read_assist": False,
                    "models": {"grounding_judge": "gemini/gemini-3.8-flash"},
                }
            ]
        },
    )

    uses: list[LLMUse] = Field(
        max_length=len(ALL_LLM_USES), description="what the model may be used for in this tenant"
    )
    read_assist: bool = Field(
        description="whether reads (/v1/context, /v1/recall, /v1/verify, /v1/graph/entities) "
        "consult the model"
    )
    models: dict[LLMUse, str] = Field(
        default_factory=dict,
        description="the model a use calls, as the endpoint names it; a use not named here "
        "calls the service's default for it",
    )

    @field_validator("models")
    @classmethod
    def _permitted(cls, value: dict[LLMUse, str]) -> dict[LLMUse, str]:
        for model in value.values():
            require_permitted_model(model)
        return value


class ModelPolicyStatus(BaseModel):
    stored: bool = Field(
        description="false: the tenant has set no policy, so the default applies (every use, "
        "reads assisted, the default model per use)"
    )
    uses: list[LLMUse] = Field(
        description="What the model may be called for in this tenant "
        "(contextual_extraction, summaries, grounding_judge, ...)."
    )
    read_assist: bool = Field(
        description="Whether reads (/v1/context, /v1/recall, /v1/verify, "
        "/v1/graph/entities) consult the model."
    )
    models: dict[str, str] = Field(
        description="The model each use calls, as the endpoint names it; a use not named "
        "calls the service's default."
    )
    revision: int = Field(description="How many times the policy was set (0: never).")
    updated_at: datetime | None = Field(
        default=None, description="When it was last set; null when never."
    )


def _policy_status(stored: StoredPolicy | None) -> ModelPolicyStatus:
    if stored is None:
        return ModelPolicyStatus(
            stored=False,
            uses=_uses(DEFAULT_ACCESS.uses),
            read_assist=DEFAULT_ACCESS.read_assist,
            models={},
            revision=0,
        )
    return ModelPolicyStatus(
        stored=True,
        uses=_uses(stored.uses),
        read_assist=stored.read_assist,
        models=dict(stored.models),
        revision=stored.revision,
        updated_at=stored.updated_at,
    )


def _uses(uses: frozenset[str]) -> list[LLMUse]:
    return cast("list[LLMUse]", sorted(uses))


def _policies(container) -> ModelPolicies:  # type: ignore[no-untyped-def]
    return container.services["model_policies"]


@router.get(
    "/model-key/policy",
    response_model=ModelPolicyStatus,
    tags=["tenancy"],
    responses=_READ_ERRORS,
    summary="Read the tenant's model policy: uses, read assistance, the model per use",
)
async def tenant_policy(
    container: ContainerDep, tenant_id: AdministeredTenantDep
) -> ModelPolicyStatus:
    return _policy_status(await _policies(container).get(tenant_id))


@router.put(
    "/model-key/policy",
    response_model=ModelPolicyStatus,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Set the tenant's model policy",
)
async def set_tenant_policy(
    request: Request,
    body: ModelPolicyRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
):
    ctx = request_context(request, tenant_id)

    async def write(uow):  # type: ignore[no-untyped-def]
        stored = await _policies(container).set(
            uow,
            tenant_id,
            uses=body.uses,
            read_assist=body.read_assist,
            models={str(use): model for use, model in body.models.items()},
        )
        return 200, _policy_status(stored).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key
        or default_idempotency_key(ctx, "model-policy-put", ctx.request_id),
        payload=body.model_dump(mode="json"),
        handler=write,
    )


# --------------------------------------------------------------------------- usage


class UsageDayOut(BaseModel):
    day: date = Field(description="The day (UTC, YYYY-MM-DD).")
    use: LLMUse = Field(
        description="What the model was called for, as the tenant policy names it "
        "(contextual_extraction, summaries, grounding_judge, ...)."
    )
    tokens: int = Field(
        description="Model tokens spent that day for that use (prompt and completion)."
    )
    calls: int = Field(description="Model calls made that day for that use.")


class ModelUsageResponse(BaseModel):
    since: date = Field(description="The first day of the window (UTC).")
    until: date = Field(description="The last day of the window (UTC).")
    days: list[UsageDayOut] = Field(
        description="One row per day and use that saw a call, oldest first."
    )


@router.get(
    "/model-key/usage",
    response_model=ModelUsageResponse,
    tags=["tenancy"],
    responses=_READ_ERRORS,
    summary="The tenant's model tokens and calls per day and use",
)
async def tenant_usage(
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    since: Annotated[date | None, Query(description="first day (UTC); default 30 days ago")] = None,
    until: Annotated[date | None, Query(description="last day (UTC); default today")] = None,
) -> ModelUsageResponse:
    until = until or datetime.now(UTC).date()
    since = since or until - timedelta(days=USAGE_DEFAULT_DAYS - 1)
    if since > until or (until - since).days >= USAGE_MAX_DAYS:
        raise ValidationFailed(
            f"since..until must be an ordered range of at most {USAGE_MAX_DAYS} days"
        )
    usage: LLMUsage = container.services["llm_usage"]
    days = await usage.between(tenant_id, since, until)
    return ModelUsageResponse(
        since=since,
        until=until,
        days=[
            UsageDayOut(day=d.day, use=cast("LLMUse", d.use), tokens=d.tokens, calls=d.calls)
            for d in days
        ],
    )
