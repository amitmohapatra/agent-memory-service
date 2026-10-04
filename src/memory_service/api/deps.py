"""FastAPI dependencies: container access, service authentication, execution context."""

from __future__ import annotations

from typing import Annotated, Any, cast

from fastapi import Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memory_service.api.headers import require_one_value
from memory_service.api.params import (
    AgentGroupIdQuery,
    AgentIdQuery,
    AgentRunIdQuery,
    ParentAgentRunIdQuery,
    SessionIdQuery,
    TaskIdQuery,
    ThreadIdPath,
    ThreadIdQuery,
    WorkIdQuery,
)
from memory_service.api.schemas.tenancy import KeySelfResponse, KeySelfRole
from memory_service.api.validation import CustomMetadata
from memory_service.application.container import Container
from memory_service.config.constants import HEADERS
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import (
    AuthorizationFailed,
    NotFound,
    ScopeDenied,
    ValidationFailed,
)
from memory_service.domain.tenancy import (
    ANY_PRINCIPAL,
    PLATFORM_SCOPE,
    KeyRole,
    is_valid_tenant_id,
)
from memory_service.modules.auth.authentication import ServiceAuthenticator, ServicePrincipal
from memory_service.observability.logging import bind_log_context
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span


class ScopeBody(BaseModel):
    """Lineage part of the scope, sent in request bodies by the SDK.

    Security fields (tenant/workspace/user/groups) travel in trusted headers. When they are
    also present in the body they must match the headers exactly.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str | None = Field(default=None, examples=["acme"])
    workspace_id: str | None = Field(default=None, examples=["ws-finance"])
    user_id: str | None = Field(default=None, examples=["u-123"])
    thread_id: str | None = Field(default=None, examples=["thr_01J8Z"])
    session_id: str | None = Field(default=None, examples=["ses_01J8Z"])
    turn_id: str | None = Field(default=None, examples=["trn_01J8Z"])
    work_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = Field(default=None, examples=["research"])
    agent_group_id: str | None = None
    agent_run_id: str | None = None
    parent_agent_run_id: str | None = None
    correlation_id: str | None = None
    custom_metadata: CustomMetadata = Field(default_factory=dict)


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


async def get_service_principal(request: Request, container: ContainerDep) -> ServicePrincipal:
    authenticator: ServiceAuthenticator = container.services["authenticator"]
    with span("auth"), stage_seconds.labels("auth").time():
        principal = await authenticator.authenticate(
            {k.lower(): v for k, v in request.headers.items()}
        )
    request.state.service_principal = principal
    return principal


ServicePrincipalDep = Annotated[ServicePrincipal, Depends(get_service_principal)]


def _header_scope(request: Request, container: Container) -> dict[str, Any]:
    h = request.headers
    return {
        "tenant_id": require_one_value(h, HEADERS.tenant),
        "workspace_id": require_one_value(h, HEADERS.workspace),
        "user_id": require_one_value(h, HEADERS.user),
    }


def _principal(request: Request) -> ServicePrincipal | None:
    return getattr(request.state, "service_principal", None)


def credential_claims(request: Request) -> dict[str, Any]:
    principal = _principal(request)
    return principal.claims if principal is not None else {}


def credential_mode(request: Request) -> str | None:
    principal = _principal(request)
    return principal.mode if principal is not None else None


def _require_tenant_matches_credential(
    request: Request, container: Container, tenant_id: str
) -> None:
    """Bind the asserted tenant to the credential asserting it, when configured to.

    In ``api_key`` mode the binding is unconditional: a key names its tenant, so the header
    is a claim to be checked against it, never the source of it. The platform key names no
    tenant and is refused here - onboarding tenants and acting for one are different powers.

    Outside ``api_key`` mode, authentication identifies the calling SERVICE - the principal
    carries no tenant of its own - while the tenant arrives in a header, and nothing compared
    the two. Every boundary below this point then works perfectly, on behalf of whichever
    tenant the caller claimed to be.
    One credential reaches every tenant on the deployment by changing one header.

    That is sound where the tenant is a constant a gateway stamps, which is one deployment
    per customer. On a SHARED deployment it makes the credential the entire boundary, so
    ``authentication.tenant_claim`` names the claim that must agree with the header.

    Fails closed: with the setting on, a credential that carries no such claim is refused
    rather than trusted, so switching a deployment to trusted_dev keys cannot quietly turn
    the check off.
    """
    if credential_mode(request) == "api_key":
        claims = credential_claims(request)
        if claims.get("role") == KeyRole.PLATFORM.value:
            raise ScopeDenied(
                "the platform key onboards tenants; it does not act for one",
                details={"field": HEADERS.tenant},
            )
        if claims.get("tenant") != tenant_id:
            raise ScopeDenied(
                "credential is not valid for this tenant", details={"field": HEADERS.tenant}
            )
        return
    claim = container.settings.authentication.tenant_claim
    if not claim:
        return
    principal: ServicePrincipal | None = getattr(request.state, "service_principal", None)
    if principal is None or str(principal.claims.get(claim) or "") != tenant_id:
        raise ScopeDenied(
            "credential is not valid for this tenant",
            details={"field": HEADERS.tenant},
        )


def _credential_scope(
    request: Request, container: Container, headers: dict[str, Any], body: ScopeBody
) -> tuple[str, str | None]:
    """The tenant and workspace this request acts in, bound to its credential.

    A key carries its tenant, so a caller holding one need not repeat it; when it does, the
    check still binds the two. A key bound to a workspace pins the workspace the same way.
    """
    claims = credential_claims(request) if credential_mode(request) == "api_key" else {}
    tenant_id = headers["tenant_id"] or body.tenant_id or claims.get("tenant")
    if not tenant_id:
        raise ValidationFailed(f"tenant_id is required ({HEADERS.tenant} header)")
    _require_tenant_matches_credential(request, container, tenant_id)
    workspace_id = headers["workspace_id"] or body.workspace_id
    bound_workspace = claims.get("workspace")
    if bound_workspace:
        if workspace_id not in (None, bound_workspace):
            raise ScopeDenied(
                "credential is bound to another workspace", details={"field": HEADERS.workspace}
            )
        workspace_id = bound_workspace
    return str(tenant_id), workspace_id


def _require_tenant_active(container: Container, tenant_id: str) -> None:
    """A suspended tenant is not served, whatever credential kind the caller holds. The
    verifier already refuses its keys; this covers ``jwt`` and ``trusted_dev`` callers from
    the in-process registry, so it costs no store read."""
    registry = container.services.get("tenant_registry")
    if registry is not None and registry.is_suspended(tenant_id):
        raise AuthorizationFailed("tenant is suspended", details={"tenant_id": tenant_id})


def _require_may_act_as(request: Request, user_id: str | None) -> None:
    """A key restricted to some principals (``may_act_as``) acts for no other user."""
    claims = credential_claims(request) if credential_mode(request) == "api_key" else {}
    allowed = claims.get("may_act_as")
    if user_id is None or allowed is None or ANY_PRINCIPAL in allowed:
        return
    if f"user:{user_id}" not in allowed:
        raise ScopeDenied("this key may not act for that user", details={"field": HEADERS.user})


def request_context(request: Request, tenant_id: str) -> MemoryExecutionContext:
    """A bare execution context for administrative writes: the tenant acted on and the
    request's correlation ids, nothing about users or threads."""
    return MemoryExecutionContext(
        tenant_id=tenant_id,
        request_id=request.state.request_id,
        correlation_id=request.state.correlation_id,
        trace_id=request.state.trace_id,
    )


def build_context(
    request: Request, container: Container, body_scope: ScopeBody | None
) -> MemoryExecutionContext:
    """Merge trusted headers with body lineage into an immutable execution context."""
    headers = _header_scope(request, container)
    body = body_scope or ScopeBody()
    for field in ("tenant_id", "workspace_id", "user_id"):
        header_value = headers[field]
        body_value = getattr(body, field)
        if body_value is not None and header_value is not None and body_value != header_value:
            raise ValidationFailed(
                f"{field} in body does not match trusted header", details={"field": field}
            )
    tenant_id, workspace_id = _credential_scope(request, container, headers, body)
    _require_tenant_active(container, tenant_id)
    user_id = headers["user_id"] or body.user_id
    _require_may_act_as(request, user_id)
    try:
        ctx = MemoryExecutionContext(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            user_id=user_id,
            thread_id=body.thread_id,
            session_id=body.session_id,
            turn_id=body.turn_id,
            work_id=body.work_id,
            task_id=body.task_id,
            agent_id=body.agent_id,
            agent_group_id=body.agent_group_id,
            agent_run_id=body.agent_run_id,
            parent_agent_run_id=body.parent_agent_run_id,
            request_id=request.state.request_id,
            correlation_id=body.correlation_id or request.state.correlation_id,
            # never the body's: the trace id in headers, logs, rows and problems is one id
            trace_id=request.state.trace_id,
            custom_metadata=body.custom_metadata,
        )
    except ValidationError as exc:
        raise ValidationFailed(
            "Invalid execution context",
            details={"errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]},
        ) from exc
    bind_log_context(**ctx.log_fields())
    request.state.context = ctx
    # the response echoes the id the logs carry, whichever of header or body named it
    request.state.correlation_id = ctx.correlation_id
    return ctx


def _lineage_but_thread(
    session_id: SessionIdQuery = None,
    work_id: WorkIdQuery = None,
    task_id: TaskIdQuery = None,
    agent_id: AgentIdQuery = None,
    agent_group_id: AgentGroupIdQuery = None,
    agent_run_id: AgentRunIdQuery = None,
    parent_agent_run_id: ParentAgentRunIdQuery = None,
) -> ScopeBody:
    return ScopeBody(
        session_id=session_id,
        work_id=work_id,
        task_id=task_id,
        agent_id=agent_id,
        agent_group_id=agent_group_id,
        agent_run_id=agent_run_id,
        parent_agent_run_id=parent_agent_run_id,
    )


_LineageButThread = Annotated[ScopeBody, Depends(_lineage_but_thread)]


def lineage_query(base: _LineageButThread, thread_id: ThreadIdQuery = None) -> ScopeBody:
    """The lineage of a GET/DELETE route (no body) from its optional query parameters, so
    an agent reads and forgets with the same identity it wrote with."""
    return base.model_copy(update={"thread_id": thread_id})


def thread_lineage(base: _LineageButThread, thread_id: ThreadIdPath) -> ScopeBody:
    """:func:`lineage_query` for a route under ``/threads/{thread_id}``: the path names the
    thread the call acts in."""
    return base.model_copy(update={"thread_id": thread_id})


LineageDep = Annotated[ScopeBody, Depends(lineage_query)]


async def get_header_context(
    request: Request, container: ContainerDep, _: ServicePrincipalDep, lineage: LineageDep
) -> MemoryExecutionContext:
    """Context for GET/DELETE routes (no body): security fields from trusted headers, the
    lineage (thread, work, agent run, agent group) from optional query parameters."""
    return build_context(request, container, lineage)


async def get_thread_context(
    request: Request,
    container: ContainerDep,
    _: ServicePrincipalDep,
    lineage: Annotated[ScopeBody, Depends(thread_lineage)],
) -> MemoryExecutionContext:
    """:func:`get_header_context` for the routes of one thread (its id is the path's)."""
    return build_context(request, container, lineage)


ThreadContextDep = Annotated[MemoryExecutionContext, Depends(get_thread_context)]
HeaderContextDep = Annotated[MemoryExecutionContext, Depends(get_header_context)]


def key_self_of(principal: ServicePrincipal) -> KeySelfResponse:
    """What ``GET /v1/keys/self`` says about the authenticated caller."""
    claims = principal.claims
    if principal.mode == "api_key" and is_platform(principal):
        return KeySelfResponse(
            key_id=PLATFORM_SCOPE,
            tenant_id=None,
            principal=PLATFORM_SCOPE,
            role=KeyRole.PLATFORM.value,
            may_act_as=[ANY_PRINCIPAL],
        )
    if principal.mode == "api_key":
        return KeySelfResponse(
            key_id=str(claims["key_id"]),
            tenant_id=str(claims["tenant"]),
            principal=principal.service_id,
            role=cast("KeySelfRole", str(claims["role"])),
            may_act_as=list(claims.get("may_act_as") or []),
        )
    # a development key or an issuer's token names its tenant per request
    return KeySelfResponse(
        key_id=principal.service_id,
        tenant_id=None,
        principal=principal.service_id,
        role=cast("KeySelfRole", principal.mode),
        may_act_as=[ANY_PRINCIPAL],
    )


def is_platform(principal: ServicePrincipal) -> bool:
    """The platform is the bootstrap secret and nothing else: minted only by ``api_key``
    authentication, never by an issuer's token - whatever its ``sub`` or ``role`` say."""
    return principal.mode == "api_key" and principal.service_id == PLATFORM_SCOPE


def _has_role(principal: ServicePrincipal, roles: tuple[KeyRole, ...]) -> bool:
    """Administration is an ``api_key``-mode power: keys carry the role the service gave
    them. ``jwt`` is the per-customer deployment - the issuer's token is the calling service
    and administers nothing here. ``trusted_dev`` keys are the laptop's and may do anything,
    which is one of the reasons deployed environments refuse them."""
    if principal.mode == "trusted_dev":
        return True
    if principal.mode != "api_key":
        return False
    role = principal.claims.get("role")
    if role == KeyRole.PLATFORM.value:
        return is_platform(principal) and KeyRole.PLATFORM in roles
    return role in {r.value for r in roles}


def is_tenant_administrator(principal: ServicePrincipal) -> bool:
    """Whether the credential is a tenant's administrator (or the platform's)."""
    return _has_role(principal, (KeyRole.ADMIN, KeyRole.PLATFORM))


def ensure_role(principal: ServicePrincipal, *roles: KeyRole) -> ServicePrincipal:
    """The principal, when its credential holds one of ``roles``; else 403."""
    if not _has_role(principal, roles):
        raise AuthorizationFailed(
            "this credential may not perform that administration",
            details={"required_role": [r.value for r in roles]},
        )
    return principal


def require_role(*roles: KeyRole) -> Any:
    async def dependency(principal: ServicePrincipalDep) -> ServicePrincipal:
        return ensure_role(principal, *roles)

    return Depends(dependency)


#: The platform operator: onboards tenants, issues their admin keys, reads the audit trail.
PlatformDep = Annotated[ServicePrincipal, require_role(KeyRole.PLATFORM)]
#: A tenant's administrator (or the platform, for its own tenant-scoped calls).
TenantAdminDep = Annotated[ServicePrincipal, require_role(KeyRole.ADMIN, KeyRole.PLATFORM)]


def administered_tenant(request: Request, principal: ServicePrincipal, container: Container) -> str:
    """The tenant an administrative call acts on: the key's own tenant, or - for a
    credential that names none, the platform key and development keys - the header."""
    claimed = principal.claims.get("tenant") if principal.mode == "api_key" else None
    named = require_one_value(request.headers, HEADERS.tenant)
    if named and not is_valid_tenant_id(named):
        raise ValidationFailed("invalid tenant_id", details={"field": HEADERS.tenant})
    if claimed and named and named != claimed:
        # An admin key names its tenant; a header naming another one is a mistake or an
        # attempt, and either deserves a refusal rather than silently acting on the claim.
        raise ScopeDenied(
            "credential is not valid for this tenant", details={"field": HEADERS.tenant}
        )
    tenant = claimed or named
    if not tenant:
        raise ValidationFailed(f"tenant_id is required ({HEADERS.tenant} header)")
    if not is_platform(principal):
        # a suspended tenant's own administrators are suspended with it; the platform is
        # who resumes it, so it is not stopped here
        _require_tenant_active(container, str(tenant))
    return str(tenant)


async def get_administered_tenant(
    request: Request,
    principal: TenantAdminDep,
    container: ContainerDep,
    tenant_header: Annotated[
        str | None,
        Header(
            alias=HEADERS.tenant,
            description="The tenant to administer. Required for the platform key and "
            "development keys; an admin key names its own tenant and this must agree with it.",
        ),
    ] = None,
) -> str:
    """``administered_tenant`` as a dependency, which also puts the header in the OpenAPI
    document. The tenant must exist when the header alone names it: the platform's typo
    must not create rows for a tenant nobody onboarded."""
    return await existing_administered_tenant(request, principal, container)


async def existing_administered_tenant(
    request: Request, principal: ServicePrincipal, container: Container
) -> str:
    """:func:`administered_tenant`, which must exist when the header alone names it."""
    tenant_id = administered_tenant(request, principal, container)
    if principal.claims.get("tenant") != tenant_id:
        async with container.services["uow_factory"]() as uow:
            if await uow.tenants.get(tenant_id) is None:
                raise NotFound("Tenant not found")
    return tenant_id


AdministeredTenantDep = Annotated[str, Depends(get_administered_tenant)]
