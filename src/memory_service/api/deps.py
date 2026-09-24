"""FastAPI dependencies: container access, service authentication, execution context."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memory_service.api.validation import CustomMetadata
from memory_service.application.container import Container
from memory_service.config.constants import AUTHORIZATION, HEADERS
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ScopeDenied, ValidationFailed
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
    group_ids: list[str] = Field(default_factory=list, examples=[["legal", "finance"]])
    thread_id: str | None = Field(default=None, examples=["thr_01J8Z"])
    session_id: str | None = Field(default=None, examples=["ses_01J8Z"])
    turn_id: str | None = Field(default=None, examples=["trn_01J8Z"])
    work_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = Field(default=None, examples=["research"])
    agent_group_id: str | None = None
    agent_run_id: str | None = None
    parent_agent_run_id: str | None = None
    trace_id: str | None = None
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
    groups_raw = h.get(HEADERS.groups, "")
    return {
        "tenant_id": h.get(HEADERS.tenant),
        "workspace_id": h.get(HEADERS.workspace),
        "user_id": h.get(HEADERS.user),
        # Capped: each group becomes a store-side read key, and an uncapped header let one
        # request assert as many as it liked. The body half of this was closed already; the
        # header half is the same shape and was missed because only the body was examined.
        "group_ids": [g.strip() for g in groups_raw.split(",") if g.strip()][
            : AUTHORIZATION.max_asserted_groups
        ],
    }


def _require_tenant_matches_credential(
    request: Request, container: Container, tenant_id: str
) -> None:
    """Bind the asserted tenant to the credential asserting it, when configured to.

    Authentication identifies the calling SERVICE - ``ServicePrincipal`` carries no tenant -
    while the tenant arrives in a header, and nothing compared the two. Every boundary below
    this point then works perfectly, on behalf of whichever tenant the caller claimed to be.
    One credential reaches every tenant on the deployment by changing one header.

    That is sound where the tenant is a constant a gateway stamps, which is one deployment
    per customer. On a SHARED deployment it makes the credential the entire boundary, so
    ``authentication.tenant_claim`` names the claim that must agree with the header.

    Fails closed: with the setting on, a credential that carries no such claim is refused
    rather than trusted, so switching a deployment to trusted_dev keys cannot quietly turn
    the check off.
    """
    claim = container.settings.authentication.tenant_claim
    if not claim:
        return
    principal: ServicePrincipal | None = getattr(request.state, "service_principal", None)
    if principal is None or str(principal.claims.get(claim) or "") != tenant_id:
        raise ScopeDenied(
            "credential is not valid for this tenant",
            details={"field": HEADERS.tenant},
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
    tenant_id = headers["tenant_id"] or body.tenant_id
    if not tenant_id:
        raise ValidationFailed("tenant_id is required (X-Memory-Tenant header)")
    _require_tenant_matches_credential(request, container, tenant_id)
    # Groups are a security field, and they were the one that was merged instead of checked.
    # A group id is not a hint: ScopeResolver folds it straight into the AuthorizedScope and
    # VisibilitySpecification turns it into a ``group:{tenant}/{g}`` read key, so a body that
    # could add one could read every GROUP-visibility row for it without any membership ever
    # being checked - and the list is uncapped, so ids could be enumerated in bulk in a single
    # request. The tenant prefix held, so nothing crossed a tenant; everything inside one was
    # reachable. ADR 0005 and this class's own docstring already say body security fields must
    # match the headers exactly, so this restores the documented contract rather than
    # inventing a rule: the header is authoritative, and a body that disagrees is refused
    # rather than quietly widened.
    header_groups = sorted(set(headers["group_ids"]))
    if body.group_ids and sorted(set(body.group_ids)) != header_groups:
        raise ValidationFailed(
            "group_ids in body does not match trusted header", details={"field": "group_ids"}
        )
    group_ids = header_groups
    try:
        ctx = MemoryExecutionContext(
            tenant_id=tenant_id,
            workspace_id=headers["workspace_id"] or body.workspace_id,
            user_id=headers["user_id"] or body.user_id,
            group_ids=group_ids,
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
            trace_id=body.trace_id or request.state.trace_id,
            custom_metadata=body.custom_metadata,
        )
    except ValidationError as exc:
        raise ValidationFailed(
            "Invalid execution context",
            details={"errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]},
        ) from exc
    bind_log_context(**ctx.log_fields())
    request.state.context = ctx
    return ctx


async def get_header_context(
    request: Request,
    container: ContainerDep,
    _: ServicePrincipalDep,
    thread_id: str | None = None,
    work_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    agent_group_id: str | None = None,
    agent_run_id: str | None = None,
    parent_agent_run_id: str | None = None,
) -> MemoryExecutionContext:
    """Context for GET/DELETE routes (no body): security fields from trusted headers, the
    lineage (thread, work, agent run, agent group) from optional query parameters so an
    agent reads and forgets with the same identity it wrote with."""
    return build_context(
        request,
        container,
        ScopeBody(
            thread_id=thread_id,
            work_id=work_id,
            task_id=task_id,
            agent_id=agent_id,
            agent_group_id=agent_group_id,
            agent_run_id=agent_run_id,
            parent_agent_run_id=parent_agent_run_id,
        ),
    )


HeaderContextDep = Annotated[MemoryExecutionContext, Depends(get_header_context)]
