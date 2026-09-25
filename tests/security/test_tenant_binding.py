"""A credential may act for the tenant it names, and on a shared deployment only that one.

Authentication identifies the calling SERVICE: ``ServicePrincipal`` carries service_id, mode
and claims, and no tenant. The tenant arrives in ``X-Memory-Tenant``. Nothing compared the
two, so every boundary below worked perfectly on behalf of whichever tenant the caller said
it was - one credential reached every tenant on the deployment by changing one header.

That is sound where a gateway stamps the tenant as a constant, which is one deployment per
customer. It is not sound when several product teams share one deployment, which is why
``authentication.tenant_claim`` exists.
"""

from __future__ import annotations

import pytest

from memory_service.api.deps import ScopeBody, build_context
from memory_service.domain.errors import ScopeDenied
from memory_service.modules.auth.authentication import ServicePrincipal

pytestmark = pytest.mark.security


class _State:
    """Request state. Only ``service_principal`` matters here; the rest is request lineage."""

    def __init__(self, principal: ServicePrincipal | None) -> None:
        self.service_principal = principal

    def __getattr__(self, name: str) -> str:
        return f"{name}_test"


class _Request:
    def __init__(self, tenant: str, principal: ServicePrincipal | None) -> None:
        self.headers = {"X-Memory-Tenant": tenant, "X-Memory-User": "u1"}
        self.state = _State(principal)


class _Container:
    def __init__(self, claim: str | None) -> None:
        from memory_service.config.settings import Settings

        self.settings = Settings()
        self.settings.authentication.tenant_claim = claim


def _principal(**claims: str) -> ServicePrincipal:
    return ServicePrincipal(service_id="svc", mode="jwt", claims=dict(claims))


def _build(tenant: str, principal: ServicePrincipal | None, claim: str | None) -> None:
    build_context(
        _Request(tenant, principal),  # type: ignore[arg-type]
        _Container(claim),  # type: ignore[arg-type]
        ScopeBody(),
    )


def test_unset_the_credential_may_assert_any_tenant() -> None:
    """The per-customer-deployment default, stated so it is a choice and not an oversight."""
    _build("arhaus", _principal(), claim=None)
    _build("someone-else", _principal(tenant="arhaus"), claim=None)


def test_set_the_header_must_agree_with_the_claim() -> None:
    _build("arhaus", _principal(tenant="arhaus"), claim="tenant")
    with pytest.raises(ScopeDenied):
        _build("another-team", _principal(tenant="arhaus"), claim="tenant")


def test_a_credential_without_the_claim_is_refused_not_trusted() -> None:
    """Fails closed: switching a deployment to keys that carry no claims cannot disable it."""
    with pytest.raises(ScopeDenied):
        _build("arhaus", _principal(), claim="tenant")
    with pytest.raises(ScopeDenied):
        _build("arhaus", None, claim="tenant")


def test_the_claim_is_compared_exactly() -> None:
    """No prefix match: 'arhaus' must not satisfy a request for 'arhaus-staging'."""
    with pytest.raises(ScopeDenied):
        _build("arhaus-staging", _principal(tenant="arhaus"), claim="tenant")
