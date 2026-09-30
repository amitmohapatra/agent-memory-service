"""A key names its tenant; the request may agree with it and may not contradict it."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from starlette.datastructures import Headers

from memory_service.api.deps import ScopeBody, build_context
from memory_service.domain.errors import ScopeDenied, ValidationFailed
from memory_service.modules.auth.authentication import ServicePrincipal

pytestmark = pytest.mark.unit


class _State:
    def __init__(self, principal: ServicePrincipal | None) -> None:
        self.service_principal = principal

    def __getattr__(self, name: str) -> str:
        return f"{name}_test"


class _Request:
    def __init__(self, principal: ServicePrincipal | None, **headers: str) -> None:
        self.headers = Headers(headers)
        self.state = _State(principal)


class _Container:
    def __init__(self) -> None:
        from memory_service.config.settings import Settings

        self.settings = Settings(_env_file=None)
        self.services: dict = {}


def _key(tenant: str = "acme", role: str = "service", workspace: str | None = None):
    return ServicePrincipal(
        service_id="key:k1",
        mode="api_key",
        claims={"role": role, "tenant": tenant, "workspace": workspace, "key_id": "k1"},
    )


def _build(principal: ServicePrincipal | None, body: ScopeBody | None = None, **headers: str):
    return build_context(_Request(principal, **headers), _Container(), body or ScopeBody())  # type: ignore[arg-type]


def test_the_tenant_comes_from_the_key_when_the_caller_sends_none() -> None:
    ctx = _build(_key("acme"), **{"X-Trellis-User": "u1"})
    assert ctx.tenant_id == "acme" and ctx.user_id == "u1"


def test_a_header_may_agree_with_the_key_and_may_not_contradict_it() -> None:
    assert _build(_key("acme"), **{"X-Trellis-Tenant": "acme"}).tenant_id == "acme"
    with pytest.raises(ScopeDenied):
        _build(_key("acme"), **{"X-Trellis-Tenant": "globex"})
    with pytest.raises(ScopeDenied):
        _build(_key("acme"), ScopeBody(tenant_id="globex"))


def test_the_platform_key_acts_for_no_tenant() -> None:
    platform = ServicePrincipal(service_id="platform", mode="api_key", claims={"role": "platform"})
    with pytest.raises(ScopeDenied, match="platform key"):
        _build(platform, **{"X-Trellis-Tenant": "acme"})
    with pytest.raises(ValidationFailed):
        _build(platform)


def test_a_workspace_bound_key_pins_the_workspace() -> None:
    bound = _key("acme", workspace="finance")
    assert _build(bound).workspace_id == "finance"
    assert _build(bound, **{"X-Trellis-Workspace": "finance"}).workspace_id == "finance"
    with pytest.raises(ScopeDenied, match="another workspace"):
        _build(bound, **{"X-Trellis-Workspace": "legal"})


def test_development_keys_keep_the_header_semantics() -> None:
    dev = ServicePrincipal(service_id="dev:1", mode="trusted_dev", claims={})
    assert _build(dev, **{"X-Trellis-Tenant": "anything"}).tenant_id == "anything"


def test_an_admin_key_may_not_name_another_tenant_on_administration_routes() -> None:
    from memory_service.api.deps import administered_tenant

    c = _Container()
    admin = _key("acme", role="admin")
    assert administered_tenant(_Request(admin), admin, c) == "acme"  # type: ignore[arg-type]
    named = _Request(admin, **{"X-Trellis-Tenant": "acme"})
    assert administered_tenant(named, admin, c) == "acme"  # type: ignore[arg-type]
    with pytest.raises(ScopeDenied):
        administered_tenant(_Request(admin, **{"X-Trellis-Tenant": "globex"}), admin, c)  # type: ignore[arg-type]
    platform = ServicePrincipal(service_id="platform", mode="api_key", claims={"role": "platform"})
    by_header = _Request(platform, **{"X-Trellis-Tenant": "globex"})
    assert administered_tenant(by_header, platform, c) == "globex"  # type: ignore[arg-type]
    with pytest.raises(ValidationFailed):
        administered_tenant(_Request(platform), platform, c)  # type: ignore[arg-type]
    with pytest.raises(ValidationFailed, match="invalid tenant_id"):
        administered_tenant(
            _Request(platform, **{"X-Trellis-Tenant": "not a/valid id"}), platform, c
        )  # type: ignore[arg-type]


def test_roles_come_from_the_credential_s_claims_in_every_deployed_mode() -> None:
    from memory_service.api.deps import _has_role
    from memory_service.domain.tenancy import KeyRole

    admin = (KeyRole.ADMIN, KeyRole.PLATFORM)
    jwt_admin = ServicePrincipal(
        service_id="svc", mode="jwt", claims={"role": "admin", "tenant": "acme"}
    )
    jwt_plain = ServicePrincipal(service_id="svc", mode="jwt", claims={"tenant": "acme"})
    dev = ServicePrincipal(service_id="dev:1", mode="trusted_dev", claims={})
    assert _has_role(_key("acme", role="admin"), admin)
    assert not _has_role(jwt_admin, admin), "a jwt administers nothing, whatever it claims"
    assert not _has_role(jwt_plain, admin)
    assert not _has_role(_key("acme", role="service"), admin)
    assert not _has_role(_key("acme", role="service"), (KeyRole.PLATFORM,))
    assert not _has_role(_key("acme", role="admin"), (KeyRole.PLATFORM,)), (
        "a tenant admin is not the platform"
    )
    assert _has_role(dev, (KeyRole.PLATFORM,)), "the laptop key may do anything, by design"


def test_a_suspended_tenant_is_refused_for_every_credential_kind() -> None:
    """The verifier stops a suspended tenant's keys; jwt and development callers are stopped
    by the registry the context builder consults, so the answer does not depend on the mode."""
    from memory_service.domain.errors import AuthorizationFailed

    class _Registry:
        def is_suspended(self, tenant_id: str) -> bool:
            return tenant_id == "acme"

    c = _Container()
    c.services = {"tenant_registry": _Registry()}
    dev = ServicePrincipal(service_id="dev:1", mode="trusted_dev", claims={})
    with pytest.raises(AuthorizationFailed, match="suspended"):
        build_context(_Request(dev, **{"X-Trellis-Tenant": "acme"}), c, ScopeBody())  # type: ignore[arg-type]
    assert (
        build_context(_Request(dev, **{"X-Trellis-Tenant": "globex"}), c, ScopeBody()).tenant_id
        == "globex"
    )  # type: ignore[arg-type]


def test_a_suspended_tenant_s_administrators_are_stopped_but_the_platform_is_not() -> None:
    from memory_service.api.deps import administered_tenant
    from memory_service.domain.errors import AuthorizationFailed

    class _Registry:
        def is_suspended(self, tenant_id: str) -> bool:
            return tenant_id == "acme"

    c = _Container()
    c.services = {"tenant_registry": _Registry()}
    admin = _key("acme", role="admin")
    with pytest.raises(AuthorizationFailed, match="suspended"):
        administered_tenant(_Request(admin), admin, c)  # type: ignore[arg-type]
    platform = ServicePrincipal(service_id="platform", mode="api_key", claims={"role": "platform"})
    assert (
        administered_tenant(_Request(platform, **{"X-Trellis-Tenant": "acme"}), platform, c)
        == "acme"
    )  # type: ignore[arg-type]


def test_the_platform_role_is_the_bootstrap_secret_and_nothing_else() -> None:
    from memory_service.api.deps import _has_role
    from memory_service.domain.tenancy import KeyRole

    forged = ServicePrincipal(service_id="svc", mode="jwt", claims={"role": "platform"})
    assert not _has_role(forged, (KeyRole.PLATFORM,)), "an issuer's token cannot be the platform"
    assert not _has_role(forged, (KeyRole.ADMIN, KeyRole.PLATFORM))
    named = ServicePrincipal(service_id="platform", mode="jwt", claims={"role": "platform"})
    assert not _has_role(named, (KeyRole.PLATFORM,)), "nor one whose subject is called platform"
    listed = ServicePrincipal(
        service_id="svc", mode="jwt", claims={"role": ["admin"], "tenant": "acme"}
    )
    assert not _has_role(listed, (KeyRole.ADMIN,)), "a list is not a role, and not a crash"
    real = ServicePrincipal(service_id="platform", mode="api_key", claims={"role": "platform"})
    assert _has_role(real, (KeyRole.PLATFORM,))


def test_the_removed_spellings_bind_nothing() -> None:
    """``X-Memory-*`` was removed in 0.3.0: a key's tenant is not overridden, a workspace is
    not named, and an administrative call is not redirected by the old spelling."""
    from memory_service.api.deps import administered_tenant

    key = _key("acme")
    assert _build(key, **{"X-Memory-Tenant": "globex"}).tenant_id == "acme"
    bound = _key("acme", workspace="finance")
    assert _build(bound, **{"X-Memory-Workspace": "legal"}).workspace_id == "finance"
    admin = _key("acme", role="admin")
    named = administered_tenant(
        _Request(admin, **{"X-Memory-Tenant": "globex"}), admin, _Container()
    )  # type: ignore[arg-type]
    assert named == "acme"


def test_the_body_cannot_name_the_trace() -> None:
    """The trace id in headers, logs, rows and problems is the one the correlation middleware
    resolved; a body ``trace_id`` is refused like any other unknown field."""
    with pytest.raises(ValidationError):
        ScopeBody.model_validate({"trace_id": "opaque-from-the-body"})
    assert _build(_key("acme")).trace_id == "trace_id_test"
