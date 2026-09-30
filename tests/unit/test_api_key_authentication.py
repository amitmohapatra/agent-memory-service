"""``api_key`` mode: the bootstrap secret is the platform; everything else is a stored key."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from memory_service.config.settings import AuthenticationSettings
from memory_service.domain.errors import AuthenticationFailed, DependencyUnavailable
from memory_service.domain.tenancy import KeyRole
from memory_service.modules.auth.authentication import ServiceAuthenticator

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Verified:
    key_id: str
    tenant_id: str
    role: KeyRole
    workspace_id: str | None
    may_act_as: tuple[str, ...] = ("*",)


class _Keys:
    """Stands in for ``ApiKeyVerifier``: one known token."""

    def __init__(self, token: str, verified: _Verified) -> None:
        self.token, self.verified = token, verified

    async def verify(self, token: str) -> _Verified:
        if token != self.token:
            raise AuthenticationFailed("Missing or invalid API key")
        return self.verified


def _auth(keys=None, bootstrap: str | None = "boot-secret") -> ServiceAuthenticator:
    return ServiceAuthenticator(
        AuthenticationSettings(mode="api_key", bootstrap_admin_key=bootstrap), keys=keys
    )


async def test_the_bootstrap_secret_is_the_platform_and_names_no_tenant() -> None:
    principal = await _auth().authenticate({"x-api-key": "boot-secret"})
    assert principal.mode == "api_key" and principal.service_id == "platform"
    assert principal.claims == {"role": "platform"}


async def test_a_stored_key_carries_its_tenant_role_and_workspace() -> None:
    keys = _Keys("mk_x.y", _Verified("k1", "acme", KeyRole.SERVICE, "finance"))
    principal = await _auth(keys).authenticate({"x-api-key": "mk_x.y"})
    assert principal.service_id == "key:k1"
    assert principal.claims == {
        "role": "service",
        "tenant": "acme",
        "workspace": "finance",
        "key_id": "k1",
        "may_act_as": ["*"],
    }


async def test_bearer_is_accepted_as_a_carrier_too() -> None:
    keys = _Keys("mk_x.y", _Verified("k1", "acme", KeyRole.ADMIN, None))
    principal = await _auth(keys).authenticate({"authorization": "Bearer mk_x.y"})
    assert principal.claims["role"] == "admin"


async def test_missing_or_wrong_keys_are_refused_with_one_message() -> None:
    keys = _Keys("mk_x.y", _Verified("k1", "acme", KeyRole.ADMIN, None))
    for headers in ({}, {"x-api-key": "mk_x.z"}, {"authorization": "Basic abc"}):
        with pytest.raises(AuthenticationFailed, match="Missing or invalid API key"):
            await _auth(keys).authenticate(headers)


async def test_without_a_key_store_only_the_bootstrap_secret_works() -> None:
    auth = _auth(keys=None)
    assert (await auth.authenticate({"x-api-key": "boot-secret"})).service_id == "platform"
    with pytest.raises(DependencyUnavailable):
        await auth.authenticate({"x-api-key": "mk_x.y"})


async def test_no_bootstrap_secret_means_nobody_is_the_platform() -> None:
    keys = _Keys("mk_x.y", _Verified("k1", "acme", KeyRole.ADMIN, None))
    with pytest.raises(AuthenticationFailed):
        await _auth(keys, bootstrap=None).authenticate({"x-api-key": "boot-secret"})


async def test_a_non_ascii_credential_is_refused_not_crashed() -> None:
    """A header decodes to whatever bytes arrived; ``hmac.compare_digest`` on such text raised
    a TypeError, which reached the client as a 500 with no authentication having happened."""
    keys = _Keys("mk_x.y", _Verified("k1", "acme", KeyRole.SERVICE, None))
    with pytest.raises(AuthenticationFailed):
        await _auth(keys).authenticate({"x-api-key": "boot-secr\u00e9t"})
    dev = ServiceAuthenticator(
        AuthenticationSettings(mode="trusted_dev", trusted_dev_api_keys=["dev-key"])
    )
    with pytest.raises(AuthenticationFailed):
        await dev.authenticate({"x-api-key": "dev-k\u00e9y"})
