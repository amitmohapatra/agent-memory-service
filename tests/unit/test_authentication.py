import pytest

from memory_service.config.settings import AuthenticationSettings
from memory_service.domain.errors import AuthenticationFailed
from memory_service.modules.auth.authentication import ServiceAuthenticator


async def test_trusted_dev_api_key() -> None:
    auth = ServiceAuthenticator(
        AuthenticationSettings(mode="trusted_dev", trusted_dev_api_keys=["k1"])
    )
    p = await auth.authenticate({"x-api-key": "k1"})
    assert p.mode == "trusted_dev"
    with pytest.raises(AuthenticationFailed):
        await auth.authenticate({"x-api-key": "wrong"})
    with pytest.raises(AuthenticationFailed):
        await auth.authenticate({})


async def test_jwt_mode_without_a_jwks_url_fails_closed() -> None:
    """The HS256 shared-secret spelling of ``jwt`` went with ``jwt_hs256_secret``; a bearer
    token with nowhere to fetch signing keys from is refused, never accepted unverified."""
    auth = ServiceAuthenticator(AuthenticationSettings(mode="jwt"))
    with pytest.raises(AuthenticationFailed, match="jwt_jwks_url"):
        await auth.authenticate({"authorization": "Bearer not.a.jwt"})
    with pytest.raises(AuthenticationFailed, match="bearer"):
        await auth.authenticate({"authorization": "Basic abc"})
