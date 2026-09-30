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


async def test_jwt_mode_refuses_what_is_not_a_verifiable_bearer_token() -> None:
    """``jwt`` is the mode exactly when the issuer's JWKS is configured, so a token always has
    somewhere its signing key comes from; anything that is not a bearer token is refused."""
    auth = ServiceAuthenticator(AuthenticationSettings(jwt_jwks_url="http://127.0.0.1:9/jwks"))
    with pytest.raises(AuthenticationFailed, match="Invalid token"):
        await auth.authenticate({"authorization": "Bearer not.a.jwt"})
    with pytest.raises(AuthenticationFailed, match="bearer"):
        await auth.authenticate({"authorization": "Basic abc"})
