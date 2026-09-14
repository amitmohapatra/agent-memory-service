import time

import pytest

from memory_service.config.settings import AuthenticationSettings
from memory_service.domain.errors import AuthenticationFailed
from memory_service.modules.auth.authentication import ServiceAuthenticator, mint_hs256


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


async def test_jwt_hs256_claims_validation() -> None:
    settings = AuthenticationSettings(
        mode="jwt", jwt_hs256_secret="s3cret", jwt_issuer="iss", jwt_audience="memory"
    )
    auth = ServiceAuthenticator(settings)
    good = mint_hs256(
        {"sub": "planner", "iss": "iss", "aud": ["memory"], "exp": time.time() + 60}, "s3cret"
    )
    p = await auth.authenticate({"authorization": f"Bearer {good}"})
    assert p.service_id == "planner" and p.mode == "jwt"
    for bad in (
        mint_hs256(
            {"sub": "planner", "iss": "iss", "aud": "memory", "exp": time.time() - 1}, "s3cret"
        ),
        mint_hs256({"sub": "planner", "iss": "other", "aud": "memory"}, "s3cret"),
        mint_hs256({"sub": "planner", "iss": "iss", "aud": "other"}, "s3cret"),
        mint_hs256({"sub": "planner", "iss": "iss", "aud": "memory"}, "wrong-secret"),
        "not.a.jwt",
    ):
        with pytest.raises(AuthenticationFailed):
            await auth.authenticate({"authorization": f"Bearer {bad}"})
    with pytest.raises(AuthenticationFailed):
        await auth.authenticate({"authorization": "Basic abc"})


async def test_mtls_requires_verified_header() -> None:
    auth = ServiceAuthenticator(AuthenticationSettings(mode="mtls"))
    p = await auth.authenticate(
        {"ssl-client-verify": "SUCCESS", "ssl-client-subject-dn": "CN=planner"}
    )
    assert p.service_id == "CN=planner"
    with pytest.raises(AuthenticationFailed):
        await auth.authenticate(
            {"ssl-client-verify": "FAILED", "ssl-client-subject-dn": "CN=planner"}
        )
