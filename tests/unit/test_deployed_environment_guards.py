"""A deployed environment may not run the laptop defaults.

The startup check fired on ``prod`` alone, while ``.env.example`` told the operator that
anything other than ``dev`` refused the development authentication mode. So a deployment set
to ``staging`` started with header-trust authentication and the single API key whose value is
the default in settings.py and is printed in .env.example: anyone who could reach the port and
had read the repository could authenticate as any tenant and user by setting three headers.
Nothing said so - the service started clean and readiness went green.

``test`` stays exempt deliberately: the suite and the benchmarks run under it with exactly
those defaults, which is what they are for.
"""

from __future__ import annotations

import pytest

from memory_service.config.settings import Settings

pytestmark = pytest.mark.unit


def _settings(environment: str, **auth):
    return Settings(
        service={"environment": environment},
        # the mode follows from what is configured: development keys are trusted_dev
        authentication={"trusted_dev_api_keys": ["dev-key"], **auth},
    )


@pytest.mark.parametrize("environment", ["staging", "prod"])
def test_a_deployed_environment_refuses_the_development_defaults(environment: str) -> None:
    with pytest.raises(ValueError) as raised:
        _settings(environment)
    assert "trusted_dev" in str(raised.value)
    assert environment in str(raised.value), "the message should name where it refused"


@pytest.mark.parametrize("environment", ["dev", "test"])
def test_the_laptop_and_the_suite_keep_their_defaults(
    environment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    settings = _settings(environment)
    assert settings.authentication.mode == "trusted_dev"
    assert settings.blob_provider == "filesystem"


@pytest.mark.parametrize("environment", ["staging", "prod"])
def test_a_deployed_environment_stores_blobs_in_gcs(environment: str) -> None:
    settings = Settings(
        service={"environment": environment},
        authentication={
            "jwt_jwks_url": "https://i/jwks",
            "jwt_issuer": "https://i",
            "jwt_audience": "a",
            "tenant_claim": "tenant",
        },
    )
    assert settings.blob_provider == "gcs"


_JWT = {"jwt_jwks_url": "https://i/jwks", "jwt_issuer": "https://i", "jwt_audience": "a"}


@pytest.mark.parametrize("environment", ["staging", "prod"])
def test_a_deployed_jwt_environment_must_bind_the_tenant(environment: str) -> None:
    """In ``jwt`` mode the token names the calling service and the tenant arrives in a
    header. With neither ``tenant_claim`` nor ``single_tenant`` nothing compared the two, and
    the service started clean: one token reached every tenant by changing the header."""
    with pytest.raises(ValueError) as raised:
        Settings(service={"environment": environment}, authentication=_JWT)
    assert "tenant_claim" in str(raised.value) and "single_tenant" in str(raised.value)
    assert environment in str(raised.value)


@pytest.mark.parametrize("environment", ["staging", "prod"])
@pytest.mark.parametrize("binding", [{"tenant_claim": "tenant"}, {"single_tenant": "arhaus"}])
def test_a_deployed_jwt_environment_starts_with_either_binding(
    environment: str, binding: dict[str, str]
) -> None:
    settings = Settings(service={"environment": environment}, authentication={**_JWT, **binding})
    assert settings.authentication.mode == "jwt"


@pytest.mark.parametrize("environment", ["dev", "test"])
def test_an_undeployed_jwt_environment_needs_no_binding(environment: str) -> None:
    assert Settings(service={"environment": environment}, authentication=_JWT)


def test_the_single_tenant_is_a_tenant_id() -> None:
    with pytest.raises(ValueError, match="single_tenant"):
        Settings(authentication={**_JWT, "single_tenant": "not a tenant id"})
