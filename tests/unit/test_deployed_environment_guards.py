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
        authentication={"mode": "trusted_dev", **auth},
        blob={"provider": "filesystem"},
    )


@pytest.mark.parametrize("environment", ["staging", "prod"])
def test_a_deployed_environment_refuses_the_development_defaults(environment: str) -> None:
    with pytest.raises(ValueError) as raised:
        _settings(environment)
    assert "trusted_dev" in str(raised.value)
    assert environment in str(raised.value), "the message should name where it refused"


@pytest.mark.parametrize("environment", ["dev", "test"])
def test_the_laptop_and_the_suite_keep_their_defaults(environment: str) -> None:
    settings = _settings(environment)
    assert settings.authentication.mode == "trusted_dev"
    assert settings.blob.provider == "filesystem"


def test_staging_also_refuses_the_filesystem_blob_store() -> None:
    with pytest.raises(ValueError) as raised:
        Settings(
            service={"environment": "staging"},
            authentication={"mode": "jwt", "jwt_issuer": "https://i", "jwt_audience": "a"},
            blob={"provider": "filesystem"},
        )
    assert "blob.provider" in str(raised.value)
