"""``jwt`` mode: a bearer token signed by a key from the issuer's JWKS authenticates the
calling service as its subject; anything unsigned, mis-signed, expired, or for another issuer
or audience is refused. The JWKS document is served in-process - no network."""

from __future__ import annotations

import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from memory_service.config.settings import AuthenticationSettings
from memory_service.domain.errors import AuthenticationFailed
from memory_service.modules.auth.authentication import ServiceAuthenticator

pytestmark = pytest.mark.unit

JWKS_URL = "https://issuer.test/.well-known/jwks.json"
ISSUER = "https://issuer.test/"
AUDIENCE = "memory-service"

RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
EC_KEY = ec.generate_private_key(ec.SECP256R1())
STRANGER = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks() -> dict[str, Any]:
    rsa_jwk = RSAAlgorithm.to_jwk(RSA_KEY.public_key(), as_dict=True)
    ec_jwk = ECAlgorithm.to_jwk(EC_KEY.public_key(), as_dict=True)
    return {
        "keys": [
            {**rsa_jwk, "kid": "rsa-1", "use": "sig", "alg": "RS256"},
            {**ec_jwk, "kid": "ec-1", "use": "sig", "alg": "ES256"},
        ]
    }


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve the issuer's JWKS from memory and record which URL was asked for."""
    urls: list[str] = []

    def fetch_data(self: jwt.PyJWKClient) -> Any:
        urls.append(self.uri)
        return _jwks()

    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", fetch_data)
    return urls


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    base: dict[str, Any] = {
        "sub": "planner-service",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 300,
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def _token(claims: dict[str, Any], *, key: Any = RSA_KEY, kid: str = "rsa-1", alg: str = "RS256"):
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": kid})


def _auth(*, issuer: str | None = ISSUER, audience: str | None = AUDIENCE) -> ServiceAuthenticator:
    return ServiceAuthenticator(
        AuthenticationSettings(jwt_jwks_url=JWKS_URL, jwt_issuer=issuer, jwt_audience=audience)
    )


def _bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- verified tokens


async def test_a_token_signed_by_the_issuers_key_authenticates_its_subject(fetched) -> None:
    principal = await _auth().authenticate(_bearer(_token(_claims())))
    assert principal.mode == "jwt"
    assert principal.service_id == "planner-service"
    assert principal.claims["iss"] == ISSUER and principal.claims["aud"] == AUDIENCE
    assert fetched == [JWKS_URL]


async def test_an_ec_signed_token_is_accepted_too(fetched) -> None:
    token = _token(_claims(), key=EC_KEY, kid="ec-1", alg="ES256")
    assert (await _auth().authenticate(_bearer(token))).service_id == "planner-service"


async def test_a_client_credentials_token_is_known_by_its_client_id(fetched) -> None:
    token = _token(_claims(sub=None, client_id="etl-runner"))
    assert (await _auth().authenticate(_bearer(token))).service_id == "etl-runner"


async def test_a_token_naming_nobody_is_refused(fetched) -> None:
    with pytest.raises(AuthenticationFailed, match="no subject"):
        await _auth().authenticate(_bearer(_token(_claims(sub=None))))


async def test_an_audience_list_that_includes_ours_is_accepted(fetched) -> None:
    token = _token(_claims(aud=["billing", AUDIENCE]))
    assert (await _auth().authenticate(_bearer(token))).service_id == "planner-service"


async def test_without_a_configured_audience_or_issuer_neither_is_checked(fetched) -> None:
    token = _token(_claims(aud=None, iss="https://someone-else.test/"))
    principal = await _auth(issuer=None, audience=None).authenticate(_bearer(token))
    assert principal.service_id == "planner-service"


async def test_the_bearer_scheme_is_case_insensitive_and_padding_is_ignored(fetched) -> None:
    token = _token(_claims())
    principal = await _auth().authenticate({"authorization": f"bearer  {token} "})
    assert principal.service_id == "planner-service"


# --------------------------------------------------------------------------- refused tokens


@pytest.mark.parametrize(
    ("token_of", "error"),
    [
        (lambda: _token(_claims(), key=STRANGER), "InvalidSignatureError"),
        (lambda: _token(_claims(exp=int(time.time()) - 60)), "ExpiredSignatureError"),
        (lambda: _token(_claims(aud="another-service")), "InvalidAudienceError"),
        (lambda: _token(_claims(iss="https://evil.test/")), "InvalidIssuerError"),
        (lambda: _token(_claims(), kid="unknown-kid"), "PyJWKClientError"),
        (lambda: jwt.encode(_claims(), "shared-secret" * 4, algorithm="HS256"), "PyJWKClientError"),
    ],
)
async def test_a_token_that_does_not_verify_is_refused(fetched, token_of, error: str) -> None:
    with pytest.raises(AuthenticationFailed, match=f"Invalid token: {error}"):
        await _auth().authenticate(_bearer(token_of()))


async def test_a_token_using_the_issuers_kid_but_an_unexpected_algorithm_is_refused(
    fetched,
) -> None:
    """The kid finds the RSA key; an HMAC token "signed" with it must not verify."""
    token = jwt.encode(_claims(), "x" * 32, algorithm="HS256", headers={"kid": "rsa-1"})
    with pytest.raises(AuthenticationFailed, match="Invalid token"):
        await _auth().authenticate(_bearer(token))


@pytest.mark.parametrize(
    "headers", [{}, {"authorization": "Bearer"}, {"authorization": "Bearer "}, {"x-api-key": "k"}]
)
async def test_a_request_without_a_bearer_token_is_refused(headers: dict[str, str]) -> None:
    with pytest.raises(AuthenticationFailed, match="Missing bearer token"):
        await _auth().authenticate(headers)


# --------------------------------------------------------------------------- claims re-checked


def _with_claims(auth: ServiceAuthenticator, claims: dict[str, Any]) -> ServiceAuthenticator:
    """The authenticator with verification answering ``claims``: what is checked after the
    signature, whatever the verifying library already enforced."""

    async def verified(token: str) -> dict[str, Any]:
        return claims

    auth._verify_with_jwks = verified  # type: ignore[method-assign]
    return auth


async def test_verified_claims_that_have_expired_are_refused() -> None:
    auth = _with_claims(_auth(), _claims(exp=time.time() - 1))
    with pytest.raises(AuthenticationFailed, match="Token expired"):
        await auth.authenticate(_bearer("t"))


async def test_verified_claims_from_another_issuer_are_refused() -> None:
    auth = _with_claims(_auth(), _claims(iss="https://evil.test/"))
    with pytest.raises(AuthenticationFailed, match="Unexpected issuer"):
        await auth.authenticate(_bearer("t"))


@pytest.mark.parametrize("aud", ["another-service", ["a", "b"], None])
async def test_verified_claims_for_another_audience_are_refused(aud: Any) -> None:
    auth = _with_claims(_auth(), _claims(aud=aud))
    with pytest.raises(AuthenticationFailed, match="Unexpected audience"):
        await auth.authenticate(_bearer("t"))


async def test_verified_claims_without_an_expiry_are_accepted() -> None:
    auth = _with_claims(_auth(), _claims(exp=None))
    assert (await auth.authenticate(_bearer("t"))).service_id == "planner-service"
