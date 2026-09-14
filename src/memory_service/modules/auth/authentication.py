"""Service authentication.

The upstream planner/gateway authenticates *end users*. The Memory Service authenticates the
*calling service* and only then honors the trusted context headers it sends. Modes:

- ``trusted_dev``: static API keys (development only; rejected in prod by Settings).
- ``jwt``: HS256 shared secret (dev/test) or RS256/ES256 via JWKS (prod).
- ``gcp_iam``: Google-signed ID token whose email is an allowed service account.
- ``mtls``: the ingress terminates TLS and forwards the verified client certificate subject.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from memory_service.config.settings import AuthenticationSettings
from memory_service.domain.errors import AuthenticationFailed, DependencyUnavailable


@dataclass(frozen=True)
class ServicePrincipal:
    service_id: str
    mode: str
    claims: dict[str, Any]


class ServiceAuthenticator:
    def __init__(
        self,
        settings: AuthenticationSettings,
        *,
        gcp_verifier: Callable[[str, str | None], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._gcp_verifier = gcp_verifier
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_fetched_at = 0.0

    async def authenticate(self, headers: dict[str, str]) -> ServicePrincipal:
        mode = self.settings.mode
        if mode == "trusted_dev":
            return self._trusted_dev(headers)
        if mode == "jwt":
            return await self._jwt(headers)
        if mode == "gcp_iam":
            return await self._gcp_iam(headers)
        if mode == "mtls":
            return self._mtls(headers)
        raise AuthenticationFailed(f"unsupported authentication mode {mode}")  # pragma: no cover

    # -- modes ------------------------------------------------------------------
    def _trusted_dev(self, headers: dict[str, str]) -> ServicePrincipal:
        key = headers.get(self.settings.header_api_key.lower())
        if not key or not any(
            hmac.compare_digest(key, k) for k in self.settings.trusted_dev_api_keys
        ):
            raise AuthenticationFailed("Missing or invalid API key")
        return ServicePrincipal(
            service_id=f"dev:{hashlib.sha256(key.encode()).hexdigest()[:8]}",
            mode="trusted_dev",
            claims={},
        )

    def _bearer(self, headers: dict[str, str]) -> str:
        auth = headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthenticationFailed("Missing bearer token")
        return token.strip()

    async def _jwt(self, headers: dict[str, str]) -> ServicePrincipal:
        token = self._bearer(headers)
        if self.settings.jwt_hs256_secret is not None:
            claims = _verify_hs256(token, self.settings.jwt_hs256_secret.get_secret_value())
        else:
            claims = await self._verify_with_jwks(token)
        now = time.time()
        if "exp" in claims and float(claims["exp"]) < now:
            raise AuthenticationFailed("Token expired")
        if self.settings.jwt_issuer and claims.get("iss") != self.settings.jwt_issuer:
            raise AuthenticationFailed("Unexpected issuer")
        if self.settings.jwt_audience:
            aud = claims.get("aud")
            auds = aud if isinstance(aud, list) else [aud]
            if self.settings.jwt_audience not in auds:
                raise AuthenticationFailed("Unexpected audience")
        sub = str(claims.get("sub") or claims.get("client_id") or "")
        if not sub:
            raise AuthenticationFailed("Token has no subject")
        return ServicePrincipal(service_id=sub, mode="jwt", claims=claims)

    async def _verify_with_jwks(self, token: str) -> dict[str, Any]:
        if not self.settings.jwt_jwks_url:
            raise AuthenticationFailed("jwt mode requires jwt_hs256_secret or jwt_jwks_url")
        try:
            import jwt as pyjwt
            from jwt import PyJWKClient
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailable("PyJWT[crypto] is required for JWKS verification") from exc
        try:
            client = PyJWKClient(self.settings.jwt_jwks_url, cache_keys=True)
            signing_key = client.get_signing_key_from_jwt(token)
            return pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256", "RS512", "ES512"],
                audience=self.settings.jwt_audience,
                issuer=self.settings.jwt_issuer,
                options={"verify_aud": self.settings.jwt_audience is not None},
            )
        except Exception as exc:
            raise AuthenticationFailed(f"Invalid token: {type(exc).__name__}") from exc

    async def _gcp_iam(self, headers: dict[str, str]) -> ServicePrincipal:
        token = self._bearer(headers)
        if self._gcp_verifier is None:
            raise DependencyUnavailable("gcp_iam mode requires a Google ID token verifier")
        claims = self._gcp_verifier(token, self.settings.jwt_audience)
        email = str(claims.get("email") or "")
        allowed = self.settings.gcp_allowed_service_accounts
        if allowed and email not in allowed:
            raise AuthenticationFailed("Service account not allowed")
        return ServicePrincipal(
            service_id=email or str(claims.get("sub")), mode="gcp_iam", claims=claims
        )

    def _mtls(self, headers: dict[str, str]) -> ServicePrincipal:
        verified = headers.get("ssl-client-verify") or headers.get("x-ssl-client-verify")
        subject = headers.get("ssl-client-subject-dn") or headers.get("x-ssl-client-s-dn")
        if verified != "SUCCESS" or not subject:
            raise AuthenticationFailed("Client certificate not verified by ingress")
        return ServicePrincipal(service_id=subject, mode="mtls", claims={"subject": subject})


# --------------------------------------------------------------------------
# HS256 without a third-party dependency (dev/test only)
# --------------------------------------------------------------------------


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _verify_hs256(token: str, secret: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        signature = _b64url_decode(sig_b64)
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuthenticationFailed("Malformed token") from exc
    if header.get("alg") != "HS256":
        raise AuthenticationFailed("Unsupported algorithm")
    expected = hmac.new(
        secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected, signature):
        raise AuthenticationFailed("Invalid signature")
    try:
        return json.loads(_b64url_decode(payload_b64))
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuthenticationFailed("Malformed token payload") from exc


def mint_hs256(claims: dict[str, Any], secret: str) -> str:
    """Test/dev helper to mint a token the ``jwt`` mode accepts."""
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_encode(json.dumps(claims).encode())
    sig = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(sig)}"
