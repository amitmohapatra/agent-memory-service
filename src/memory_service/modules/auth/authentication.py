"""Service authentication.

The upstream planner/gateway authenticates *end users*. The Memory Service authenticates the
*calling service* and only then honors the trusted context headers it sends. Modes:

- ``trusted_dev``: static API keys (development only; rejected in prod by Settings).
- ``jwt``: RS256/ES256 via the issuer's JWKS.

``gcp_iam`` and ``mtls`` were declared and never deployed: no compose target, deploy file
or Makefile named either, and the HS256 shared-secret path that ``jwt`` also carried was a
dev-only spelling of a mode the dev stack does not use. One package ships two modes.
"""

from __future__ import annotations

import hashlib
import hmac
import time
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
    def __init__(self, settings: AuthenticationSettings) -> None:
        self.settings = settings

    async def authenticate(self, headers: dict[str, str]) -> ServicePrincipal:
        mode = self.settings.mode
        if mode == "trusted_dev":
            return self._trusted_dev(headers)
        if mode == "jwt":
            return await self._jwt(headers)
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
            raise AuthenticationFailed("jwt mode requires jwt_jwks_url")
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
