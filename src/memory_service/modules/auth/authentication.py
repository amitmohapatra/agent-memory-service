"""Service authentication.

The upstream planner/gateway authenticates *end users*. The Memory Service authenticates the
*calling service* and only then honors the trusted context headers it sends. Modes:

- ``trusted_dev``: static API keys (development only; rejected in prod by Settings).
- ``jwt``: RS256/ES256 via the issuer's JWKS.
- ``api_key``: keys the service issued itself (``modules/tenancy``), verified against their
  stored hashes, each naming the tenant - and optionally the workspace - it may act for. One
  bootstrap secret is the platform operator, and it is the whole configuration a shared
  deployment needs beyond the store URLs.

``gcp_iam`` and ``mtls`` were declared and never deployed: no compose target, deploy file
or Makefile named either, and the HS256 shared-secret path that ``jwt`` also carried was a
dev-only spelling of a mode the dev stack does not use. One package ships two modes.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any, Protocol

from memory_service.config.constants import HEADERS
from memory_service.config.settings import AuthenticationSettings
from memory_service.domain.errors import AuthenticationFailed, DependencyUnavailable
from memory_service.domain.tenancy import PLATFORM_SCOPE, KeyRole, bare_credential
from memory_service.modules.auth.keys import VerifiedKey


@dataclass(frozen=True)
class ServicePrincipal:
    service_id: str
    mode: str
    claims: dict[str, Any]


class KeyVerifier(Protocol):
    """``modules/auth/keys.py``; a Protocol so the authenticator needs no store to test."""

    async def verify(self, token: str) -> VerifiedKey:
        """The verified key, or AuthenticationFailed / AuthorizationFailed (suspended)."""
        ...


def _same_secret(presented: str, expected: str) -> bool:
    """Constant-time equality for header-borne secrets.

    ``hmac.compare_digest`` refuses non-ASCII ``str`` with a TypeError, and a header decodes
    to whatever bytes a client sent, so comparing the raw text turned a garbage credential
    into a 500 instead of a 401. Bytes compare for any input.
    """
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


class ServiceAuthenticator:
    def __init__(
        self, settings: AuthenticationSettings, *, keys: KeyVerifier | None = None
    ) -> None:
        self.settings = settings
        self.keys = keys

    async def authenticate(self, headers: dict[str, str]) -> ServicePrincipal:
        mode = self.settings.mode
        if mode == "trusted_dev":
            return self._trusted_dev(headers)
        if mode == "jwt":
            return await self._jwt(headers)
        if mode == "api_key":
            return await self._api_key(headers)
        raise AuthenticationFailed(f"unsupported authentication mode {mode}")  # pragma: no cover

    # -- api_key ------------------------------------------------------------------
    async def _api_key(self, headers: dict[str, str]) -> ServicePrincipal:
        token = headers.get(HEADERS.api_key.lower())
        if not token:
            auth = headers.get("authorization", "")
            token = bare_credential(auth) if auth[:7].lower() == "bearer " else ""
        if not token:
            raise AuthenticationFailed("Missing or invalid API key")
        bootstrap = self.settings.bootstrap_admin_key
        if bootstrap is not None and _same_secret(token, bootstrap.get_secret_value()):
            return ServicePrincipal(
                service_id=PLATFORM_SCOPE, mode="api_key", claims={"role": KeyRole.PLATFORM.value}
            )
        if self.keys is None:
            raise DependencyUnavailable("api_key authentication requires the key store")
        key = await self.keys.verify(token)
        return ServicePrincipal(
            service_id=f"key:{key.key_id}",
            mode="api_key",
            claims={
                "role": key.role.value,
                "tenant": key.tenant_id,
                "workspace": key.workspace_id,
                "key_id": key.key_id,
                "may_act_as": list(key.may_act_as),
            },
        )

    # -- modes ------------------------------------------------------------------
    def _trusted_dev(self, headers: dict[str, str]) -> ServicePrincipal:
        key = headers.get(HEADERS.api_key.lower())
        if not key or not any(
            _same_secret(key, k.get_secret_value()) for k in self.settings.trusted_dev_api_keys
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
        jwks_url = str(self.settings.jwt_jwks_url)
        try:
            import jwt as pyjwt
            from jwt import PyJWKClient
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailable("PyJWT[crypto] is required for JWKS verification") from exc
        try:
            client = PyJWKClient(jwks_url, cache_keys=True)
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
