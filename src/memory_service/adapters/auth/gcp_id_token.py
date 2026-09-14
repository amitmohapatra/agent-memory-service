"""Google ID token verification for the ``gcp_iam`` authentication mode (needs [gcp])."""

from __future__ import annotations

from typing import Any

from memory_service.domain.errors import AuthenticationFailed, DependencyUnavailable


def verify_google_id_token(token: str, audience: str | None) -> dict[str, Any]:
    try:
        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token
    except ImportError as exc:
        raise DependencyUnavailable(
            "google-auth is required for gcp_iam mode (install [gcp])"
        ) from exc
    try:
        return dict(id_token.verify_oauth2_token(token, g_requests.Request(), audience=audience))
    except Exception as exc:
        raise AuthenticationFailed(f"Invalid Google ID token: {type(exc).__name__}") from exc
