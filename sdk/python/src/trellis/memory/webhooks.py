"""Verify a webhook delivery (ADR 0023).

The service signs every delivery with ``X-Trellis-Signature: t=<unix seconds>,v1=<hex
hmac-sha256 of "<t>.<body>">`` using the secret shown once when the subscription was created.
A receiver checks the signature over the *raw* body bytes and rejects a stale ``t``.

    from trellis.memory.webhooks import verify_signature, parse_event

    if not verify_signature(secret, request.headers["X-Trellis-Signature"], raw_body):
        return Response(status_code=401)
    event = parse_event(raw_body)
"""

from __future__ import annotations

import hmac
import re
import time
from hashlib import sha256
from typing import Final

from trellis.memory.models import WebhookEventPayload

SIGNATURE_HEADER: Final = "X-Trellis-Signature"
EVENT_HEADER: Final = "X-Trellis-Event"
DELIVERY_HEADER: Final = "X-Trellis-Delivery"
SIGNATURE_VERSION: Final = "v1"
DEFAULT_TOLERANCE_SECONDS: Final = 300
_HEX_DIGEST: Final = re.compile(r"[0-9a-f]{64}")


def sign(secret: str, timestamp: int, body: bytes) -> str:
    digest = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, sha256).hexdigest()
    return f"t={timestamp},{SIGNATURE_VERSION}={digest}"


def parse_signature(header: str) -> tuple[int, str] | None:
    parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
    stamp, digest = parts.get("t"), parts.get(SIGNATURE_VERSION)
    if stamp is None or digest is None or not stamp.isdigit() or not _HEX_DIGEST.fullmatch(digest):
        return None
    return int(stamp), digest


def verify_signature(
    secret: str,
    header: str | None,
    body: bytes,
    *,
    now: int | None = None,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
) -> bool:
    """True when ``header`` signs ``body`` with ``secret`` and is not older than the tolerance."""
    if not header:
        return False
    parsed = parse_signature(header)
    if parsed is None:
        return False
    timestamp, digest = parsed
    current = int(time.time()) if now is None else now
    if abs(current - timestamp) > tolerance_seconds:
        return False
    expected = sign(secret, timestamp, body).partition(f",{SIGNATURE_VERSION}=")[2]
    return hmac.compare_digest(expected, digest)


def parse_event(body: bytes) -> WebhookEventPayload:
    return WebhookEventPayload.model_validate_json(body)
