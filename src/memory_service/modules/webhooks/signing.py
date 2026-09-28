"""The delivery signature: ``X-Trellis-Signature: t=<unix seconds>,v1=<hex hmac-sha256>``.

The MAC covers ``"<t>.<body>"`` so a captured body cannot be replayed under another
timestamp; a receiver rejects a ``t`` older than its tolerance. The service only signs;
``trellis.memory.webhooks.verify_signature`` in the SDK is the receiver's side of the same
scheme, and a contract test keeps the two equal.
"""

from __future__ import annotations

import hmac
from hashlib import sha256
from typing import Final

SIGNATURE_VERSION: Final = "v1"


def sign(secret: str, timestamp: int, body: bytes) -> str:
    digest = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, sha256).hexdigest()
    return f"t={timestamp},{SIGNATURE_VERSION}={digest}"
