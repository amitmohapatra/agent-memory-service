"""AES-256-GCM with tenant/principal/key-version binding and random 96-bit nonces."""

import base64
import hashlib
import json
import os
from typing import Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from memory_service.config.settings import AgentCredentialSettings, Settings
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured
from memory_service.observability.logging import get_logger
from memory_service.ports.credentials import ModelIdentity, StoredCredential

log = get_logger(__name__)

#: Where a missing envelope key is replaced by a development one. Never a deployed
#: environment: staging and prod keep refusing registration until the operator sets one.
DEVELOPMENT_ENVIRONMENTS: Final = frozenset({"dev", "test"})
DEVELOPMENT_KEY_ID: Final = "dev-unprotected"
_DEVELOPMENT_KEY_LABEL: Final = b"trellis-memory development envelope key, not a secret\0"


def envelope_settings(settings: Settings) -> AgentCredentialSettings:
    """The operator's envelope keys, or - in ``dev``/``test`` with none configured - a
    development key, so registering a model key works on a laptop with nothing set.

    The harness registers ``BIFROST_VIRTUAL_KEY`` for each agent on its first run, and with
    the ``.env.example`` defaults that was a ``503 Agent credential encryption is not
    configured`` on every first run, which looked like an outage rather than a missing key.

    The development key is derived at startup and never stored: from a fixed label and the
    database URL, so every API worker and the background worker (which decrypts keys for
    jobs) derive the same one, and a restart can still read what was written before it.
    That makes it a key anyone with the configuration can recompute - fine for a laptop's
    throwaway virtual key, which is why it is loudly logged and why deployed environments
    never get one.
    """
    configured = settings.agent_credentials
    if (
        configured.active_key_id is not None
        or configured.encryption_keys
        or settings.service.environment not in DEVELOPMENT_ENVIRONMENTS
    ):
        return configured
    material = hashlib.sha256(
        _DEVELOPMENT_KEY_LABEL + settings.database.url.get_secret_value().encode()
    ).digest()
    log.warning(
        "agent_credentials.development_key",
        message="no envelope key is configured: registered model keys are encrypted with a "
        "development key derived from the configuration, which protects nothing. Set "
        "MEMORY__AGENT_CREDENTIALS__ACTIVE_KEY_ID and __ENCRYPTION_KEYS before storing a "
        "real key; staging and prod refuse registration without them.",
        environment=settings.service.environment,
        key_id=DEVELOPMENT_KEY_ID,
    )
    return AgentCredentialSettings(
        active_key_id=DEVELOPMENT_KEY_ID,
        encryption_keys={
            DEVELOPMENT_KEY_ID: SecretStr(base64.urlsafe_b64encode(material).decode())
        },
    )


def _aad(identity: ModelIdentity, key_id: str) -> bytes:
    return json.dumps([identity.tenant_id, identity.principal_id, key_id]).encode()


class AesCredentialCipher:
    def __init__(self, settings: AgentCredentialSettings) -> None:
        self.active_key_id = settings.active_key_id
        self.keys: dict[str, AESGCM] = {}
        for name, secret in settings.encryption_keys.items():
            try:
                key = base64.b64decode(secret.get_secret_value(), altchars=b"-_", validate=True)
                if len(key) != 32:
                    raise ValueError
            except (ValueError, TypeError) as exc:
                raise ValueError("Agent credential encryption keys must encode 32 bytes") from exc
            self.keys[name] = AESGCM(key)
        if self.active_key_id is not None and self.active_key_id not in self.keys:
            raise ValueError("Agent credential active_key_id is absent from encryption_keys")

    def encrypt(self, identity: ModelIdentity, value: SecretStr) -> tuple[str, bytes]:
        key_id = self.active_key_id
        if key_id is None:
            raise ProviderNotConfigured("Agent credential encryption is not configured")
        nonce = os.urandom(12)
        ciphertext = self.keys[key_id].encrypt(
            nonce, value.get_secret_value().encode(), _aad(identity, key_id)
        )
        return key_id, nonce + ciphertext

    def decrypt(self, record: StoredCredential) -> SecretStr:
        key = self.keys.get(record.key_id)
        if key is None:
            raise ProviderNotConfigured("Agent credential encryption key is unavailable")
        encrypted = record.ciphertext
        if encrypted is None:
            raise ProviderNotConfigured("Agent model credential is revoked")
        try:
            plain = key.decrypt(
                encrypted[:12], encrypted[12:], _aad(record.identity, record.key_id)
            )
            return SecretStr(plain.decode())
        except (InvalidTag, ValueError, UnicodeError) as exc:
            raise DependencyUnavailable("Agent credential authentication failed") from exc
