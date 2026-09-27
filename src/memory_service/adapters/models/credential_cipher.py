"""AES-256-GCM with tenant/principal/key-version binding and random 96-bit nonces."""

import base64
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from memory_service.config.settings import AgentCredentialSettings
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured
from memory_service.ports.credentials import ModelIdentity, StoredCredential


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
