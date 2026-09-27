"""No gateway calls: encrypted credentials are bound to tenant and owner."""

import base64
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from memory_service.adapters.models.credential_cipher import AesCredentialCipher
from memory_service.config.settings import AgentCredentialSettings
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured
from memory_service.ports.credentials import ModelIdentity, StoredCredential

pytestmark = pytest.mark.unit
TEST_KEY = base64.urlsafe_b64encode(b"a" * 32).decode()


def cipher(active="a", **keys):
    return AesCredentialCipher(AgentCredentialSettings(active_key_id=active, encryption_keys=keys))


def test_credential_cipher_randomizes_and_authenticates_owner_tenant_and_key_version():
    crypt = cipher(a=TEST_KEY)
    identity = ModelIdentity("tenant", "agent:alice/research")
    key_id, encrypted = crypt.encrypt(identity, SecretStr("vk-dummy-secret"))
    assert encrypted != crypt.encrypt(identity, SecretStr("vk-dummy-secret"))[1]
    assert b"vk-dummy-secret" not in encrypted
    record = StoredCredential(identity, key_id, encrypted, 1, datetime.now(UTC))
    assert crypt.decrypt(record).get_secret_value() == "vk-dummy-secret"
    for other in (
        ModelIdentity("other", identity.principal_id),
        ModelIdentity("tenant", "agent:bob/research"),
    ):
        with pytest.raises(DependencyUnavailable):
            crypt.decrypt(replace(record, identity=other))
    damaged = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
    with pytest.raises(DependencyUnavailable):
        crypt.decrypt(replace(record, ciphertext=damaged))
    assert "vk-dummy-secret" not in repr(record)


def test_envelope_key_rotation_keeps_old_keys_readable_until_retired():
    old = cipher(a=TEST_KEY)
    identity = ModelIdentity("tenant", "agent:alice/research")
    key_id, encrypted = old.encrypt(identity, SecretStr("vk-dummy-secret"))
    record = StoredCredential(identity, key_id, encrypted, 1, datetime.now(UTC))
    next_key = base64.urlsafe_b64encode(b"b" * 32).decode()
    new = cipher(active="b", a=TEST_KEY, b=next_key)
    assert new.decrypt(record).get_secret_value() == "vk-dummy-secret"
    assert new.encrypt(identity, SecretStr("vk-new"))[0] == "b"
    with pytest.raises(ProviderNotConfigured):
        cipher(active="b", b=next_key).decrypt(record)
    with pytest.raises(ProviderNotConfigured):
        cipher(active=None).encrypt(identity, SecretStr("vk-any"))


@pytest.mark.parametrize("key", ["invalid!", base64.b64encode(b"short").decode()])
def test_invalid_envelope_configuration_fails_without_echoing_key(key):
    with pytest.raises(ValueError) as exc:
        cipher(a=key)
    assert key not in str(exc.value)


def test_settings_snapshot_redacts_the_envelope_keyring():
    from memory_service.config.settings import Settings

    settings = Settings(
        _env_file=None,
        agent_credentials={"active_key_id": "test", "encryption_keys": {"test": TEST_KEY}},
    )
    redacted = settings.redacted()["agent_credentials"]
    assert redacted["active_key_id"] == "test"
    assert redacted["encryption_keys"]["test"] == "**********"
