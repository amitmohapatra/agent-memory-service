"""API keys: one-time tokens, stored hashes, constant-time checks; member principals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.tenancy import (
    KEY_ID_LENGTH,
    ApiKey,
    KeyRole,
    hash_secret,
    mint_token,
    parse_principal,
    parse_token,
    secret_matches,
)

pytestmark = pytest.mark.unit


def test_a_minted_token_parses_back_to_its_id_and_verifies_against_its_hash() -> None:
    key_id, token, secret_hash = mint_token()
    assert token.startswith(f"mk_{key_id}.") and len(key_id) == KEY_ID_LENGTH
    parsed = parse_token(token)
    assert parsed is not None and parsed[0] == key_id
    assert secret_matches(parsed[1], secret_hash)
    assert not secret_matches(parsed[1] + "x", secret_hash)
    assert secret_hash != parsed[1], "the secret itself is never what is stored"


def test_two_mints_never_collide() -> None:
    assert len({mint_token()[0] for _ in range(64)}) == 64


@pytest.mark.parametrize(
    "token", ["", "mk_", "mk_short.secret", "sk_" + "a" * 16 + ".s", "mk_" + "a" * 16, "a" * 40]
)
def test_tokens_that_are_not_ours_do_not_parse(token: str) -> None:
    assert parse_token(token) is None


def test_hash_is_deterministic_and_hex() -> None:
    assert hash_secret("s") == hash_secret("s") and len(hash_secret("s")) == 64


def test_usable_at_respects_revocation_and_expiry() -> None:
    now = datetime.now(UTC)
    base = {
        "key_id": "k" * 16,
        "tenant_id": "acme",
        "role": KeyRole.SERVICE,
        "name": "n",
        "secret_hash": hash_secret("s"),
        "created_by": "platform",
    }
    assert ApiKey(**base).usable_at(now)
    assert not ApiKey(**base, revoked_at=now).usable_at(now)
    assert not ApiKey(**base, expires_at=now - timedelta(seconds=1)).usable_at(now)
    assert ApiKey(**base, expires_at=now + timedelta(days=1)).usable_at(now)


@pytest.mark.parametrize("principal", ["user:u1", "agent:research", "group:analysts"])
def test_member_principals_parse(principal: str) -> None:
    kind, ident = parse_principal(principal)
    assert principal == f"{kind}:{ident}"


@pytest.mark.parametrize("principal", ["u1", "service:x", "user:", "tenant:acme", "user:bad id"])
def test_anything_else_is_not_a_member_principal(principal: str) -> None:
    with pytest.raises(ValueError, match="invalid principal"):
        parse_principal(principal)


@pytest.mark.parametrize("tenant_id", ["platform", "acme:prod", "a/b", ""])
def test_tenant_ids_the_platform_cannot_serve_are_refused(tenant_id: str) -> None:
    """``platform`` is the onboarding idempotency scope; ``:`` joins tenant and key in cache
    keys, so a tenant carrying one would alias another tenant's entries."""
    from pydantic import ValidationError

    from memory_service.domain.tenancy import Tenant

    with pytest.raises(ValidationError):
        Tenant(tenant_id=tenant_id, name="x")


def test_a_credential_is_the_same_whichever_carrier_brought_it() -> None:
    from memory_service.domain.tenancy import bare_credential

    assert bare_credential("mk_a.b") == "mk_a.b"
    assert bare_credential("Bearer mk_a.b") == "mk_a.b"
    assert bare_credential("bearer  mk_a.b ") == "mk_a.b"
    assert bare_credential("Basic xyz") == "Basic xyz", "only the bearer scheme is a carrier"


@pytest.mark.parametrize(
    "token", ["mk_ABCDEFGHIJKLMNOP.s", "mk_0123456789abcde-.s", "mk_" + "\x00" * 16 + ".s"]
)
def test_a_key_id_is_only_what_minting_makes(token: str) -> None:
    """Any other sixteen characters are not a key id: they must reach neither the cache,
    nor the store, nor the unknown-id budget."""
    from memory_service.domain.tenancy import parse_token

    assert parse_token(token) is None
