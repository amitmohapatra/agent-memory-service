"""A development key acts in the development tenant when a request names none.

A laptop's harness asks the key who it is (``GET /v1/keys/self``) and runs in the tenant
the answer names. A development key used to answer ``tenant_id: null`` - "names its tenant
per request" - so the harness refused to start without a ``tenant=`` the operator had to
know to pass, while every memory call without ``X-Trellis-Tenant`` was a 422. The key now
names ``authentication.trusted_dev_tenant`` and every route acts in it unless a request
names another.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from memory_service.config.settings import AuthenticationSettings

#: The unit database is shared across runs: each run has a development tenant of its own.
DEV_TENANT = f"dev-{secrets.token_hex(3)}"
KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def dev_client(make_settings, overrides) -> Iterator[TestClient]:
    settings = make_settings()
    settings.authentication.trusted_dev_tenant = DEV_TENANT
    with TestClient(create_app(settings, overrides=overrides)) as client:
        yield client


def test_the_default_development_tenant_is_default() -> None:
    assert AuthenticationSettings().trusted_dev_tenant == "default"
    with pytest.raises(ValueError, match="trusted_dev_tenant"):
        AuthenticationSettings(trusted_dev_tenant="has:colon")


def test_keys_self_names_the_development_tenant(dev_client: TestClient) -> None:
    me = dev_client.get("/v1/keys/self", headers=KEY)
    assert me.status_code == 200, me.text
    assert me.json()["tenant_id"] == DEV_TENANT and me.json()["role"] == "trusted_dev"


def test_memory_calls_without_a_tenant_act_in_the_development_tenant(
    dev_client: TestClient,
) -> None:
    user = {**KEY, "X-Trellis-User": "alice"}
    written = dev_client.post(
        "/v1/memories",
        headers=user,
        json={"scope": {}, "content": "Alice reviews budgets on Mondays.", "visibility": "USER"},
    )
    assert written.status_code in (200, 201), written.text
    memory_id = written.json()["memory_id"]
    own = dev_client.get(f"/v1/memories/{memory_id}", headers=user)
    assert own.status_code == 200 and own.json()["memory_id"] == memory_id
    named = dev_client.get(
        f"/v1/memories/{memory_id}", headers={**user, "X-Trellis-Tenant": DEV_TENANT}
    )
    assert named.status_code == 200, "naming the development tenant is the same tenant"
    elsewhere = dev_client.get(
        f"/v1/memories/{memory_id}", headers={**user, "X-Trellis-Tenant": f"{DEV_TENANT}-x"}
    )
    assert elsewhere.status_code == 404, "an explicit tenant header still picks another tenant"


def test_administration_without_a_tenant_acts_on_the_development_tenant(
    dev_client: TestClient,
) -> None:
    """The development tenant needs no onboarding row; a header naming a tenant nobody
    onboarded is still refused."""
    policy = dev_client.get("/v1/model-key/policy", headers=KEY)
    assert policy.status_code == 200, policy.text
    missing = dev_client.get(
        "/v1/model-key/policy", headers={**KEY, "X-Trellis-Tenant": f"nobody-{DEV_TENANT}"}
    )
    assert missing.status_code == 404
