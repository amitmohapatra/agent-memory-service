"""Learned skills against a running Bifrost gateway's skills repository (``bifrost`` marker):
the first publication creates the skill at 1.0.0, the next is 1.1.0 and served, the version
history holds both, and another tenant cannot publish over it. Needs ``BIFROST_URL`` (with
``MEMORY_TEST_PROVIDERS=env`` so the suite keeps it); ``BIFROST_ADMIN_TOKEN`` when the
gateway's management API wants one. Spends no model tokens."""

from __future__ import annotations

import uuid

import httpx
import pytest
from bifrost_sdk.admin import Admin

from memory_service.adapters.skills import BifrostSkills
from memory_service.config.settings import Settings
from memory_service.domain.errors import Conflict
from memory_service.ports.skills import SkillContent

pytestmark = pytest.mark.bifrost


async def test_a_learned_skill_is_versioned_served_and_owned_on_the_gateway() -> None:
    settings = Settings()
    if not settings.bifrost_url:
        pytest.skip("Bifrost not configured (BIFROST_URL, with MEMORY_TEST_PROVIDERS=env)")
    token = settings.bifrost_admin_token
    secret = token.get_secret_value() if token else None
    try:
        httpx.get(settings.bifrost_url.removesuffix("/v1") + "/health", timeout=3)
    except httpx.HTTPError:
        pytest.skip(f"Bifrost not reachable at {settings.bifrost_url}")
    name = f"learned-{uuid.uuid4().hex[:8]}"
    skill = SkillContent(
        name=name,
        description="Refund an order. Use for tasks like: refund order {id}.",
        body="# Refund an order\n\n## Steps\n\n1. `find_order`\n2. `refund`",
        metadata={"source": "trellis-memory", "trellis_tenant": "acme", "trellis_procedure": "p1"},
    )
    store = BifrostSkills(settings.bifrost_url, token=secret)
    async with Admin(settings.bifrost_url, token=secret) as admin:
        try:
            assert await store.publish(skill, tenant_id="acme") == "1.0.0"
            assert await store.publish(skill, tenant_id="acme") == "1.1.0"
            served = await admin.skills.find(name)
            assert served is not None and served.version == "1.1.0"
            assert served.body.startswith("# Refund an order")
            assert served.metadata["trellis_tenant"] == "acme"
            history = {v.version for v in await admin.skills.versions(served.id)}
            assert history == {"1.0.0", "1.1.0"}
            with pytest.raises(Conflict):
                await store.publish(skill, tenant_id="globex")
        finally:
            if (found := await admin.skills.find(name)) is not None:
                await admin.skills.delete(found.id)
