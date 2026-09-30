"""Per-tenant model policies, the usage ledger and key-gated background work on PostgreSQL,
with a mocked gateway: no provider credit is consumed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.policies import LLMUsage, ModelPolicies
from memory_service.modules.memory.reflection import ReflectionService
from memory_service.ports.credentials import ModelIdentity, tenant_identity, workspace_identity
from tests.integration.test_agent_credentials import service as credentials_service
from tests.integration.test_reflection import _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

AGENT = ModelIdentity("acme", "agent:u1/research", "finance")
U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
G1 = MemoryExecutionContext(tenant_id="globex", user_id="u1", workspace_id="ws1")


async def _set_key(credentials, uow_factory, identity: ModelIdentity, key: str | None) -> None:
    async with uow_factory() as uow:
        await credentials.set_for(uow, identity, SecretStr(key) if key else None)
        await uow.commit()


async def _set_policy(policies, uow_factory, level, uses, read_assist) -> None:
    async with uow_factory() as uow:
        await policies.set(uow, level, uses=uses, read_assist=read_assist)
        await uow.commit()


async def test_policies_and_keys_resolve_most_specific_first(container, uow_factory) -> None:
    policies = ModelPolicies(uow_factory)
    credentials = credentials_service(uow_factory)
    default = await policies.access(AGENT)
    assert default.read_assist and not default.has_key and "reflection" in default.uses

    await _set_key(credentials, uow_factory, tenant_identity("acme"), "vk-tenant-test")
    assert (await policies.access(AGENT)).has_key, "the tenant's key pays for its agents"

    await _set_policy(policies, uow_factory, tenant_identity("acme"), ["summaries"], False)
    tenant_wide = await policies.access(AGENT)
    assert tenant_wide.uses == {"summaries"} and not tenant_wide.read_assist

    finance = workspace_identity("acme", "finance")
    await _set_policy(policies, uow_factory, finance, ["reflection", "summaries"], True)
    team = await policies.access(AGENT)
    assert team.uses == {"reflection", "summaries"} and team.read_assist
    elsewhere = await policies.access(ModelIdentity("acme", "agent:u1/research", "legal"))
    assert elsewhere.uses == {"summaries"}, "another team falls back to the tenant's policy"

    # a revocation at the agent's own level is final: it never borrows the tenant's key
    await _set_key(credentials, uow_factory, ModelIdentity("acme", AGENT.principal_id), None)
    assert not (await policies.access(AGENT)).has_key

    stored = await policies.get(finance)
    assert stored is not None and stored.revision == 1
    await _set_policy(policies, uow_factory, finance, ["summaries"], True)
    assert (await policies.get(finance)).revision == 2  # type: ignore[union-attr]
    assert await policies.get(workspace_identity("acme", "legal")) is None


async def test_a_policy_change_invalidates_cached_bundles(container, uow_factory) -> None:
    async with uow_factory() as uow:
        before = await uow.revisions.get_many("acme", [(RevisionKind.TENANT, "")])
    await _set_policy(
        ModelPolicies(uow_factory), uow_factory, tenant_identity("acme"), ["summaries"], True
    )
    async with uow_factory() as uow:
        after = await uow.revisions.get_many("acme", [(RevisionKind.TENANT, "")])
    assert after != before


async def test_the_usage_ledger_counts_tokens_and_calls_per_day_and_use(
    container, uow_factory
) -> None:
    usage = LLMUsage(uow_factory)
    await usage.record("acme", "summaries", 30)
    await usage.record("acme", "summaries", 12)
    await usage.record("acme", "reflection", 100)
    await usage.record("globex", "summaries", 5)
    today = datetime.now(UTC).date()
    days = await usage.between("acme", today - timedelta(days=1), today)
    assert [(d.use, d.tokens, d.calls) for d in days] == [
        ("reflection", 100, 1),
        ("summaries", 42, 2),
    ]
    assert await usage.between("acme", today - timedelta(days=3), today - timedelta(days=2)) == []


async def test_reflection_runs_only_where_a_key_pays_and_the_policy_allows(
    container, uow_factory
) -> None:
    """Automatic mode, no operator key: the acme tenant registers a key, globex does not.
    The job reflects for acme only; once acme's policy drops reflection, its pending sources
    are acknowledged without a model call."""
    for ctx in (U1, G1):
        await _observe(container, uow_factory, ctx, "I prefer concise answers with code samples.")
        await _observe(container, uow_factory, ctx, "I prefer bullet points over long paragraphs.")
    sources = await _memories(container, uow_factory, U1)
    credentials = credentials_service(uow_factory)
    policies = ModelPolicies(uow_factory)
    await _set_key(credentials, uow_factory, tenant_identity("acme"), "vk-acme-test")
    reply = {
        "insights": [
            {
                "content": "User prefers terse, skimmable answers",
                "memory_type": "PREFERENCE",
                "source_memory_ids": sorted(m.memory_id for m in sources),
            }
        ]
    }
    with mocked_gateway([reply]) as gw:
        auto = gw.assist(uses=["reflection"], enabled="auto", api_key=None)
        auto.provider.credentials = credentials
        auto.provider.usage = LLMUsage(uow_factory)  # as production wires it
        assist = LLMAssist(auto.provider, auto.settings, policies)
        service = ReflectionService(uow_factory, assist=assist)
        created = await service.reflect_all()
        assert len(created) == 1 and gw.route.call_count == 1, "acme only"
        assert gw.route.calls.last.request.headers["authorization"] == "Bearer vk-acme-test"
        assert await service.reflect_all() == [], "globex has no key: never scanned"
        assert gw.route.call_count == 1

        await _observe(container, uow_factory, U1, "I prefer dark mode in every editor.")
        await _set_policy(policies, uow_factory, tenant_identity("acme"), ["summaries"], True)
        assert await service.reflect_all() == []
        assert gw.route.call_count == 1, "the policy no longer allows reflection"
        async with uow_factory() as uow:
            pending = await uow.memories.reflection_pending(tenant_id="acme")
        assert pending == [], "acknowledged rather than rescanned forever"
    async with uow_factory() as uow:
        ledger = await uow.llm_usage.between(
            "acme", datetime.now(UTC).date(), datetime.now(UTC).date()
        )
    assert [(d.use, d.calls) for d in ledger] == [("reflection", 1)]
