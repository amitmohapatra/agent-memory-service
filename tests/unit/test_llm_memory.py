"""LLM-assisted memory paths against the mocked gateway: ambiguous extraction, ambiguous
worthiness, conflict adjudication and reflection. Every use is checked three ways: the
model's answer is applied, a failing gateway leaves the native result, and a disabled flag
never reaches the gateway."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.application.container import Container
from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import DedupDecision, MemoryType, ObservationKind, Visibility
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.memory.reflection import ReflectionService
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.unit

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1")
SCOUT = CTX.model_copy(update={"agent_id": "scout"})

FACT = "The Atlas project uses Postgres 16."
UNMATCHED = "Our standup moved to the afternoon this week."
EXISTING = "The build server runs on Ubuntu 22.04 in Berlin."
NEAR = "The build server runs on Ubuntu 22.04 in the Berlin office."


def _obs(text: str, ctx: MemoryExecutionContext = CTX) -> Observation:
    return Observation(
        tenant_id=ctx.tenant_id,
        kind=ObservationKind.MESSAGE,
        content=text,
        content_hash=content_hash(text),
        user_id=ctx.user_id,
        thread_id=ctx.thread_id,
        workspace_id=ctx.workspace_id,
        agent_id=ctx.agent_id,
        principal_id=ctx.principal_id,
        message_id="msg_1",
    )


FACT_OBS = _obs(FACT)  # one observation, so candidates from different runs compare equal


def _native(assist=None) -> NativeMemoryIntelligence:
    return NativeMemoryIntelligence(MemoryIntelligenceSettings(), assist=assist)


async def _first(provider: NativeMemoryIntelligence, text: str, ctx=CTX):
    cands = await provider.extract(_obs(text, ctx), ctx)
    return await provider.classify(cands[0], ctx)


# --- ambiguous_extraction ------------------------------------------------------------------


async def test_extraction_refinement_is_applied_and_keeps_native_evidence() -> None:
    reply = {
        "memory_type": "SEMANTIC",
        "content": "The Atlas project runs on Postgres 16",
        "subject": "Atlas project",
        "predicate": "database",
        "object": "Postgres 16",
    }
    with mocked_gateway([reply]) as gw:
        provider = _native(gw.assist(uses=["ambiguous_extraction"]))
        native = (await _native().extract(FACT_OBS, CTX))[0]
        assert native.confidence <= 0.7
        cand = (await provider.extract(FACT_OBS, CTX))[0]
    assert gw.route.call_count == 1
    assert gw.prompts()[0]["model"] == "test/strong"
    assert FACT in gw.prompts()[0]["messages"][1]["content"]
    assert cand.content == "The Atlas project runs on Postgres 16"
    assert (cand.subject, cand.predicate, cand.object) == (
        "atlas project",
        "database",
        "postgres 16",
    )
    assert cand.memory_type is MemoryType.SEMANTIC
    assert cand.evidence == native.evidence and cand.confidence == native.confidence


async def test_extraction_refinement_changes_type_and_lifetime() -> None:
    reply = {"memory_type": "TASK", "content": "Upgrade the Atlas project to Postgres 16"}
    with mocked_gateway([reply]) as gw:
        cand = await _first(_native(gw.assist(uses=["ambiguous_extraction"])), FACT)
    assert cand.memory_type is MemoryType.TASK and cand.category == "task"
    assert cand.lifetime.value == "SHORT_TERM"


@pytest.mark.parametrize(
    "reply",
    [
        {"memory_type": "BOGUS", "content": "x"},
        {"memory_type": "SEMANTIC", "content": "   "},
        {"memory_type": "CUSTOM", "content": "custom types need a plugin"},
    ],
)
async def test_extraction_refinement_keeps_native_on_invalid_output(reply) -> None:
    with mocked_gateway([reply]) as gw:
        cand = (await _native(gw.assist(uses=["ambiguous_extraction"])).extract(FACT_OBS, CTX))[0]
    native = (await _native().extract(FACT_OBS, CTX))[0]
    assert cand == native


async def test_extraction_refinement_falls_back_when_gateway_fails() -> None:
    with mocked_gateway(failing=True) as gw:
        cand = (await _native(gw.assist(uses=["ambiguous_extraction"])).extract(FACT_OBS, CTX))[0]
    assert gw.route.call_count >= 1
    assert cand == (await _native().extract(FACT_OBS, CTX))[0]


async def test_extraction_refinement_not_consulted_when_flag_off_or_confident() -> None:
    with mocked_gateway(['{"memory_type": "SEMANTIC", "content": "x"}']) as gw:
        off = _native(gw.assist(uses=["ambiguous_worthiness"]))
        assert (await off.extract(FACT_OBS, CTX)) == (await _native().extract(FACT_OBS, CTX))
        on = _native(gw.assist(uses=["ambiguous_extraction"]))
        confident = await on.extract(_obs("My timezone is CET."), CTX)
        assert confident[0].confidence > 0.7 and confident[0].predicate == "timezone"
    assert gw.route.call_count == 0


def _extracted(cands):
    """What extraction understood, without the verbatim copy of the turn.

    `keep_verbatim_turns` makes every substantive MESSAGE also produce an OBSERVATION of
    itself, so a turn no rule could parse stays retrievable. These tests are about whether
    the *assist* produced a candidate, which is a separate question.
    """
    return [c for c in cands if c.category != "verbatim_turn"]


# --- ambiguous_worthiness ------------------------------------------------------------------


async def test_worthiness_stores_model_candidate_with_native_evidence() -> None:
    # No rule matches this sentence, and CTX is inside a thread - where the thread, not
    # a verbatim copy, keeps the turn - so the native path produces nothing at all.
    # That is the gap ambiguous_worthiness exists to close.
    assert await _native().extract(_obs(UNMATCHED), CTX) == []
    reply = {"worthy": True, "memory_type": "SEMANTIC", "content": "Standup is in the afternoon"}
    with mocked_gateway([reply]) as gw:
        cands = await _native(gw.assist(uses=["ambiguous_worthiness"])).extract(
            _obs(UNMATCHED), CTX
        )
    assert gw.route.call_count == 1
    assert gw.prompts()[0]["model"] == "test/fast"
    assert [c.category for c in cands] == ["assisted"]
    cand = cands[0]
    assert cand.content == "Standup is in the afternoon"
    assert cand.memory_type is MemoryType.SEMANTIC and cand.confidence == 0.6
    assert cand.category == "assisted" and cand.subject == "thread:thr_1"
    assert cand.evidence[0].message_id == "msg_1"


async def test_worthiness_user_type_is_about_the_user() -> None:
    reply = {"worthy": True, "memory_type": "PREFERENCE", "content": "Prefers afternoon standups"}
    with mocked_gateway([reply]) as gw:
        cand = await _first(_native(gw.assist(uses=["ambiguous_worthiness"])), UNMATCHED)
    assert cand.subject == "user:u1" and cand.visibility is Visibility.USER


@pytest.mark.parametrize(
    "reply",
    [
        {"worthy": False},
        {"worthy": True},
        {"worthy": True, "memory_type": "SEMANTIC", "content": ""},
        {"worthy": "yes", "memory_type": "SEMANTIC", "content": "x"},
    ],
)
async def test_worthiness_drops_unless_answer_validates(reply) -> None:
    with mocked_gateway([reply]) as gw:
        cands = await _native(gw.assist(uses=["ambiguous_worthiness"])).extract(
            _obs(UNMATCHED), CTX
        )
    assert _extracted(cands) == []


async def test_worthiness_skips_questions_and_chitchat_and_is_bounded() -> None:
    reply = {"worthy": True, "memory_type": "SEMANTIC", "content": "Something durable"}
    with mocked_gateway([reply]) as gw:
        provider = _native(gw.assist(uses=["ambiguous_worthiness"]))
        # Questions are skipped by the assist AND kept out of the verbatim store, so a
        # question produces nothing at all — the one input where both rules agree.
        assert await provider.extract(_obs("What time is the standup?"), CTX) == []
        assert await provider.extract(_obs("Can you move the standup?"), CTX) == []
        assert gw.route.call_count == 0
        many = " ".join(f"Item {i} sits quietly on the shelf." for i in range(12))
        cands = await provider.extract(_obs(many), CTX)
    assert gw.route.call_count == 8
    assert len(_extracted(cands)) == 1  # identical normalized content is collapsed


async def test_worthiness_falls_back_to_drop_when_gateway_fails_or_flag_off() -> None:
    with mocked_gateway(failing=True) as gw:
        assert (
            _extracted(
                await _native(gw.assist(uses=["ambiguous_worthiness"])).extract(
                    _obs(UNMATCHED), CTX
                )
            )
            == []
        )
    assert gw.route.call_count >= 1
    with mocked_gateway(['{"worthy": true, "memory_type": "SEMANTIC", "content": "x"}']) as gw:
        assert (
            _extracted(
                await _native(gw.assist(uses=["ambiguous_extraction"])).extract(
                    _obs(UNMATCHED), CTX
                )
            )
            == []
        )
    assert gw.route.call_count == 0


# --- conflict_adjudication -----------------------------------------------------------------


async def _pair(existing_text: str, incoming_text: str, *, existing_ctx=CTX):
    plain = _native()
    now = datetime.now(UTC)
    existing = [
        build_memory(await _first(plain, existing_text, existing_ctx), existing_ctx, now=now)
    ]
    return await _first(plain, incoming_text), existing


@pytest.mark.parametrize(
    ("verdict", "decision"),
    [
        ("same", DedupDecision.MERGE),
        ("update", DedupDecision.SUPERSEDE),
        ("contradict", DedupDecision.CONTRADICT),
        ("different", DedupDecision.CREATE),
    ],
)
async def test_grey_band_is_adjudicated_by_the_model(verdict, decision) -> None:
    cand, existing = await _pair(EXISTING, NEAR)
    assert (await _native().consolidate(cand, existing, CTX)).decision is DedupDecision.CREATE
    with mocked_gateway([{"verdict": verdict}]) as gw:
        out = await _native(gw.assist(uses=["conflict_adjudication"])).consolidate(
            cand, existing, CTX
        )
    assert gw.route.call_count == 1
    prompt = gw.prompts()[0]["messages"][1]["content"]
    assert EXISTING in prompt and NEAR in prompt and "user:u1" in prompt
    assert out.decision is decision, out.reason
    if decision is not DedupDecision.CREATE:
        assert out.target_memory_id == existing[0].memory_id
        assert out.reason.startswith("model:")


async def test_grey_band_hard_blocks_are_never_sent_to_the_model() -> None:
    with mocked_gateway([{"verdict": "same"}]) as gw:
        provider = _native(gw.assist(uses=["conflict_adjudication"]))
        for incoming in (
            "The build server runs on Ubuntu 24.04 in the Berlin office.",
            "The build server does not run on Ubuntu 22.04 in the Berlin office.",
        ):
            cand, existing = await _pair(EXISTING, incoming)
            out = await provider.consolidate(cand, existing, CTX)
            assert out.decision is DedupDecision.CREATE, incoming
    assert gw.route.call_count == 0


async def test_grey_band_falls_back_natively_when_gateway_fails_or_flag_off() -> None:
    cand, existing = await _pair(EXISTING, NEAR)
    with mocked_gateway(failing=True) as gw:
        out = await _native(gw.assist(uses=["conflict_adjudication"])).consolidate(
            cand, existing, CTX
        )
    assert gw.route.call_count >= 1 and out.decision is DedupDecision.CREATE
    with mocked_gateway([{"verdict": "same"}]) as gw:
        out = await _native(gw.assist(uses=["reflection"])).consolidate(cand, existing, CTX)
    assert gw.route.call_count == 0 and out.decision is DedupDecision.CREATE
    # the exact-duplicate and slot rules still decide without the model
    with mocked_gateway([{"verdict": "different"}]) as gw:
        dup, existing = await _pair("My timezone is CET.", "my timezone is CET")
        out = await _native(gw.assist(uses=["conflict_adjudication"])).consolidate(
            dup, existing, CTX
        )
    assert gw.route.call_count == 0 and out.decision is DedupDecision.REINFORCE


async def _shared_conflict():
    """Another agent wrote 'timezone = PST' into the thread; the user now says CET."""
    plain = _native()
    other = await _first(plain, "My timezone is PST.", SCOUT)
    other = other.model_copy(
        update={"visibility": Visibility.THREAD, "memory_type": MemoryType.SEMANTIC}
    )
    existing = [build_memory(other, SCOUT, now=datetime.now(UTC))]
    assert existing[0].scope.level.value == "THREAD"
    assert existing[0].owner_principal == "agent:u1/scout"
    cand = await _first(plain, "My timezone is CET.")
    assert (cand.subject, cand.predicate) == (existing[0].subject, existing[0].predicate)
    return cand, existing


@pytest.mark.parametrize(
    ("verdict", "decision"),
    [
        ("update", DedupDecision.SUPERSEDE),
        ("same", DedupDecision.REINFORCE),
        ("contradict", DedupDecision.CONTRADICT),
        ("different", DedupDecision.CONTRADICT),
    ],
)
async def test_shared_scope_conflict_is_adjudicated(verdict, decision) -> None:
    cand, existing = await _shared_conflict()
    native = await _native().consolidate(cand, existing, CTX)
    assert native.decision is DedupDecision.CONTRADICT
    with mocked_gateway([{"verdict": verdict}]) as gw:
        out = await _native(gw.assist(uses=["conflict_adjudication"])).consolidate(
            cand, existing, CTX
        )
    assert gw.route.call_count == 1
    assert "agent:u1/scout" in gw.prompts()[0]["messages"][1]["content"]
    assert out.decision is decision and out.target_memory_id == existing[0].memory_id


async def test_shared_scope_conflict_keeps_native_on_failure_or_flag_off() -> None:
    cand, existing = await _shared_conflict()
    with mocked_gateway(failing=True) as gw:
        out = await _native(gw.assist(uses=["conflict_adjudication"])).consolidate(
            cand, existing, CTX
        )
    assert gw.route.call_count >= 1 and out.decision is DedupDecision.CONTRADICT
    with mocked_gateway([{"verdict": "update"}]) as gw:
        out = await _native(gw.assist(uses=[])).consolidate(cand, existing, CTX)
    assert gw.route.call_count == 0 and out.decision is DedupDecision.CONTRADICT


# --- reflection ----------------------------------------------------------------------------


class _Memories:
    def __init__(self, recent) -> None:
        self.recent = list(recent)
        self.added = []

    async def list_recent(self, *, since, limit=1000):
        return [m for m in self.recent if m.created_at >= since][:limit]

    async def candidates(
        self, tenant_id, *, scope_key, normalized_hash=None, subject=None, limit=20
    ):
        return [m for m in self.added if m.scope.key() == scope_key][:limit]

    async def add(self, memory, *, visibility_keys):
        memory.system_metadata["visibility_keys"] = list(visibility_keys)
        self.added.append(memory)
        self.recent.insert(0, memory)


class _Revisions:
    def __init__(self) -> None:
        self.bumped = []

    async def bump(self, tenant_id, kind, object_id=""):
        self.bumped.append((kind, object_id))
        return 1


class _UoW:
    def __init__(self, memories: _Memories) -> None:
        self.memories = memories
        self.revisions = _Revisions()
        self.enqueued = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def enqueue(self, spec):
        self.enqueued.append(spec)
        return 1

    async def commit(self):
        self.commits += 1


async def _sources(*texts: str, ctx=CTX, age=timedelta(hours=1)):
    plain = _native()
    now = datetime.now(UTC) - age
    return [build_memory(await _first(plain, t, ctx), ctx, now=now) for t in texts]


async def test_reflection_stores_validated_insights_with_provenance() -> None:
    sources = await _sources("I prefer concise answers.", "I prefer bullet points.")
    ids = [m.memory_id for m in sources]
    uow = _UoW(_Memories(sources))
    reply = {
        "insights": [
            {
                "content": "User prefers terse, structured answers",
                "memory_type": "PREFERENCE",
                "source_memory_ids": ids,
            },
            {"content": "Unfounded", "memory_type": "SEMANTIC", "source_memory_ids": ["nope"]},
            {"content": "", "memory_type": "SEMANTIC", "source_memory_ids": ids[:1]},
        ]
    }
    with mocked_gateway([reply]) as gw:
        service = ReflectionService(lambda: uow, assist=gw.assist(uses=["reflection"]))
        created = await service.reflect_all()
        assert gw.route.call_count == 1
        prompt = gw.prompts()[0]["messages"][1]["content"]
        assert all(i in prompt for i in ids) and "user:u1" in prompt
        assert len(created) == 1 and uow.memories.added[0].memory_id == created[0]
        insight = uow.memories.added[0]
        assert insight.content == "User prefers terse, structured answers"
        assert insight.memory_type is MemoryType.PREFERENCE and insight.confidence == 0.6
        assert insight.owner_principal == "user:u1" and insight.visibility is Visibility.PRIVATE
        assert insight.scope.level.value == "USER" and insight.scope.user_id == "u1"
        assert sorted(e.source_id for e in insight.evidence) == sorted(ids)
        assert all(e.source_type == "memory" for e in insight.evidence)
        assert insight.system_metadata["category"] == "reflection"
        assert insight.system_metadata["source_memory_ids"] == sorted(ids)
        assert insight.system_metadata["contributors"] == ["user:u1"]
        assert insight.system_metadata["visibility_keys"] == ["principal:acme/user:u1"]
        assert [j.task_name for j in uow.enqueued] == ["memory.index"]
        assert uow.enqueued[0].payload == {"tenant_id": "acme", "memory_ids": created}
        assert uow.revisions.bumped and uow.commits == 1
        # the scope has nothing newer than its last insight: no second consultation
        assert await service.reflect_all() == []
        assert gw.route.call_count == 1
        # an identical insight for the same scope is not stored twice
        assert await service.reflect("acme", "user:u1", sources) == []
        assert gw.route.call_count == 2 and len(uow.memories.added) == 1


async def test_reflection_bounds_sources_and_skips_old_memories() -> None:
    fresh = await _sources(*[f"The widget {i} costs {i} USD." for i in range(45)])
    old = await _sources("I prefer tabs.", age=timedelta(days=3))
    uow = _UoW(_Memories(fresh + old))
    with mocked_gateway(['{"insights": []}']) as gw:
        service = ReflectionService(lambda: uow, assist=gw.assist(uses=["reflection"]))
        assert await service.reflect_all() == []
    prompt = gw.prompts()[0]["messages"][1]["content"]
    assert prompt.count("\n- ") == 40 and old[0].memory_id not in prompt
    assert uow.commits == 0 and uow.enqueued == []


async def test_reflection_is_a_no_op_when_gateway_fails_or_flag_off() -> None:
    sources = await _sources("I prefer concise answers.")
    uow = _UoW(_Memories(sources))
    with mocked_gateway(failing=True) as gw:
        service = ReflectionService(lambda: uow, assist=gw.assist(uses=["reflection"]))
        assert await service.reflect_all() == []
    assert gw.route.call_count >= 1 and uow.memories.added == [] and uow.commits == 0
    with mocked_gateway(['{"insights": []}']) as gw:
        service = ReflectionService(lambda: uow, assist=gw.assist(uses=["summaries"]))
        assert await service.reflect_all() == []
        assert await service.reflect("acme", "user:u1", sources) == []
    assert gw.route.call_count == 0
    assert await ReflectionService(lambda: uow).reflect_all() == []


def test_reflect_job_is_registered_only_with_the_flag(make_settings) -> None:
    from memory_service.adapters.tasks.inline_queue import RecordingTaskQueue

    def handlers(**llm):
        settings = make_settings(models={"llm": llm})
        container = Container(settings=settings, version="test")
        container.tasks = RecordingTaskQueue()
        container.services["uow_factory"] = None
        register_handlers(container)
        return container.tasks

    off = handlers(enabled=False)
    assert "memory.reflect" not in off.handlers and "periodic.memory_reflect" not in off.periodic
    enabled = {"enabled": True, "model": "test/strong"}
    without_use = handlers(**enabled, uses=["summaries"])
    assert "memory.reflect" not in without_use.handlers
    on = handlers(**enabled, uses=["reflection"])
    assert "memory.reflect" in on.handlers and "periodic.memory_reflect" in on.periodic
