"""LLM-assisted memory paths against the mocked gateway: conflict adjudication and
reflection. Every use is checked three ways: the
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
        self.updated = []
        self.reflected = {}

    async def list_recent(self, *, since, limit=1000):
        return [m for m in self.recent if m.updated_at >= since][:limit]

    async def reflection_pending(self, *, limit=1000, tenant_id=None):
        return [
            m
            for m in self.recent
            if not m.system_metadata.get("source_revisions")
            and (tenant_id is None or m.tenant_id == tenant_id)
            and self.reflected.get(m.memory_id) != m.revision
        ][:limit]

    async def mark_reflected(self, sources, *, at):
        self.reflected.update({m.memory_id: m.revision for m in sources})

    async def related(
        self,
        tenant_id,
        *,
        scope_key,
        subject,
        owner_principal,
        visibility_keys,
        exclude=(),
        limit=8,
        include_derived=True,
        include_verbatim=False,
    ):
        return [
            m
            for m in self.recent
            if m.tenant_id == tenant_id
            and m.scope.key() == scope_key
            and m.subject == subject
            and m.owner_principal == owner_principal
            and set(m.system_metadata.get("visibility_keys", [])) == set(visibility_keys)
            and m.memory_id not in exclude
            and (include_derived or not m.system_metadata.get("source_revisions"))
        ][:limit]

    async def candidates(
        self, tenant_id, *, scope_key, normalized_hash=None, subject=None, limit=20
    ):
        return [m for m in self.added if m.scope.key() == scope_key][:limit]

    async def current_derived(self, tenant_id, slot):
        return next(
            (
                m
                for m in self.added
                if m.tenant_id == tenant_id and m.system_metadata.get("derived_slot") == slot
            ),
            None,
        )

    async def add(self, memory, *, visibility_keys):
        memory.system_metadata["visibility_keys"] = list(visibility_keys)
        self.added.append(memory)
        self.recent.insert(0, memory)

    async def update(self, memory):
        """Persist in place and bump the row revision, as the repository does."""
        self.updated.append(memory.memory_id)
        memory.revision += 1
        self.recent = [memory if m.memory_id == memory.memory_id else m for m in self.recent]

    async def get_many(self, tenant_id, memory_ids):
        wanted = set(memory_ids)
        return [m for m in self.recent if m.tenant_id == tenant_id and m.memory_id in wanted]


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

    async def serialize(self, *keys):
        return None

    async def enqueue(self, spec):
        self.enqueued.append(spec)
        return 1

    async def commit(self):
        self.commits += 1


async def _sources(*texts: str, ctx=CTX, age=timedelta(hours=1)):
    plain = _native()
    now = datetime.now(UTC) - age
    from memory_service.modules.memory.pipeline import keys_for

    memories = [build_memory(await _first(plain, t, ctx), ctx, now=now) for t in texts]
    for memory in memories:
        memory.system_metadata["visibility_keys"] = keys_for(memory.scope, memory.visibility, ctx)
    return memories


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
        schema = gw.prompts()[0]["response_format"]["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["insights"]["items"]["additionalProperties"] is False
        prompt = gw.prompts()[0]["messages"][1]["content"]
        assert all(i in prompt for i in ids) and "user:u1" in prompt
        assert len(created) == 1 and uow.memories.added[0].memory_id == created[0]
        insight = uow.memories.added[0]
        assert insight.content == "User prefers terse, structured answers"
        assert insight.memory_type is MemoryType.PREFERENCE and insight.confidence == 0.6
        assert insight.owner_principal == "user:u1" and insight.visibility is sources[0].visibility
        assert insight.scope.level.value == "USER" and insight.scope.user_id == "u1"
        assert sorted(e.source_id for e in insight.evidence) == sorted(ids)
        assert all(e.source_type == "memory" for e in insight.evidence)
        assert insight.system_metadata["category"] == "reflection"
        assert insight.system_metadata["source_memory_ids"] == sorted(ids)
        assert insight.system_metadata["contributors"] == ["user:u1"]
        assert insight.system_metadata["visibility_keys"] == sorted(
            sources[0].system_metadata["visibility_keys"]
        )
        assert [j.task_name for j in uow.enqueued] == ["memory.index"]
        assert uow.enqueued[0].payload == {"tenant_id": "acme", "memory_ids": created}
        assert uow.revisions.bumped and uow.commits == 1
        # the scope has nothing newer than its last insight: no second consultation
        assert await service.reflect_all() == []
        assert gw.route.call_count == 1
        # an identical insight for the same scope is not stored twice
        assert await service.reflect("acme", "user:u1", sources) == []
        assert gw.route.call_count == 2 and len(uow.memories.added) == 1


async def test_reflection_bounds_batches_and_processes_older_pending_sources() -> None:
    fresh = await _sources(*[f"The widget {i} costs {i} USD." for i in range(45)])
    old = await _sources("I prefer tabs.", age=timedelta(days=3))
    uow = _UoW(_Memories(fresh + old))
    with mocked_gateway(['{"insights": []}'] * 4) as gw:
        service = ReflectionService(lambda: uow, assist=gw.assist(uses=["reflection"]))
        assert await service.reflect_all() == []
        assert await service.reflect_all() == []
        assert gw.route.call_count == 3
    prompts = [p["messages"][1]["content"] for p in gw.prompts()]
    assert all(prompt.count("\n- ") <= 40 for prompt in prompts)
    assert len(uow.memories.reflected) == 46
    assert uow.enqueued == []


async def test_reflection_is_a_no_op_when_gateway_fails_or_flag_off() -> None:
    sources = await _sources("I prefer concise answers.", "I prefer bullet points.")
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


# --- dedup batching -----------------------------------------------------------------------


class _CountingEmbedding:
    """A real-looking encoder that counts how it is entered, not what it returns."""

    representative = True
    dimension = 4

    def __init__(self) -> None:
        self.queries = 0
        self.batches = 0
        self.texts = 0

    def fingerprint(self) -> str:
        # not "hash-", so the dense branch is live
        return "st-counting-d4"

    async def embed_query(self, text: str) -> list[float]:
        self.queries += 1
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts) -> list[list[float]]:
        self.batches += 1
        self.texts += len(texts)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


async def test_the_dense_band_is_embedded_in_one_batch() -> None:
    """Every memory in the band used to cost its own single-text round trip.

    Up to ``dedup_candidate_k`` (20) of them per candidate, through the encoder's one-caller
    gate, with the pipeline walking candidates sequentially as well - the most expensive
    component in the service driven at batch size one. The port has exposed
    ``embed_documents`` the whole time.
    """
    cand, existing = await _pair(EXISTING, NEAR)
    band = existing * 3
    embedding = _CountingEmbedding()
    provider = NativeMemoryIntelligence(MemoryIntelligenceSettings(), embedding=embedding)

    await provider.consolidate(cand, band, CTX)

    assert embedding.queries == 0, "a per-memory embed_query is the defect this replaces"
    assert embedding.batches == 1, f"entered the encoder {embedding.batches} times, expected 1"
    assert embedding.texts == 1 + len(band), "the candidate and the whole band go in together"


async def test_generated_insight_cannot_reinforce_a_new_source_fact() -> None:
    provider = _native()
    candidate = await _first(provider, "I prefer concise answers.", CTX)
    generated = build_memory(candidate, CTX, now=datetime.now(UTC))
    generated.system_metadata["source_revisions"] = {"source-1": 1, "source-2": 1}
    outcome = await provider.consolidate(candidate, [generated], CTX)
    assert outcome.decision is DedupDecision.CREATE
