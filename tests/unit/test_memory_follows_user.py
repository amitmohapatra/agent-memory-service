"""What a user says is theirs, kept as they said it.

Two guarantees behind single-fact recall across sessions ("The master lock code for the
hazardous materials cage in Warehouse 3 is 8492" on Monday, "what's the code?" in a new chat
on Tuesday): the statement's default audience is the user, not the conversation it was said
in, and the user's exact words survive whatever a rule makes of them.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    DedupDecision,
    Lifetime,
    MemoryType,
    ObservationKind,
    Visibility,
)
from memory_service.domain.ids import content_hash
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.observation import Observation
from memory_service.modules.agent_tools.service import AgentTools
from memory_service.modules.memory.native import (
    VERBATIM_MAX_PIECES,
    NativeMemoryIntelligence,
    default_visibility,
    said_by_user,
    verbatim_windows,
)
from memory_service.modules.memory.pipeline import (
    ObservationPipeline,
    _told,
    authored_type,
    build_memory,
)
from memory_service.modules.memory.service import MemoryService
from memory_service.modules.retrieval.engine import (
    Candidate,
    _dedup,
    in_hand,
    memory_candidate,
    not_in_hand,
)

pytestmark = pytest.mark.unit

USER = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="t1")
HARNESS = USER.model_copy(update={"agent_id": "planner", "agent_run_id": "run_1"})
CODE = "The master lock code for the hazardous materials cage in Warehouse 3 is 8492."


def _obs(
    text: str,
    ctx: MemoryExecutionContext = USER,
    *,
    role: str | None = "USER",
    kind: ObservationKind = ObservationKind.MESSAGE,
) -> Observation:
    return Observation(
        tenant_id=ctx.tenant_id,
        kind=kind,
        content=text,
        content_hash=content_hash(text),
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        thread_id=ctx.thread_id,
        agent_id=ctx.agent_id,
        agent_run_id=ctx.agent_run_id,
        principal_id=ctx.principal_id,
        message_id="msg_1",
        custom_metadata={"role": role} if role else {},
    )


@pytest.fixture
def native() -> NativeMemoryIntelligence:
    return NativeMemoryIntelligence(MemoryIntelligenceSettings())


async def _classified(native, observation: Observation, ctx: MemoryExecutionContext = USER):
    return [await native.classify(c, ctx) for c in await native.extract(observation, ctx)]


# --- whose words ---------------------------------------------------------------------------


def test_a_users_own_words_are_told_apart_from_an_agents() -> None:
    assert said_by_user(_obs(CODE))
    assert said_by_user(_obs(CODE, HARNESS)), "a harness relaying the user's turn"
    assert said_by_user(_obs("Go with Qdrant.", kind=ObservationKind.DECISION, role=None))
    assert not said_by_user(_obs(CODE, role="ASSISTANT")), "the assistant's reply"
    assert not said_by_user(_obs(CODE, HARNESS, role="AGENT"))
    assert not said_by_user(_obs("Draft plan.", HARNESS, kind=ObservationKind.AGENT_RESULT))
    assert not said_by_user(_obs("Deploy finished.", kind=ObservationKind.EVENT)), "a record"
    service = USER.model_copy(update={"user_id": None})
    assert not said_by_user(_obs(CODE, service)), "nobody to follow"


@pytest.mark.parametrize(
    "mtype",
    [MemoryType.SEMANTIC, MemoryType.TASK, MemoryType.EPISODIC, MemoryType.PROCEDURAL],
)
def test_a_users_statement_defaults_to_the_user_in_any_thread(mtype) -> None:
    assert default_visibility(mtype, USER, user_statement=True) is Visibility.USER
    assert default_visibility(mtype, HARNESS, user_statement=True) is Visibility.USER, (
        "relayed by an agent is still the user's, not the agent's private note"
    )
    # not the user's words: the conversation's, as before
    assert default_visibility(mtype, USER) is Visibility.THREAD


def test_an_agents_own_writing_keeps_its_rules() -> None:
    assert default_visibility(MemoryType.TASK, HARNESS) is Visibility.PRIVATE
    assert default_visibility(MemoryType.EPISODIC, HARNESS) is Visibility.PRIVATE
    assert default_visibility(MemoryType.SEMANTIC, HARNESS) is Visibility.THREAD
    assert default_visibility(MemoryType.AGENT, HARNESS, user_statement=True) is Visibility.RUN
    # explicitly shared types keep their audience whoever wrote them
    assert default_visibility(MemoryType.SHARED, USER, user_statement=True) is Visibility.THREAD


async def test_every_memory_of_a_users_message_follows_the_user(native) -> None:
    said = await _classified(native, _obs(CODE))
    assert said and all(c.visibility is Visibility.USER for c in said)
    reply = await _classified(native, _obs("The cage code is 8492.", role="ASSISTANT"))
    assert reply and all(c.visibility is Visibility.THREAD for c in reply), "the assistant's words"


# --- the words as said -----------------------------------------------------------------------


async def test_a_long_named_subject_is_a_fact_and_the_turn_is_kept(native) -> None:
    fact, turn = await _classified(native, _obs(CODE))
    assert fact.category == "fact" and fact.object == "8492"
    assert fact.subject == "master lock code for the hazardous materials cage in warehouse 3"
    assert turn.category == "verbatim_turn" and turn.content == CODE
    assert turn.evidence == fact.evidence, "one source, two readings"


@pytest.mark.parametrize(
    "text",
    [
        "The kids and I went to the beach yesterday and it was great.",
        "The new guy that we hired from Berlin last week is great.",
        "Deployed the payment service to prod and the dashboard is green.",
        "Yesterday John Smith sent the report and the deadline is Friday.",
        "The report John Smith sent with his notes on Q3 is due Friday.",
    ],
)
async def test_a_clause_is_not_read_as_a_long_subject(native, text) -> None:
    cands = await _classified(native, _obs(text))
    assert all(c.category != "fact" for c in cands), [c.subject for c in cands]
    assert cands[-1].category == "verbatim_turn"


async def test_a_short_lived_reading_does_not_take_the_turn_with_it(native) -> None:
    """A rule may read an instruction as a task: the task lapses in seven days, the user's
    sentence does not."""
    text = "The seasonal holiday merchandise must be routed to Overflow Storage Facility B."
    task, turn = await _classified(native, _obs(text))
    assert task.memory_type is MemoryType.TASK and task.lifetime is Lifetime.SHORT_TERM
    assert turn.category == "verbatim_turn" and turn.lifetime is Lifetime.LONG_TERM
    assert turn.content == text and turn.visibility is Visibility.USER
    # said as a universal obligation it is a standing rule (ADR 0037): a lasting reading that
    # keeps the words itself, so no twin is needed
    [rule] = await _classified(native, _obs("All " + text[4:]))
    assert rule.category == "rule" and rule.lifetime is Lifetime.LONG_TERM


async def test_a_reading_and_its_turn_never_reinforce_each_other(native) -> None:
    now = datetime.now(UTC)
    first = await _classified(native, _obs(CODE))
    stored = [build_memory(c, USER, now=now) for c in first]
    again = await _classified(native, _obs(CODE))
    outcomes = [await native.consolidate(c, stored, USER) for c in again]
    assert [o.decision for o in outcomes] == [DedupDecision.REINFORCE] * 2
    # each reinforces its own kind: the fact the fact, the turn the turn
    assert [o.target_memory_id for o in outcomes] == [m.memory_id for m in stored]
    # and within one message neither absorbs the other
    fact, turn = first
    assert (await native.consolidate(turn, stored[:1], USER)).decision is DedupDecision.CREATE
    assert (await native.consolidate(fact, stored[1:], USER)).decision is DedupDecision.CREATE


def test_a_long_turn_is_kept_whole_in_sentence_pieces() -> None:
    filler = "Pallet counts for aisle four were checked against the manifest. " * 40
    text = f"{filler}The new dock door code is 7731. {filler}".strip()
    pieces = verbatim_windows(text, 2000)
    assert len(pieces) > 1 and all(len(p) <= 2000 for p in pieces)
    assert all(p in text for p in pieces), "every piece is a slice of what was said"
    assert any("The new dock door code is 7731." in p for p in pieces), "cut at sentence ends"
    assert "".join(text.split()) == "".join("".join(pieces).split()), "nothing dropped"
    assert verbatim_windows("A short turn.", 2000) == ["A short turn."]
    # one unbroken sentence longer than a piece is cut at a space, still losing nothing
    long_one = " ".join(["SKU-12345"] * 300)
    assert "".join("".join(verbatim_windows(long_one, 200)).split()) == "".join(long_one.split())


async def test_a_long_noun_phrase_opened_by_a_possessive_is_still_a_fact(native) -> None:
    text = "Our primary supplier for heavy-duty pallets is Uline."
    fact, _turn = await _classified(native, _obs(text))
    assert fact.category == "fact" and fact.object == "uline"


async def test_a_pasted_corpus_keeps_a_bounded_number_of_pieces(native) -> None:
    """A 2.1-million-character turn is not a statement to keep for good a thousand times
    over: at most ``VERBATIM_MAX_PIECES`` pieces, the first ones, every one a slice of it."""
    text = "Pallet 4471 was counted against the manifest in aisle seven today. " * 31_000
    assert len(text) > 2_000_000
    turns = [c for c in await _classified(native, _obs(text)) if c.category == "verbatim_turn"]
    assert 1 < len(turns) == VERBATIM_MAX_PIECES
    assert all(len(t.content) <= 2000 and t.content in text for t in turns)
    assert text.startswith(turns[0].content)


async def test_a_code_said_after_two_thousand_characters_is_kept(native) -> None:
    filler = "Pallet counts for aisle four were checked against the manifest. " * 40
    text = f"{filler}The new dock door code is 7731."
    turns = [c for c in await _classified(native, _obs(text)) if c.category == "verbatim_turn"]
    assert len(turns) == 2 and "7731" in turns[-1].content


# --- one statement, one slot -----------------------------------------------------------------


def _hit(record_id: str, subject: str, score: float) -> Candidate:
    return Candidate(
        record_id=record_id,
        kind="memory",
        text=CODE,
        score=score,
        retrievers=["fusion"],
        payload={
            "subject": subject,
            "owner_principal": "user:u1",
            "observed_at": "2026-10-06T07:40:00Z",
            "source_refs": [{"source_type": "message", "source_id": "msg_1"}],
        },
    )


def test_a_fact_and_its_turn_take_one_slot() -> None:
    fact = _hit("fact", "master lock code for the hazardous materials cage in warehouse 3", 0.9)
    turn = _hit("turn", "user:u1", 0.7)
    [kept] = _dedup([fact, turn])
    assert kept.record_id == "fact" and kept.payload["duplicates"] == ["turn"]


def test_equal_words_of_different_speakers_still_take_two() -> None:
    assert len(_dedup([_hit("a", "user:u1", 0.9), _hit("b", "user:u2", 0.8)])) == 2


def test_a_collapsed_twin_counts_as_in_hand() -> None:
    """Graph and derived-source expansion add memories by id; the twin ranking collapsed is
    already in hand under its survivor and must not come back as a second copy."""
    fact = _hit("fact", "master lock code for the hazardous materials cage in warehouse 3", 0.9)
    kept = _dedup([fact, _hit("turn", "user:u1", 0.7)])
    assert in_hand(kept) == {"fact", "turn"}


def test_a_turn_relayed_by_a_harness_and_its_fact_take_one_slot() -> None:
    """Through a harness the turn names the user and the fact's writer is the harness agent
    bound to that user: both are the user's words, one statement."""
    fact = _hit("fact", "master lock code for the hazardous materials cage in warehouse 3", 0.9)
    turn = _hit("turn", "user:u1", 0.7)
    for hit in (fact, turn):
        hit.payload["owner_principal"] = "agent:u1/harness"
    [kept] = _dedup([fact, turn])
    assert kept.record_id == "fact" and kept.payload["duplicates"] == ["turn"]
    # another person's agent saying the same words from the same source is not this user
    other = _hit("other", "user:u2", 0.6)
    other.payload["owner_principal"] = "agent:u2/harness"
    assert len(_dedup([fact, other])) == 2


def test_an_expansion_does_not_add_the_copy_of_a_statement_in_hand() -> None:
    """Graph evidence or a derived memory's sources can reach the fact when ranking kept
    only the turn; the fact is the same statement and does not take a second slot."""
    from memory_service.domain.enums import ScopeLevel
    from memory_service.domain.evidence import EvidenceRef, EvidenceSource
    from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState

    when = datetime(2026, 10, 6, 7, 40, tzinfo=UTC)
    source = EvidenceRef(
        source_type=EvidenceSource.MESSAGE, source_id="msg_1", message_id="msg_1", observed_at=when
    )

    def memory(category: str, subject: str, content: str = CODE) -> CanonicalMemory:
        return CanonicalMemory(
            tenant_id="acme",
            scope=Scope(level=ScopeLevel.THREAD, tenant_id="acme", thread_id="t1"),
            visibility=Visibility.USER,
            owner_principal="agent:u1/harness",
            lifetime=Lifetime.LONG_TERM,
            memory_type=MemoryType.SEMANTIC if category == "fact" else MemoryType.OBSERVATION,
            content=content,
            normalized_hash=content_hash(content),
            subject=subject,
            temporal=TemporalState(observed_at=when),
            evidence=[source],
            system_metadata={"category": category},
        )

    turn = memory_candidate(memory("verbatim_turn", "user:u1"), retriever="fusion", score=0.9)
    fact = memory_candidate(memory("fact", "cage code"), retriever="graph", score=0.4)
    other = memory_candidate(
        memory("fact", "dock door", "The dock door code is 7731."), retriever="graph", score=0.4
    )
    assert turn.payload["owner_principal"] == "agent:u1/harness"
    assert not_in_hand([fact, other], [turn]) == [other]


# --- what a run's transcript forms ---------------------------------------------------------


class _Observations:
    def __init__(self, observation: Observation) -> None:
        self.observation, self.status = observation, None

    async def get(self, tenant_id, observation_id):
        return self.observation if self.status is None else None

    async def mark_processed(self, tenant_id, observation_id, *, status) -> None:
        self.status = status


class _Uow:
    def __init__(self, observations: _Observations) -> None:
        self.observations = observations

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def commit(self) -> None:
        return None


class _Unreachable:
    async def extract(self, *_: object) -> list:
        raise AssertionError("nothing is extracted from it")


@pytest.mark.parametrize(
    ("role", "text"),
    [
        ("ASSISTANT", "The supplier system is unavailable, so we cannot determine the ETA."),
        ("TOOL", "supplier_eta failed: the supplier system is unavailable"),
        ("TOOL", '[{"id": "mem_1", "kind": "memory", "text": "S12 sells 10 a week"}]'),
    ],
)
async def test_an_agent_runs_own_words_and_its_tools_results_form_no_memory(role, text) -> None:
    """B15: replayed into a fresh tenant, a real run's transcript stored its answers, hedges,
    tool errors and raw tool JSON as facts. They stay in the transcript, and form nothing."""
    observations = _Observations(_obs(text, HARNESS, role=role))
    pipeline = ObservationPipeline(
        lambda: _Uow(observations),  # type: ignore[arg-type,return-value]
        _Unreachable(),  # type: ignore[arg-type]
        settings=MemoryIntelligenceSettings(),
    )
    assert await pipeline.run({"tenant_id": "acme", "observation_id": "obs_1"}) == []
    assert observations.status == "NO_MEMORY"


def test_a_persons_words_an_event_and_an_agents_result_still_form_memories() -> None:
    assert _told(_obs(CODE, HARNESS)), "the user's turn the harness relays"
    assert _told(_obs("Deploy finished.", HARNESS, role="EVENT"))
    assert _told(_obs(CODE, role="ASSISTANT")), "a chat app's assistant, outside any run"
    assert _told(_obs("Draft plan.", HARNESS, kind=ObservationKind.AGENT_RESULT, role=None))
    assert not _told(_obs("Thinking: UTC.", HARNESS, role="AGENT"))


#: The same three kinds of a run's own writing as above, in five languages: whose words they
#: are, not the language they are in, decides that they form no memory.
_RUN_OWN_WORDS = {
    "de": (
        "Das Lieferantensystem ist nicht erreichbar, daher kann ich die Lieferzeit nicht nennen.",
        "supplier_eta fehlgeschlagen: das Lieferantensystem ist nicht erreichbar",
    ),
    "es": (
        "El sistema del proveedor no está disponible, así que no puedo dar la fecha de entrega.",
        "supplier_eta falló: el sistema del proveedor no está disponible",
    ),
    "hi": (
        "आपूर्तिकर्ता प्रणाली उपलब्ध नहीं है, इसलिए मैं डिलीवरी का समय नहीं बता सकता।",
        "supplier_eta विफल: आपूर्तिकर्ता प्रणाली उपलब्ध नहीं है",
    ),
    "ar": (
        "نظام المورد غير متاح، لذلك لا أستطيع تحديد موعد التسليم.",
        "فشل supplier_eta: نظام المورد غير متاح",
    ),
    "zh": (
        "供应商系统不可用，因此我无法确定交货时间。",  # noqa: RUF001
        "supplier_eta 失败：供应商系统不可用",  # noqa: RUF001
    ),
}
#: A person's own statement in each language: still theirs to remember.
_PERSONS_WORDS = {
    "de": "Der Code für den Gefahrstoffkäfig in Lager 3 ist 8492.",
    "es": "El código de la jaula de materiales peligrosos del almacén 3 es 8492.",
    "hi": "गोदाम 3 में खतरनाक सामग्री के पिंजरे का कोड 8492 है।",
    "ar": "رمز قفص المواد الخطرة في المستودع 3 هو 8492.",
    "zh": "3号仓库危险品笼的密码是8492。",
}


@pytest.mark.parametrize("language", sorted(_RUN_OWN_WORDS))
@pytest.mark.parametrize(("role", "which"), [("ASSISTANT", 0), ("TOOL", 1)])
async def test_a_runs_own_words_form_no_memory_in_any_language(language, role, which) -> None:
    """B15 in de/es/hi/ar/zh: an agent run's hedge and its tool's error stay in the
    transcript and form nothing, whatever language the run wrote them in."""
    text = _RUN_OWN_WORDS[language][which]
    observations = _Observations(_obs(text, HARNESS, role=role))
    pipeline = ObservationPipeline(
        lambda: _Uow(observations),  # type: ignore[arg-type,return-value]
        _Unreachable(),  # type: ignore[arg-type]
        settings=MemoryIntelligenceSettings(),
    )
    assert await pipeline.run({"tenant_id": "acme", "observation_id": "obs_1"}) == []
    assert observations.status == "NO_MEMORY"


@pytest.mark.parametrize("language", sorted(_PERSONS_WORDS))
def test_a_persons_words_relayed_by_a_run_still_form_memories_in_any_language(language) -> None:
    assert _told(_obs(_PERSONS_WORDS[language], HARNESS))
    assert not _told(_obs(_RUN_OWN_WORDS[language][1], HARNESS, role="TOOL"))


# --- a user's question is not a statement (S5) ---------------------------------------------

#: A user turn that only asks, in six languages and the three question marks the service reads
#: (ASCII, full-width, Arabic), and an English one with no mark behind a heading: replaying a run's
#: transcript kept "For employee e-401: how many trips ..." as a lasting memory in each of the
#: four conversations it was asked in.
_ONLY_ASKS = {
    "en": "Which store should I restock first?",
    "en-heading": (
        "For employee e-401: how many trips this year, which flights are booked, and what the "
        "travel policy says about the meal allowance."
    ),
    "de": "Welche Filiale soll ich zuerst auffüllen?",
    "es": "¿Qué tienda debo reabastecer primero？",  # noqa: RUF001 - the full-width mark
    "ar": "أي متجر يجب أن أعيد تعبئته أولاً؟",
    "hi": "मुझे पहले किस स्टोर को भरना चाहिए?",
    "zh": "我应该先给哪家店补货？",  # noqa: RUF001 - the full-width mark
}


@pytest.mark.parametrize("ctx", [USER, HARNESS], ids=["direct", "relayed"])
@pytest.mark.parametrize("which", sorted(_ONLY_ASKS))
async def test_a_turn_that_only_asks_forms_no_memory(native, which, ctx) -> None:
    assert await native.extract(_obs(_ONLY_ASKS[which], ctx), ctx) == []


@pytest.mark.parametrize(
    ("text", "fact"),
    [
        ("I moved to Pune. Which store is nearest?", "pune"),
        ("Ich wohne in Pune. Welche Filiale ist am nächsten?", None),
        ("Stores S12 and S40 both carry style ST-101. Which should get stock first?", None),
        ("For employee e-401: the meal allowance is 40 EUR a day.", None),
    ],
)
async def test_a_turn_that_asks_and_states_keeps_what_it_states(native, text, fact) -> None:
    """Questions are told apart per sentence: the turn is kept whole when any sentence of it
    states something, with what a rule read from that sentence."""
    cands = await native.extract(_obs(text, HARNESS), HARNESS)
    assert [c.content for c in cands if c.category == "verbatim_turn"] == [text]
    if fact:
        assert any(c.object == fact for c in cands if c.category != "verbatim_turn")


# --- what an agent states about the user ---------------------------------------------------


def test_only_the_user_states_the_users_own_kinds() -> None:
    for kind in (MemoryType.USER, MemoryType.PREFERENCE):
        assert authored_type(kind, by_agent=False) is kind
        assert authored_type(kind, by_agent=True) is MemoryType.AGENT
    for kind in (MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.TASK, MemoryType.AGENT):
        assert authored_type(kind, by_agent=True) is kind


class _Rows:
    def __init__(self) -> None:
        self.added: list[CanonicalMemory] = []

    async def current_with_hash(self, *_: object, **__: object) -> None:
        return None

    async def add(self, memory: CanonicalMemory, **_: object) -> None:
        self.added.append(memory)


class _Ignored:
    async def bump(self, *_: object, **__: object) -> None:
        return None


class _Stated:
    """The unit of work a stated memory is written through: the rows it adds."""

    def __init__(self) -> None:
        self.memories, self.revisions = _Rows(), _Ignored()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def serialize(self, *_: object) -> None:
        return None

    async def enqueue(self, *_: object) -> int:
        return 1

    async def commit(self) -> None:
        return None


async def test_an_agents_remembered_user_fact_is_the_agents_not_the_users_profile() -> None:
    """A help-desk run looked up an asset's owner and called memory_remember(kind="USER",
    scope="user", content="omar (u-77)"): it was kept as the user's USER memory and became
    the pinned "### user" block, so every later context told the model the user is Omar."""
    uow = _Stated()
    tools = AgentTools(lambda: uow, {"memory": MemoryService(None)})  # type: ignore[arg-type]
    await tools.call(
        HARNESS, "memory_remember", {"content": "omar (u-77)", "kind": "USER", "scope": "user"}
    )
    await tools.call(
        HARNESS, "memory_remember", {"content": "Ann likes tables", "kind": "PREFERENCE"}
    )
    await tools.call(HARNESS, "memory_remember", {"content": "LT-2290 is Omar's"})
    omar, likes, fact = uow.memories.added
    for agents in (omar, likes):
        assert agents.memory_type is MemoryType.AGENT, "never the profile's kinds"
        assert agents.subject is None and agents.owner_principal == HARNESS.principal_id
        assert agents.visibility is Visibility.USER, "still read where the agent asked"
    assert fact.memory_type is MemoryType.SEMANTIC, "any other kind is kept as stated"


@pytest.mark.parametrize("kind", [MemoryType.USER, MemoryType.PREFERENCE])
async def test_the_users_own_statement_still_states_their_profile(kind) -> None:
    uow = _Stated()
    await MemoryService(None).remember(  # type: ignore[arg-type]
        uow,  # type: ignore[arg-type]
        USER,
        content="My name is Omar.",
        memory_type=kind,
        lifetime=Lifetime.LONG_TERM,
    )
    [said] = uow.memories.added
    assert said.memory_type is kind and said.subject == "user:u1"
    assert said.visibility is Visibility.USER
