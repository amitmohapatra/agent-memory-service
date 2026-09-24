"""Unit tests for native memory intelligence: extraction rules, classification defaults and
consolidation decisions."""

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
from memory_service.domain.observation import Observation, ProcessingHints
from memory_service.modules.memory.native import (
    NativeMemoryIntelligence,
    normalize,
    parse_date,
    split_clauses,
    split_sentences,
    strip_turn_prefix,
    tokens,
)
from memory_service.modules.memory.pipeline import build_memory, keys_for, scope_for

pytestmark = pytest.mark.unit

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1")
AGENT = CTX.model_copy(update={"agent_id": "planner", "agent_run_id": "run_1"})


def _obs(
    text: str, kind: ObservationKind = ObservationKind.MESSAGE, ctx=CTX, **hints
) -> Observation:
    return Observation(
        tenant_id=ctx.tenant_id,
        kind=kind,
        content=text,
        content_hash=content_hash(text),
        user_id=ctx.user_id,
        thread_id=ctx.thread_id,
        workspace_id=ctx.workspace_id,
        agent_id=ctx.agent_id,
        agent_run_id=ctx.agent_run_id,
        principal_id=ctx.principal_id,
        message_id="msg_1",
        hints=ProcessingHints(**hints),
    )


@pytest.fixture
def native() -> NativeMemoryIntelligence:
    return NativeMemoryIntelligence(MemoryIntelligenceSettings())


async def _extract(native, text, **kw):
    return [
        await native.classify(c, kw.get("ctx", CTX))
        for c in await native.extract(_obs(text, **kw), kw.get("ctx", CTX))
    ]


def _extracted(cands):
    """Rule- or assist-extracted candidates only, without the verbatim turn.

    `keep_verbatim_turns` makes every substantive MESSAGE also produce an OBSERVATION copy
    of itself, so that a turn no rule could parse is still retrievable. These tests are
    about what extraction *understood*, which is a different question, and the guarantee
    they protect is unchanged: noise never becomes an asserted fact. A verbatim turn is an
    OBSERVATION, which DERIVED_MEMORY_TYPES excludes from supersession and reflection.
    """
    return [c for c in cands if c.category != "verbatim_turn"]


# --- text utilities --------------------------------------------------------------------


def test_text_utils() -> None:
    assert normalize("  My Timezone is CET. ") == "my timezone is cet"
    assert tokens("I prefer tabs over spaces.") == {"prefer", "tabs", "over", "spaces"}
    assert split_sentences("First one here. Second one there!\nThird line here. ok") == [
        "First one here.",
        "Second one there!",
        "Third line here.",
    ]
    assert split_clauses("My name is Amit and my timezone is CET") == [
        "My name is Amit",
        "my timezone is CET",
    ]
    assert split_clauses("We decided to use Postgres and Redis") == [
        "We decided to use Postgres and Redis"
    ]
    assert parse_date("as of 2026-09-01") == datetime(2026, 9, 1, tzinfo=UTC)
    assert parse_date("since September 2026") == datetime(2026, 9, 1, tzinfo=UTC)
    assert parse_date("in Q3 2026") == datetime(2026, 7, 1, tzinfo=UTC)
    assert parse_date("no date here") is None


def test_a_transcript_header_is_not_part_of_the_first_sentence() -> None:
    # Sentences split on .!? alone, so a forwarded chat line's header stays glued to the
    # first sentence of the turn and every rule anchored at ^ - the fact pattern's subject,
    # "please always/never", "decision:", the chit-chat filter - misses it.
    assert strip_turn_prefix("[1:56 pm on 8 May, 2023] Caroline: I moved to Paris.") == (
        "I moved to Paris."
    )
    assert strip_turn_prefix("[2023-05-08] we shipped it") == "we shipped it"
    assert split_sentences(
        "[1:56 pm on 8 May, 2023] Caroline: Thanks for that. I moved to Paris."
    ) == ["Thanks for that.", "I moved to Paris."]
    assert strip_turn_prefix("[1:56 pm on 8 May, 2023] Mary Jane Smith: hello there") == (
        "hello there"
    )
    # a prefix this module understands is not a header and must survive
    assert strip_turn_prefix("Decision: ship it on Friday") == "Decision: ship it on Friday"
    assert strip_turn_prefix("no header at all") == "no header at all"


def test_a_header_eats_neither_a_caption_nor_the_first_clause_of_prose() -> None:
    # An image caption is bracketed like a header and, for ten answerable LoCoMo questions,
    # holds the only copy of the answer. A digit in the bracket is not enough to tell the
    # two apart: a caption counts things and names model years.
    for caption in (
        "[Shared an image of a dog: a labrador on a beach]",
        "[Shared an image: 2 dogs]",
        "[Shared an image of 2 women: two women smiling]",
        "[Shared an image of a 2018 car: a red mustang]",
    ):
        assert strip_turn_prefix(caption) == caption
    # After a real header the speaker is a name, so ordinary prose that happens to contain a
    # colon keeps its first clause - the subject, the place and the fact all live there.
    assert strip_turn_prefix("[8 May, 2023] I moved to Paris: it was great.") == (
        "I moved to Paris: it was great."
    )
    assert strip_turn_prefix("[2023-05-08] Caroline: I moved to Paris: it was great.") == (
        "I moved to Paris: it was great."
    )


# --- extraction ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "mtype", "predicate", "obj"),
    [
        ("My timezone is Europe/Berlin.", MemoryType.USER, "timezone", "europe/berlin"),
        ("My name is Amit.", MemoryType.USER, "name", "amit"),
        ("Call me Ricky.", MemoryType.PREFERENCE, "name", "ricky"),
        ("I work at ACME Corp.", MemoryType.USER, "works_at", "acme corp"),
        ("I'm a staff engineer at ACME.", MemoryType.USER, "role", "staff engineer at acme"),
        ("I prefer concise answers.", MemoryType.PREFERENCE, "prefers", "concise answers"),
        ("I don't like verbose output.", MemoryType.PREFERENCE, "dislikes", "verbose output"),
        ("My favourite editor is neovim.", MemoryType.PREFERENCE, "favourite_editor", "neovim"),
        ("Please never use emojis.", MemoryType.PREFERENCE, "instruction", "never use emojis"),
        ("We decided to use PostgreSQL.", MemoryType.SEMANTIC, "decided", "use postgresql"),
        (
            "Revenue was EUR 412 million in FY26.",
            MemoryType.SEMANTIC,
            "was",
            "eur 412 million in fy26",
        ),
    ],
)
async def test_extraction_rules(native, text, mtype, predicate, obj) -> None:
    cands = await _extract(native, text)
    assert len(cands) == 1, cands
    c = cands[0]
    assert c.memory_type is mtype and c.predicate == predicate and c.object == obj
    assert c.evidence[0].message_id == "msg_1" and c.evidence[0].source_type == "message"


async def test_extraction_skips_noise_and_questions(native) -> None:
    for text in ("What is my timezone?", "Thanks!", "ok", "it was a long day and nothing worked"):
        assert _extracted(await _extract(native, text)) == [], text
    assert await _extract(native, "My timezone is CET.", skip_extraction=True) == []


async def test_extraction_kinds_and_temporal(native) -> None:
    tool = await _extract(native, "{'rows': 3}", kind=ObservationKind.TOOL_RESULT, ctx=AGENT)
    assert tool[0].memory_type is MemoryType.TOOL and tool[0].lifetime is Lifetime.SHORT_TERM
    assert tool[0].visibility is Visibility.RUN  # AGENT carries a run id
    agent = await _extract(
        native, "Draft plan: split by region.", kind=ObservationKind.AGENT_RESULT, ctx=AGENT
    )
    assert agent[0].memory_type is MemoryType.AGENT and agent[0].subject == "agent:planner"
    decision = await _extract(
        native, "Go with Qdrant for retrieval.", kind=ObservationKind.DECISION
    )
    assert decision[0].predicate == "decided" and decision[0].subject == "thread:thr_1"
    assert decision[0].visibility is Visibility.THREAD and decision[0].importance == 0.8
    tz = await _extract(native, "Actually, my timezone is now America/New_York since 2026-09-01.")
    assert tz[0].negates_prior and tz[0].valid_from == datetime(2026, 9, 1, tzinfo=UTC)
    assert tz[0].object == "america/new_york"
    task = await _extract(native, "Remind me to follow up with legal by Friday.")
    assert task[0].memory_type is MemoryType.TASK and task[0].lifetime is Lifetime.SHORT_TERM
    event = await _extract(native, "Yesterday the deploy failed because of a missing migration.")
    assert event[0].memory_type is MemoryType.EPISODIC
    multi = await _extract(native, "My name is Amit and my timezone is CET. I prefer tea.")
    assert [c.predicate for c in _extracted(multi)] == ["name", "timezone", "prefers"]
    # The turn itself is kept alongside the facts, last, so ranking prefers the parsed fact
    # over the transcript it came from.
    #
    # This used to hold only OUTSIDE a thread: inside one the turn was dropped on the theory
    # that the thread already kept it. It does - for storage, not for search - and measured
    # on LoCoMo in the production shape that cost 57 points of answer recall (0.2026 against
    # 0.7768) and took evidence recall from 0.9785 to 0.6996. Both contexts now keep it, so
    # the two halves of this assertion are deliberately identical.
    assert [c.predicate for c in multi][-1] == "said"
    threadless = CTX.model_copy(update={"thread_id": None})
    loose = await _extract(
        native, "My name is Amit and my timezone is CET. I prefer tea.", ctx=threadless
    )
    assert [c.predicate for c in loose][-1] == "said"
    proc = await _extract(
        native, "To deploy the API, run make release and then check the dashboard."
    )
    assert proc[0].memory_type is MemoryType.PROCEDURAL and proc[0].lifetime is Lifetime.LONG_TERM
    assert proc[0].subject.startswith("workspace:") and proc[0].category == "procedure"
    runbook = await _extract(native, "The runbook for outages is to page the on-call lead first.")
    assert runbook[0].memory_type is MemoryType.PROCEDURAL


async def test_classification_defaults_and_hints(native) -> None:
    pref = (await _extract(native, "I prefer tea."))[0]
    assert pref.visibility is Visibility.USER and pref.lifetime is Lifetime.LONG_TERM
    fact = (await _extract(native, "The billing service runs on Cloud Run."))[0]
    assert fact.visibility is Visibility.THREAD  # thread in context
    no_thread = CTX.model_copy(update={"thread_id": None})
    fact2 = (await _extract(native, "The billing service runs on Cloud Run.", ctx=no_thread))[0]
    assert fact2.visibility is Visibility.USER  # no thread, but a user to anchor on
    # scope anchors follow the type
    assert scope_for(pref, CTX).level.value == "USER"
    assert scope_for(fact, CTX).level.value == "THREAD"
    agent_note = (
        await _extract(native, "Draft plan.", kind=ObservationKind.AGENT_RESULT, ctx=AGENT)
    )[0]
    assert scope_for(agent_note, AGENT).level.value == "AGENT"
    # The author is in the audience of what they wrote, so that losing a membership does not
    # lose access to it - except PRIVATE, where the author IS the audience, and except
    # THREAD, where it made the thread key dead weight: the author matched from any thread,
    # so a memory scoped to one conversation was readable in all of them. The author reaches
    # this row through the thread grant now, which the write itself creates.
    assert keys_for(scope_for(fact, CTX), fact.visibility, CTX) == ["thread:acme/thr_1"]
    assert keys_for(scope_for(agent_note, AGENT), Visibility.PRIVATE, AGENT) == [
        "principal:acme/agent:u1/planner"
    ]
    now = datetime.now(UTC)
    mem = build_memory(agent_note, AGENT, now=now)
    assert mem.system_metadata["expires_at"] is not None  # SHORT_TERM
    assert build_memory(pref, CTX, now=now).system_metadata["expires_at"] is None


# --- consolidation ------------------------------------------------------------------------


async def _consolidate(native, existing_text: str, incoming_text: str):
    now = datetime.now(UTC)
    existing = [build_memory(c, CTX, now=now) for c in await _extract(native, existing_text)]
    cand = (await _extract(native, incoming_text))[0]
    return await native.consolidate(cand, existing, CTX), existing


@pytest.mark.parametrize(
    ("existing", "incoming", "decision"),
    [
        ("My timezone is CET.", "my timezone is CET", DedupDecision.REINFORCE),
        ("My timezone is CET.", "My timezone is PST.", DedupDecision.SUPERSEDE),
        ("I prefer tabs.", "I prefer dark mode.", DedupDecision.CREATE),
        ("I prefer tabs over spaces.", "I prefer spaces instead of tabs.", DedupDecision.SUPERSEDE),
        ("The service costs 400 USD.", "The service costs 4000 USD.", DedupDecision.CREATE),
        ("I like verbose output.", "I don't like verbose output.", DedupDecision.CREATE),
        ("We decided to use Postgres.", "We decided to ship on Friday.", DedupDecision.CREATE),
    ],
)
async def test_consolidation_decisions(native, existing, incoming, decision) -> None:
    outcome, existing_mems = await _consolidate(native, existing, incoming)
    assert outcome.decision is decision, outcome.reason
    if decision is not DedupDecision.CREATE:
        assert outcome.target_memory_id == existing_mems[0].memory_id


async def test_consolidation_ignores_non_current_and_empty(native) -> None:
    outcome, existing = await _consolidate(native, "My timezone is CET.", "My timezone is CET.")
    assert outcome.decision is DedupDecision.REINFORCE
    from memory_service.domain.enums import TemporalStatus

    existing[0].temporal = existing[0].temporal.model_copy(
        update={"status": TemporalStatus.SUPERSEDED}
    )
    cand = (await _extract(native, "My timezone is CET."))[0]
    assert (await native.consolidate(cand, existing, CTX)).decision is DedupDecision.CREATE
    assert (await native.consolidate(cand, [], CTX)).decision is DedupDecision.CREATE


async def test_an_imperative_is_short_lived_and_a_restated_one_is_not(native) -> None:
    """Instructions stop being durable user facts, without a new memory type.

    An integrator reported that "Do not invent a sales number" and "Do not invent versions"
    were stored as PREFERENCE / visibility=USER / predicate=instruction, and so followed
    their author into every later thread at full ranking weight. They are right that those
    are turn instructions, not facts about a person.

    But "always answer in metric units" matches the same pattern and IS durable, and no
    regex separates them from one sentence. So the lever is durability, not classification:
    every imperative starts SHORT_TERM - a seven-day clock and 0.65 ranking weight instead
    of LONG_TERM's 1.0 - and restating one renews it. What gets repeated survives; what was
    scoped to one task lapses on its own.
    """
    for text in ("Do not invent a sales number.", "Always answer in metric units."):
        cand = (await _extract(native, text))[0]
        assert cand.predicate == "instruction", text
        assert cand.lifetime is Lifetime.SHORT_TERM, f"{text} must not be durable on sight"

    # a real preference is untouched: it states something about the person, not the task
    pref = (await _extract(native, "I prefer concise answers."))[0]
    assert pref.memory_type is MemoryType.PREFERENCE
    assert pref.lifetime is Lifetime.LONG_TERM, "stating a preference is not an imperative"


