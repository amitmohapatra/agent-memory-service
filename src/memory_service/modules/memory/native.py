"""Native memory intelligence: deterministic extraction, classification and consolidation.

The rules are complete on their own and conservative on purpose: when the service is not
sure a sentence is memory-worthy it stores nothing (the raw message is still in the thread
and the archive), and when it is not sure two memories are the same it keeps both. The
release gate for this module is the *false-merge rate*, so every merge/supersede decision
needs positive evidence (identical normalized text, the same subject - the subject matcher's
SAME, ``modules/memory/subjects.py`` - and predicate, or an explicit replacement signal), and
disagreeing numbers or negation always block a merge.

An optional ``LLMAssist`` extracts facts from the sentences no rule matched
(``contextual_extraction``) and adjudicates what consolidation leaves open
(``conflict_adjudication``): a stored memory on the same, or possibly the same, subject that
no rule settled (ADR 0035). Every consultation falls back to the native result, and with
assist disabled the module behaves exactly as without it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    DedupDecision,
    Lifetime,
    MemoryType,
    ObservationKind,
    Visibility,
)
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.ids import content_hash
from memory_service.domain.language import is_english
from memory_service.domain.memory import CanonicalMemory, unverified_representation
from memory_service.domain.observation import Observation
from memory_service.domain.predicates import is_single_valued
from memory_service.domain.subjects import SubjectVerdict
from memory_service.domain.text import ACKNOWLEDGEMENT, SENTENCE_BREAK, normalise_number
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.narrative import (
    eligible_for_contextual_extraction,
    extract_narrative_units,
)
from memory_service.modules.memory.source_facts import CONFIDENCE as SOURCE_FACT_CONFIDENCE
from memory_service.modules.memory.source_facts import extract_source_facts
from memory_service.modules.memory.subjects import NO_SUBJECT, SubjectMatcher, SubjectPair
from memory_service.ports.intelligence import (
    ConsolidationOutcome,
    ContextualExtractor,
    MemoryCandidate,
)
from memory_service.ports.models import EmbeddingProvider, ProviderInfo

# --------------------------------------------------------------------------- text utils

_WORD = re.compile(r"[a-z0-9](?:[a-z0-9'+\-./@]*[a-z0-9])?")
_NUMBER = re.compile(r"\d+(?:[.,\u066b\u066c\uff0e\uff0c]\d+)*")
_STOP_WORDS = """
a an the and or but if then of to in on at by for with from as is are was were be been
being it its this that these those there here i me my we our you your they them he she
his her do does did done have has had having not no so than too very can will would
should could may might must shall into onto about please
"""
_STOP = frozenset(_STOP_WORDS.split())
_NEGATION = frozenset(
    {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "n't", "cannot"}
)
_REPLACEMENT = re.compile(
    r"\b(no longer|not anymore|any ?more|instead of|switched to|moved to|changed to|"
    r"from now on|used to|previously|now (?:i|we|it)|update:|correction:|actually,?)\b",
    re.IGNORECASE,
)
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_NAMES = """january february march april may june july august september october
november december"""
_MONTHS = {m: i + 1 for i, m in enumerate(_MONTH_NAMES.split())}
_DATE_TEXT = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2})?,?\s*(\d{4})\b",
    re.IGNORECASE,
)
_VALID_FROM = re.compile(r"\b(?:as of|since|from|starting|effective)\s+([^,.;]+)", re.IGNORECASE)
_VALID_TO = re.compile(r"\b(?:until|through|till)\s+([^,.;]+)", re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation. Stable across paraphrase-free
    duplicates ("My timezone is CET." == "my timezone is CET")."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    return t.rstrip(" .!?;:")


def normalized_hash(text: str) -> str:
    return content_hash(normalize(text))


def tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def numbers(text: str) -> set[str]:
    return {normalise_number(n) for n in _NUMBER.findall(text)}


def has_negation(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & _NEGATION) or "n't" in text.lower()


def parse_date(text: str) -> datetime | None:
    m = _DATE_ISO.search(text)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        try:
            return datetime(y, mo, d, tzinfo=UTC)
        except ValueError:
            return None
    m = _DATE_TEXT.search(text)
    if m:
        month = _MONTHS[m.group(1).lower()]
        day = int(m.group(2) or 1)
        try:
            return datetime(int(m.group(3)), month, day, tzinfo=UTC)
        except ValueError:
            return None
    q = re.search(r"\bq([1-4])\s*(?:fy)?\s*(\d{4})\b", text, re.IGNORECASE)
    if q:
        return datetime(int(q.group(2)), (int(q.group(1)) - 1) * 3 + 1, 1, tzinfo=UTC)
    return None


#: A transcript line's own header: ``[1:56 pm on 8 May, 2023] Caroline: I moved to Paris.``
#: Clients that forward a chat log write one, and it defeats every rule anchored at the
#: start of a sentence - the fact pattern's subject group, the "please always/never"
#: preference, the decision prefix, the chit-chat filter - because sentences split on .!?
#: alone, so the header stays glued to the first sentence of the turn and that sentence is
#: unparseable by construction.
#:
#: Both halves are deliberately narrow, because this runs on every observation of every
#: tenant and a false positive silently deletes the start of the text extraction sees.
#:
#: The bracket must look like a timestamp, not merely hold a digit: it has to carry a clock
#: time or a four-digit year, and it may not contain a colon followed by whitespace. That
#: second condition is what saves an appended image caption, which is bracketed the same way
#: and may itself carry a number - "[Shared an image: 2 dogs]", "[Shared an image of a 2018
#: car: a red mustang]" - and which holds the only answer some questions have.
#:
#: The speaker is a name, at most three words long, not "anything up to a colon". Unbounded,
#: it ate ordinary prose after a real header: "[8 May, 2023] I moved to Paris: it was great."
#: left only "it was great.", losing the subject, the place and the fact. A bare
#: "Decision: ship it" is a prefix this module understands, which is why the timestamp is
#: required rather than the speaker alone.
_TURN_PREFIX = re.compile(
    r"^\s*\[(?=[^\]\n]{0,60}\])(?![^\]\n]*:\s)"
    r"(?=[^\]\n]*(?:\d{1,2}:\d{2}|(?:19|20)\d{2}))[^\]\n]*\]"
    r"[ \t]*(?:[^\W\d_][\w.'\-]*(?:[ \t][^\W\d_][\w.'\-]*){0,2}:[ \t]+)?"
)


def strip_turn_prefix(text: str) -> str:
    """``text`` without its transcript header; unchanged when there is none."""
    return _TURN_PREFIX.sub("", text, count=1)


def split_sentences(text: str, *, max_sentences: int = 40) -> list[str]:
    out = []
    for raw in SENTENCE_BREAK.split(strip_turn_prefix(text)):
        s = raw.strip().strip("-•*# ").strip()
        if _has_sentence_content(s):
            out.append(s)
        if len(out) >= max_sentences:
            break
    return out


def _has_sentence_content(text: str) -> bool:
    """Keep lexical content without using English word length as a fact classifier.

    One Turkish word or three Chinese characters can express a complete durable fact.
    Unknown short text is retained as a source observation, not promoted to a typed fact.
    The existing explicit noise/question policies still govern admission below.
    """
    return any(char.isalpha() for char in text) and not ACKNOWLEDGEMENT.fullmatch(text)


_CLAUSE_SPLIT = re.compile(
    r"\s*(?:,|;)?\s+(?:and|but)\s+(?=(?:my|i|i'm|we|please|call me)\b)", re.IGNORECASE
)


def split_clauses(sentence: str) -> list[str]:
    """'My name is X and my timezone is Y' -> two first-person clauses."""
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(sentence) if p.strip()]
    return parts or [sentence]


# --------------------------------------------------------------------------- rules

_ATTRIBUTE = re.compile(
    r"\bmy\s+(name|timezone|time zone|role|title|team|email|location|birthday|manager|company|"
    r"employer|city|country|language|pronouns|phone|department|working hours|handle|username)"
    r"\s+(?:is|are|=|:)\s+(.+)",
    re.IGNORECASE,
)
#: The article is mandatory. With it optional, "I'm going to the beach" parsed as the USER
#: role "going to the beach" and every later sentence of that shape SUPERSEDED the one
#: before: 103 false supersedes over two LoCoMo conversations, each one a true fact
#: closed as outdated. "I'm a staff engineer" still matches.
_IDENTITY = re.compile(
    r"\bi(?:'m| am)\s+(?:a|an|the)\s+"
    r"([a-z][a-z\- ]{2,40}?)(?:\s+(?:at|for|in|with)\s+([A-Z][\w&.\- ]{1,40}))?[.!]?$",
    re.IGNORECASE,
)
_WORKS_AT = re.compile(r"\bi\s+work\s+(?:at|for)\s+([A-Z][\w&.\- ]{1,40}?)[.!]?$", re.IGNORECASE)
_LIVES_IN = re.compile(
    r"\bi(?:'m| am)?\s+(?:live|living|based)\s+in\s+([A-Z][\w,.\- ]{1,40}?)[.!]?$", re.IGNORECASE
)
#: "I moved to Austin" / "I've just relocated to Berlin last month": where the user lives
#: now, so it supersedes the lives_in it replaces (lives_in is single-valued). The place is
#: a capitalised name, case-sensitively, and not "a new team" or "the marketing team".
_MOVED_TO = re.compile(
    r"\bi(?:'ve| have)?\s+(?:just\s+|recently\s+|finally\s+)?(?:moved|relocated)\s+"
    r"(?:back\s+)?to\s+(?-i:(?!(?:a|an|the|my|our|his|her|their|another|new|next)\b))"
    r"((?-i:[A-Z])[\w\-]*(?:,?\s+(?-i:[A-Z])[\w\-]*){0,3})"
    r"(?:\s+(?:last|this|in|on|a|about|recently|two|three|few)\b.*)?[.!]?$",
    re.IGNORECASE,
)
_PREFERENCE = re.compile(
    r"\b(?:i|we)\s+(?:really\s+|strongly\s+|always\s+|usually\s+)?"
    r"(prefer|like|love|hate|dislike|avoid|want|need|don't like|do not like|don't want|"
    r"can't stand|never use|always use|only use)\s+(.+)",
    re.IGNORECASE,
)
#: An imperative that says it is standing: "always ...", "never ...", "don't ever ...",
#: "from now on ...". Unlike a bare imperative ("do not invent a sales number"), the
#: sentence itself says it outlives the task, so it is kept as a lasting rule.
_STANDING_RULE = re.compile(
    r"^(?:please\s+)?(?:(?:always|never)\b|(?:don't|do not)\s+ever\b|"
    r"(?:from now on|going forward|in (?:the )?future)\b[,:]?)\s*(.+)",
    re.IGNORECASE,
)
_PREF_PLEASE = re.compile(
    r"^(?:please\s+)?(always|never|don't|do not|stop|keep)\s+(.+)", re.IGNORECASE
)
_PREF_CALL_ME = re.compile(r"\b(?:call me|address me as|refer to me as)\s+(.+)", re.IGNORECASE)
_FAVOURITE = re.compile(r"\bmy\s+favou?rite\s+([a-z ]{2,30}?)\s+(?:is|are)\s+(.+)", re.IGNORECASE)
_DECISION = re.compile(
    r"\b(?:we|i|the team|the board)\s+(?:have\s+|has\s+)?(decided|agreed|chose|chosen|"
    r"will go with|are going with|settled on|approved)\s+(?:to\s+|on\s+|that\s+)?(.+)",
    re.IGNORECASE,
)
_DECISION_PREFIX = re.compile(r"^decision:\s*(.+)", re.IGNORECASE)
_PROCEDURAL = re.compile(
    r"\b(?:to\s+[a-z]+(?:\s+[a-z]+){0,4},\s*(?:run|use|execute|open|call)\s+.+|"
    r"(?:always|first|then|before|after)\s+(?:run|use|execute|check|call)\s+.+|"
    r"steps?\s*:\s*.+|(?:the\s+)?(?:procedure|process|runbook|playbook)\s+(?:is|for)\s+.+)",
    re.IGNORECASE,
)
_TASK = re.compile(
    r"\b(todo|to-do|action item|need to|needs to|have to|must|remind me to|follow up|"
    r"follow-up|by (?:monday|tuesday|wednesday|thursday|friday|eod|end of day|next week|"
    r"tomorrow))\b",
    re.IGNORECASE,
)
_FACT = re.compile(
    r"^(?P<subject>(?:[Tt]he\s+)?[A-Za-z0-9][\w&.\-]*(?:\s+[A-Za-z0-9][\w&.\-]*){0,4}?)"
    r"\s+(?P<pred>is|are|was|were|has|have|costs?|uses?|runs? on|belongs? to|owns?|reports? to|"
    r"is owned by|is located in|is due|starts?|ends?|expires?)\s+(?P<object>.+)$"
)
_EVENT_HINT = re.compile(
    r"\b(yesterday|today|this morning|last night|just now|we shipped|we deployed|deployed|"
    r"released|failed|outage|incident|rolled back|merged|launched|migrated|happened)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"[?\uff1f\u061f]\s*$|^(?:what|why|how|when|where|who|can you|could you|do you)\b", re.I
)

_LIFETIME_BY_TYPE = {
    MemoryType.PREFERENCE: Lifetime.LONG_TERM,
    MemoryType.USER: Lifetime.LONG_TERM,
    MemoryType.SEMANTIC: Lifetime.LONG_TERM,
    MemoryType.PROCEDURAL: Lifetime.LONG_TERM,
    MemoryType.EPISODIC: Lifetime.LONG_TERM,
    MemoryType.SHARED: Lifetime.LONG_TERM,
    MemoryType.WORK: Lifetime.SHORT_TERM,
    MemoryType.TASK: Lifetime.SHORT_TERM,
    MemoryType.TOOL: Lifetime.SHORT_TERM,
    MemoryType.WORKING: Lifetime.EPHEMERAL,
    MemoryType.CONVERSATION: Lifetime.SHORT_TERM,
    MemoryType.AGENT: Lifetime.SHORT_TERM,
}
_IMPORTANCE_BY_TYPE = {
    MemoryType.PREFERENCE: 0.7,
    MemoryType.USER: 0.8,
    MemoryType.SEMANTIC: 0.5,
    MemoryType.PROCEDURAL: 0.7,
    MemoryType.EPISODIC: 0.4,
    MemoryType.TASK: 0.5,
    MemoryType.TOOL: 0.3,
    MemoryType.AGENT: 0.4,
    MemoryType.WORKING: 0.2,
    MemoryType.CONVERSATION: 0.2,
    MemoryType.SHARED: 0.6,
    MemoryType.WORK: 0.5,
}


# --------------------------------------------------------------------------- llm assist

_ASSIST_MAX_CHARS = 2000
#: A pair with no subject on one side has nothing for the subject matcher to compare: it is
#: worth asking about when it shares this much of its wording, as before ADR 0035.
_ASSIST_MIN_SIMILARITY = 0.5
#: a memory the adjudicator may be asked about: the memory, its lexical similarity, and the
#: subject matcher's view of the pair
_Uncertain = tuple[CanonicalMemory, float, SubjectPair]
_ADJUDICATION_SYSTEM = (
    "Compare a NEW memory with an EXISTING one and answer with a verdict: 'same' when both "
    "state the same fact (paraphrase, no new information); 'update' when the new one gives a "
    "newer value for the same thing and should replace the existing one; 'contradict' when "
    "they disagree about the same thing and neither clearly replaces the other; 'different' "
    "when they are about different things. When unsure answer 'different'."
)
_ADJUDICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdict"],
    "properties": {
        "verdict": {"type": "string", "enum": ["same", "update", "contradict", "different"]}
    },
}


_OBJECT_TAIL = re.compile(
    r"\s*\b(?:since|as of|from|starting|effective|until|through|till|from now on)\b.*$",
    re.IGNORECASE,
)
_OBJECT_HEAD = re.compile(r"^(?:now|currently|actually|still|also|just)\s+", re.IGNORECASE)
_OBJECT_END = re.compile(r"\s+(?:now|currently|these days|anymore|any more|again)$", re.IGNORECASE)


def _clean_object(text: str) -> str:
    t = _OBJECT_HEAD.sub("", text.strip())
    t = _OBJECT_TAIL.sub("", t)
    t = _OBJECT_END.sub("", normalize(t))
    return t.strip(" ,;:")


def _evidence(observation: Observation) -> list[EvidenceRef]:
    source_type = {
        ObservationKind.MESSAGE: EvidenceSource.MESSAGE,
        ObservationKind.FILE: EvidenceSource.FILE,
        ObservationKind.AGENT_RESULT: EvidenceSource.AGENT_RESULT,
        ObservationKind.TOOL_RESULT: EvidenceSource.TOOL_RESULT,
        ObservationKind.IMPORT: EvidenceSource.IMPORT,
    }.get(observation.kind, EvidenceSource.OBSERVATION)
    return [
        EvidenceRef(
            source_type=source_type,
            source_id=observation.message_id
            or observation.document_id
            or observation.tool_run_id
            or observation.observation_id,
            message_id=observation.message_id,
            document_id=observation.document_id,
            agent_id=observation.agent_id,
            agent_run_id=observation.agent_run_id,
            tool_run_id=observation.tool_run_id,
            observed_at=observation.occurred_at,
        )
    ]


_SHARED_LEVELS = frozenset({"AGENT_GROUP", "THREAD", "TENANT"})


def _other_principals_shared(mem: CanonicalMemory, ctx: MemoryExecutionContext) -> bool:
    """True when ``mem`` was written by a different principal into a shared scope."""
    return mem.owner_principal != ctx.principal_id and mem.scope.level.value in _SHARED_LEVELS


def _user_subject(ctx: MemoryExecutionContext) -> str:
    return f"user:{ctx.user_id}" if ctx.user_id else ctx.principal_id


def default_visibility(mt: MemoryType, ctx: MemoryExecutionContext) -> Visibility:
    """Who may read a memory of this type when the writer did not say."""
    if mt in (MemoryType.USER, MemoryType.PREFERENCE):
        visibility = Visibility.USER if ctx.user_id else Visibility.PRIVATE
    elif mt in (MemoryType.AGENT, MemoryType.TOOL, MemoryType.WORKING):
        # hand-off context flows down the run tree (child runs read it); nothing else does
        visibility = Visibility.RUN if ctx.agent_run_id else Visibility.PRIVATE
    elif mt is MemoryType.SHARED:
        visibility = (
            Visibility.AGENT_GROUP
            if ctx.agent_group_id
            else Visibility.THREAD
            if ctx.thread_id
            else Visibility.TENANT
        )
    elif ctx.thread_id:
        visibility = Visibility.THREAD
    elif ctx.user_id:
        visibility = Visibility.USER
    else:
        visibility = Visibility.TENANT
    if ctx.is_agent and mt not in (MemoryType.USER, MemoryType.PREFERENCE):
        # an agent's own working notes stay private unless the type is explicitly shared
        visibility = (
            Visibility.PRIVATE if mt in (MemoryType.TASK, MemoryType.EPISODIC) else visibility
        )
    return visibility


class NativeMemoryIntelligence:
    """Rule-based provider. ``embedding`` (optional) adds a dense-similarity dedup signal."""

    info = ProviderInfo(
        name="native", license="Apache-2.0", origin="memory-service", locality="local"
    )

    def __init__(
        self,
        settings: MemoryIntelligenceSettings,
        embedding: EmbeddingProvider | None = None,
        *,
        assist: LLMAssist | None = None,
        contextual_extractor: ContextualExtractor | None = None,
        subjects: SubjectMatcher | None = None,
    ) -> None:
        self.cfg = settings
        self.embedding = embedding
        self.assist = assist or LLMAssist.disabled()
        self.contextual_extractor = contextual_extractor
        #: whether two statements share a subject: the slot rules and the adjudicator's gate
        self.subjects = subjects or SubjectMatcher(embedding)

    # -- extraction -----------------------------------------------------------------
    async def extract(
        self, observation: Observation, ctx: MemoryExecutionContext
    ) -> list[MemoryCandidate]:
        if observation.hints.skip_extraction or not observation.content.strip():
            return []
        evidence = _evidence(observation)
        text = observation.content.strip()
        kind = observation.kind
        if kind is ObservationKind.TOOL_RESULT:
            return [
                MemoryCandidate(
                    content=text[:1000],
                    memory_type=MemoryType.TOOL,
                    lifetime=Lifetime.SHORT_TERM,
                    subject=observation.tool_run_id or ctx.agent_id,
                    predicate="result",
                    evidence=evidence,
                    importance=0.3,
                    confidence=0.9,
                    category="tool_result",
                )
            ]
        if kind is ObservationKind.AGENT_RESULT:
            return [
                MemoryCandidate(
                    content=text[:2000],
                    memory_type=MemoryType.AGENT,
                    lifetime=Lifetime.SHORT_TERM,
                    subject=f"agent:{observation.agent_id}" if observation.agent_id else None,
                    predicate="result",
                    evidence=evidence,
                    importance=0.4,
                    confidence=0.8,
                    category="agent_result",
                )
            ]
        if kind is ObservationKind.DECISION:
            return [
                self._decision(text, ctx, evidence, confidence=0.95),
            ]
        if kind is ObservationKind.FEEDBACK:
            cands = [
                c for s in split_sentences(text) if (c := self._from_sentence(s, ctx, evidence))
            ]
            if cands:
                return cands
            return [
                MemoryCandidate(
                    content=text[:1000],
                    memory_type=MemoryType.PREFERENCE,
                    lifetime=Lifetime.LONG_TERM,
                    subject=_user_subject(ctx),
                    predicate="feedback",
                    evidence=evidence,
                    importance=0.6,
                    confidence=0.7,
                    category="feedback",
                )
            ]
        out: list[MemoryCandidate] = []
        seen: set[str] = set()
        sentences = split_sentences(text)
        contextual = await self._contextual_candidates(observation, ctx, evidence, sentences)
        for sentence in sentences:
            for clause in split_clauses(sentence):
                cand = self._from_sentence(clause, ctx, evidence, kind=kind)
                if cand is None:
                    continue
                key = normalized_hash(cand.content)
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
        original_key = normalized_hash(text) if self.cfg.keep_verbatim_turns else None
        for cand in contextual:
            key = normalized_hash(cand.content)
            if key in seen or key == original_key:
                continue
            seen.add(key)
            out.append(cand)
        verbatim = self._verbatim(text, observation, ctx, evidence)
        if verbatim is not None and normalized_hash(verbatim.content) not in seen:
            out.append(verbatim)
        return out

    async def _contextual_candidates(
        self,
        observation: Observation,
        ctx: MemoryExecutionContext,
        evidence: list[EvidenceRef],
        sentences: list[str],
    ) -> list[MemoryCandidate]:
        """Model-extracted facts for the sentences no rule matched (empty when not wanted).

        English: the narrative units of the sentences no rule parsed. Any other language
        (``Observation.lang``): typed facts in that language, since no rule can parse it.
        """
        if (
            observation.kind is not ObservationKind.MESSAGE
            or observation.agent_authored
            or not self.assist.wants("contextual_extraction")
        ):
            return []
        if not is_english(observation.lang):
            return await self._source_facts(ctx, evidence, sentences)
        eligible = {
            index
            for index, sentence in enumerate(sentences)
            if not _QUESTION.search(sentence)
            and not ACKNOWLEDGEMENT.match(sentence)
            and any(
                self._from_sentence(clause, ctx, evidence, kind=observation.kind) is None
                for clause in split_clauses(sentence)
            )
        }
        if not eligible_for_contextual_extraction(sentences, eligible):
            return []
        units, provider, category, confidence = await self._contextual_units(
            observation, ctx, sentences, eligible
        )
        return [
            MemoryCandidate(
                content=content,
                memory_type=MemoryType.OBSERVATION,
                lifetime=Lifetime.LONG_TERM,
                subject=_user_subject(ctx),
                predicate="said",
                evidence=evidence,
                importance=0.4,
                confidence=confidence,
                category=category,
                provider=provider,
            )
            for content in units or []
        ]

    async def _source_facts(
        self,
        ctx: MemoryExecutionContext,
        evidence: list[EvidenceRef],
        sentences: list[str],
    ) -> list[MemoryCandidate]:
        eligible = {
            index
            for index, sentence in enumerate(sentences)
            if not _QUESTION.search(sentence) and not ACKNOWLEDGEMENT.match(sentence)
        }
        facts = await extract_source_facts(self.assist, sentences, eligible)
        return [
            MemoryCandidate(
                content=fact.text,
                memory_type=fact.memory_type,
                lifetime=_LIFETIME_BY_TYPE.get(fact.memory_type, Lifetime.LONG_TERM),
                subject=_user_subject(ctx),
                predicate=fact.predicate,
                object=fact.object,
                evidence=evidence,
                importance=_IMPORTANCE_BY_TYPE.get(fact.memory_type, 0.5),
                confidence=SOURCE_FACT_CONFIDENCE,
                category="source_fact",
                provider="llm",
            )
            for fact in facts or []
        ]

    async def _contextual_units(
        self,
        observation: Observation,
        ctx: MemoryExecutionContext,
        sentences: list[str],
        eligible: set[int],
    ) -> tuple[list[str] | None, str, str, float]:
        """Select one extraction engine before making any model request."""
        extractor = self.contextual_extractor
        if extractor is None or ctx.is_agent:
            # Every agent stays on the credential-aware transport, including before its
            # first key registration. Hindsight cannot revalidate an agent policy after
            # waiting for a server extraction slot, and cannot accept its model VK.
            units = await extract_narrative_units(self.assist, sentences, eligible)
            return units, "native", "narrative_unit", 0.99
        units = await extractor.extract(
            observation.content.strip(), timestamp=observation.occurred_at
        )
        return units, extractor.name, "contextual_fact", 0.5

    def _verbatim(
        self,
        text: str,
        observation: Observation,
        ctx: MemoryExecutionContext,
        evidence: list[EvidenceRef],
    ) -> MemoryCandidate | None:
        """The turn as it was said, so that what no rule matched is still retrievable.

        The rules above are first-person: "I work at X", "my favourite is Y". A conversation
        *about* someone is third-person, matches nothing, and used to be dropped entirely —
        measured on LoCoMo, 452 of 788 turns produced no candidate at all, which put the
        answer out of reach of every retriever before ranking was ever consulted. The
        platitude was kept and the fact was lost: "Unconditional love is so important" was
        stored while "camping at the beach", in the same turn, was not.

        Deliberately an OBSERVATION. That type is in DERIVED_MEMORY_TYPES, which is
        excluded from supersession and reflection, so a verbatim turn can never be mistaken
        for an asserted fact or merged with one. Any other memory type here would quietly
        feed raw chatter into the consolidation machinery.
        """
        if not self.cfg.keep_verbatim_turns or observation.kind is not ObservationKind.MESSAGE:
            return None
        # An agent's own messages are its working chatter: classify() gives a generic
        # OBSERVATION thread/workspace visibility, which leaked "Thinking: ..." into the
        # user's memory the first time this ran without a guard
        # (tests/integration/test_multi_agent.py caught it).
        #
        # User-authored messages are retained inside threads too: archived storage is not
        # ranked retrieval. Keep this guard about authorship, not the presence of agent
        # lineage or thread_id; a harness can relay a user's own words.
        if observation.agent_authored:
            return None
        body = text.strip()
        if not body:
            return None
        # Reuse the noise rules the extractor already trusts rather than inventing a second
        # notion of "not worth keeping". A bare question or a greeting carries no fact, so
        # storing it verbatim only dilutes retrieval — which is the one risk this whole
        # change runs. Anything that is neither is kept, including the third-person
        # narrative that no extraction rule can parse.
        # Per sentence, not per turn: "Did you hear? I moved from Sweden four years ago."
        # ends in a question mark only in its first sentence. Judged whole, 199 of 788
        # LoCoMo turns were dropped and 21 of 79 wrong answers had their gold in that set.
        if not any(
            not _QUESTION.search(s) and not ACKNOWLEDGEMENT.match(s) for s in split_sentences(body)
        ):
            return None
        return MemoryCandidate(
            content=body[: self.cfg.verbatim_max_chars],
            memory_type=MemoryType.OBSERVATION,
            lifetime=Lifetime.LONG_TERM,
            subject=_user_subject(ctx),
            predicate="said",
            evidence=evidence,
            # Below every rule-extracted candidate: a parsed fact outranks the raw turn it
            # came from, so ranking prefers the answer over the transcript when both match.
            importance=0.25,
            confidence=0.99,  # nobody is guessing what was said
            category="verbatim_turn",
        )

    def _decision(
        self,
        text: str,
        ctx: MemoryExecutionContext,
        evidence: list[EvidenceRef],
        *,
        confidence: float,
        object_text: str | None = None,
    ) -> MemoryCandidate:
        subject = (
            f"work:{ctx.work_id}"
            if ctx.work_id
            else f"thread:{ctx.thread_id}"
            if ctx.thread_id
            else f"workspace:{ctx.workspace_id}"
            if ctx.workspace_id
            else _user_subject(ctx)
        )
        return MemoryCandidate(
            content=text.strip()[:2000],
            memory_type=MemoryType.SEMANTIC,
            lifetime=Lifetime.LONG_TERM,
            subject=subject,
            predicate="decided",
            object=_clean_object(object_text or text)[:500],
            evidence=evidence,
            importance=0.8,
            confidence=confidence,
            category="decision",
            valid_from=parse_date(text),
        )

    def _from_sentence(
        self,
        s: str,
        ctx: MemoryExecutionContext,
        evidence: list[EvidenceRef],
        *,
        kind: ObservationKind = ObservationKind.MESSAGE,
    ) -> MemoryCandidate | None:
        if _QUESTION.search(s) or ACKNOWLEDGEMENT.match(s):
            return None
        negates = bool(_REPLACEMENT.search(s))
        vf_m, vt_m = _VALID_FROM.search(s), _VALID_TO.search(s)
        vf = parse_date(vf_m.group(1)) if vf_m else None
        vt = parse_date(vt_m.group(1)) if vt_m else None
        user = _user_subject(ctx)
        common: dict[str, Any] = {
            "evidence": evidence,
            "negates_prior": negates,
            "valid_from": vf,
            "valid_to": vt,
        }
        if m := _DECISION_PREFIX.match(s):
            return self._decision(m.group(1), ctx, evidence, confidence=0.9)
        if m := _ATTRIBUTE.search(s):
            attr = m.group(1).lower().replace(" ", "_")
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.USER,
                lifetime=Lifetime.LONG_TERM,
                subject=user,
                predicate=attr,
                object=_clean_object(m.group(2))[:300],
                importance=0.8,
                confidence=0.9,
                category="attribute",
                **common,
            )
        if m := _FAVOURITE.search(s):
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.PREFERENCE,
                lifetime=Lifetime.LONG_TERM,
                subject=user,
                predicate=f"favourite_{m.group(1).strip().lower().replace(' ', '_')}",
                object=_clean_object(m.group(2))[:300],
                importance=0.7,
                confidence=0.85,
                category="preference",
                **common,
            )
        if m := _PREF_CALL_ME.search(s):
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.PREFERENCE,
                lifetime=Lifetime.LONG_TERM,
                subject=user,
                predicate="name",
                object=_clean_object(m.group(1))[:100],
                importance=0.8,
                confidence=0.9,
                category="preference",
                **common,
            )
        if m := _WORKS_AT.search(s):
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.USER,
                lifetime=Lifetime.LONG_TERM,
                subject=user,
                predicate="works_at",
                object=_clean_object(m.group(1)),
                importance=0.8,
                confidence=0.85,
                category="attribute",
                **common,
            )
        if m := _LIVES_IN.search(s) or _MOVED_TO.search(s):
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.USER,
                lifetime=Lifetime.LONG_TERM,
                subject=user,
                predicate="lives_in",
                object=_clean_object(m.group(1)),
                importance=0.8,
                confidence=0.85,
                category="attribute",
                **common,
            )
        if m := _PREFERENCE.search(s):
            verb = m.group(1).lower()
            predicate = (
                "dislikes"
                if verb in ("hate", "dislike", "avoid", "don't like", "do not like", "can't stand")
                else "avoids"
                if verb in ("don't want", "never use")
                else "prefers"
            )
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.PREFERENCE,
                lifetime=Lifetime.LONG_TERM,
                subject=user,
                predicate=predicate,
                object=_clean_object(m.group(2))[:300],
                importance=0.7,
                confidence=0.8,
                category="preference",
                **common,
            )
        if rule := _STANDING_RULE.match(s):
            # A standing rule ("Never suggest recipes with cilantro", "Always write Python
            # with strict type hints") says so in its own words, so it is durable on sight:
            # a lasting PREFERENCE the user profile is kept from, not a seven-day
            # instruction that lapses unless restated.
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.PREFERENCE,
                lifetime=Lifetime.LONG_TERM,
                subject=user,
                predicate="rule",
                # the whole rule ("never suggest ..."), or what follows a marker the cleaner
                # drops as a time phrase ("from now on, reply in German")
                object=(
                    _clean_object(re.sub(r"^please\s+", "", s, flags=re.IGNORECASE))
                    or _clean_object(rule.group(1))
                )[:300],
                importance=0.8,
                confidence=0.85,
                category="rule",
                **common,
            )
        if m := _PREF_PLEASE.match(s):
            return MemoryCandidate(
                content=s,
                # An imperative is not a fact about the person who said it. "Do not invent a
                # sales number" is scoped to the work in front of them; "always answer in
                # metric units" is not, and no pattern can tell the two apart from one
                # sentence. So durability decides it instead of classification: both start
                # SHORT_TERM, and the one that keeps being restated keeps renewing (see the
                # REINFORCE branch in pipeline.py) while the task-scoped one lapses.
                #
                # They were LONG_TERM and USER-scoped, so a single "do not invent versions"
                # followed its author into every later conversation at full ranking weight.
                # An integrator reported exactly that.
                memory_type=MemoryType.PREFERENCE,
                lifetime=Lifetime.SHORT_TERM,
                subject=user,
                predicate="instruction",
                object=_clean_object(f"{m.group(1)} {m.group(2)}")[:300],
                importance=0.7,
                confidence=0.75,
                category="instruction",
                **common,
            )
        if m := _DECISION.search(s):
            cand = self._decision(s, ctx, evidence, confidence=0.85, object_text=m.group(2))
            return cand.model_copy(
                update={"negates_prior": negates, "valid_from": vf or cand.valid_from}
            )
        if m := _IDENTITY.search(s):
            role = m.group(1).strip().lower()
            if role and role not in ("sure", "not", "here", "back", "done", "sorry", "fine"):
                return MemoryCandidate(
                    content=s,
                    memory_type=MemoryType.USER,
                    lifetime=Lifetime.LONG_TERM,
                    subject=user,
                    predicate="role",
                    object=_clean_object(role + (f" at {m.group(2)}" if m.group(2) else "")),
                    importance=0.8,
                    confidence=0.7,
                    category="attribute",
                    **common,
                )
        if _PROCEDURAL.search(s):
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.PROCEDURAL,
                lifetime=Lifetime.LONG_TERM,
                subject=f"workspace:{ctx.workspace_id}" if ctx.workspace_id else user,
                predicate="procedure",
                importance=0.7,
                confidence=0.7,
                category="procedure",
                **common,
            )
        if _TASK.search(s):
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.TASK,
                lifetime=Lifetime.SHORT_TERM,
                subject=f"thread:{ctx.thread_id}" if ctx.thread_id else user,
                predicate="task",
                importance=0.5,
                confidence=0.7,
                category="task",
                valid_to=vt,
                evidence=evidence,
                negates_prior=negates,
                valid_from=vf,
            )
        if m := _FACT.match(s):
            subject = m.group("subject").strip()
            # First person is the speaker; any other pronoun has no referent here and is
            # left to the verbatim copy of the turn rather than stored as a fact about "it".
            if subject.lower() in ("i", "we", "my", "our"):
                subject = _user_subject(ctx)
            elif subject.lower() in ("it", "this", "that", "there", "they", "you", "he", "she"):
                return None
            if len(s) <= 300 and (
                subject[0].isupper() or bool(numbers(s)) or subject.startswith("user:")
            ):
                return MemoryCandidate(
                    content=s,
                    memory_type=MemoryType.SEMANTIC,
                    lifetime=Lifetime.LONG_TERM,
                    subject=re.sub(r"^the\s+", "", subject.lower()),
                    predicate=m.group("pred").lower().replace(" ", "_"),
                    object=_clean_object(m.group("object"))[:300],
                    importance=0.5,
                    confidence=0.6,
                    category="fact",
                    entities=[subject],
                    **common,
                )
        if kind in (ObservationKind.EVENT, ObservationKind.IMPORT) or _EVENT_HINT.search(s):
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.EPISODIC,
                lifetime=Lifetime.LONG_TERM,
                subject=f"thread:{ctx.thread_id}" if ctx.thread_id else user,
                predicate="event",
                importance=0.4,
                confidence=0.6,
                category="event",
                **common,
            )
        return None

    # -- classification -------------------------------------------------------------
    async def classify(
        self, candidate: MemoryCandidate, ctx: MemoryExecutionContext
    ) -> MemoryCandidate:
        mt = candidate.memory_type
        lifetime = _LIFETIME_BY_TYPE.get(mt, candidate.lifetime)
        if candidate.predicate == "instruction":
            # An imperative is a PREFERENCE by shape, and PREFERENCE is durable by type, so
            # the table above would hand "do not invent a sales number" the same permanence
            # as "I prefer metric units". It is the one case where the sentence knows better
            # than the type: see the _PREF_PLEASE branch for why durability rather than
            # classification is the lever, and the REINFORCE branch in pipeline.py for how a
            # genuinely standing instruction earns its keep by being restated.
            lifetime = Lifetime.SHORT_TERM
        importance = _IMPORTANCE_BY_TYPE.get(mt, candidate.importance)
        visibility = default_visibility(mt, ctx)
        return candidate.model_copy(
            update={
                "lifetime": lifetime,
                "importance": importance if candidate.importance == 0.5 else candidate.importance,
                "visibility": candidate.visibility or visibility,
            }
        )

    # -- consolidation --------------------------------------------------------------
    async def consolidate(
        self,
        candidate: MemoryCandidate,
        existing: Sequence[CanonicalMemory],
        ctx: MemoryExecutionContext,
    ) -> ConsolidationOutcome:
        if not existing:
            return ConsolidationOutcome(
                decision=DedupDecision.CREATE, candidate=candidate, reason="no candidates"
            )
        c_hash = normalized_hash(candidate.content)
        c_tokens = tokens(candidate.content)
        c_numbers = numbers(candidate.content)
        c_neg = has_negation(candidate.content)
        c_obj = _clean_object(candidate.object or "")
        generated = unverified_representation(
            {"category": candidate.category, "provider": candidate.provider}
        )
        best: ConsolidationOutcome | None = None
        adjudicating = self.assist.wants("conflict_adjudication")
        #: memories the conflict adjudicator may be asked about: neither a duplicate nor a
        #: slot rule settled them and no hard block (numbers, negation) stands in the way.
        #: Which one is asked is the subject matcher's call, after the loop.
        askable: list[tuple[CanonicalMemory, float]] = []
        #: Memories whose lexical similarity puts them in the band where a dense comparison
        #: decides. They are collected rather than embedded here: each one used to cost its
        #: own single-text round trip through the encoder's one-caller gate - up to
        #: ``dedup_candidate_k`` (20) per candidate, and the pipeline walks candidates
        #: sequentially too - so the most expensive component in the service was being driven
        #: at batch size one. The port has exposed ``embed_documents`` all along.
        dense_band: list[CanonicalMemory] = []
        # a principal's own memories are matched first: "actually, X is now Y" corrects the
        # writer's own earlier finding before it is compared with anyone else's
        ordered = sorted(existing, key=lambda m: m.owner_principal != ctx.principal_id)
        pairs = self.subjects.pairs(candidate, ordered, statements=adjudicating)
        for mem in ordered:
            if (
                mem.temporal.status.value != "CURRENT"
                or mem.deleted_at is not None
                or mem.system_metadata.get("source_revisions")
            ):
                continue
            if generated != unverified_representation(mem.system_metadata):
                # Never let a generated rewrite reinforce or replace an asserted source,
                # even when its text happens to be identical.
                continue
            # 1. identical normalized content -> reinforce
            if mem.normalized_hash == c_hash:
                return ConsolidationOutcome(
                    decision=DedupDecision.REINFORCE,
                    candidate=candidate,
                    target_memory_id=mem.memory_id,
                    score=1.0,
                    reason="identical normalized content",
                )
            # Source transcripts may be reinforced only by identical wording. Fuzzy
            # similarity is not equivalence: a small foreign-language negation or changed
            # relationship can otherwise replace the only surviving source statement.
            if (
                generated
                or candidate.category in {"verbatim_turn", "narrative_unit"}
                or (mem.system_metadata.get("category") in {"verbatim_turn", "narrative_unit"})
            ):
                continue
            # the same subject however it is written ("Forklift 4", "forklift #4"), never
            # across identifiers ("Forklift #3"): the subject matcher's SAME
            same_slot = (
                candidate.predicate
                and mem.predicate == candidate.predicate
                and pairs[mem.memory_id].subject.verdict is SubjectVerdict.SAME
            )
            if same_slot:
                m_obj = _clean_object(mem.object or "")
                if m_obj and c_obj and m_obj == c_obj:
                    return ConsolidationOutcome(
                        decision=DedupDecision.REINFORCE,
                        candidate=candidate,
                        target_memory_id=mem.memory_id,
                        score=0.98,
                        reason="same subject/predicate/object",
                    )
                if m_obj and c_obj and m_obj != c_obj and is_single_valued(candidate.predicate):
                    if _other_principals_shared(mem, ctx) and not candidate.negates_prior:
                        # another agent's finding in a shared scope is not silently replaced:
                        # both stay, linked as contradicting, for a human or a later signal
                        native = ConsolidationOutcome(
                            decision=DedupDecision.CONTRADICT,
                            candidate=candidate,
                            target_memory_id=mem.memory_id,
                            score=0.9,
                            reason=f"conflicting value for {candidate.predicate} from "
                            f"{ctx.principal_id} vs {mem.owner_principal}",
                        )
                        return await self._adjudicate(
                            candidate,
                            mem,
                            ctx,
                            native=native,
                            same=native.model_copy(
                                update={
                                    "decision": DedupDecision.REINFORCE,
                                    "reason": f"equivalent value for {candidate.predicate}",
                                }
                            ),
                        )
                    # a newer value for a single-valued slot replaces the older one
                    return ConsolidationOutcome(
                        decision=DedupDecision.SUPERSEDE,
                        candidate=candidate,
                        target_memory_id=mem.memory_id,
                        score=0.95,
                        reason=f"new value for single-valued slot {candidate.predicate}",
                    )
                overlap = jaccard(c_tokens, tokens(mem.content))
                if candidate.negates_prior and overlap >= 0.3:
                    # explicit replacement ("instead of", "no longer", "actually") on the same
                    # slot and about the same thing
                    return ConsolidationOutcome(
                        decision=DedupDecision.SUPERSEDE,
                        candidate=candidate,
                        target_memory_id=mem.memory_id,
                        score=0.8 + overlap / 5,
                        reason=f"replacement signal on slot {candidate.predicate} "
                        f"(overlap {overlap:.2f})",
                    )
            # 2. near-identical wording -> merge/reinforce, but never across differing
            #    numbers or negation (those are the classic false merges)
            m_tokens = tokens(mem.content)
            sim = jaccard(c_tokens, m_tokens)
            # "Dock 3 door 4" and "Dock 4 door 3", "Tower A" and "Tower": however alike the
            # sentences (the words above ignore one-letter codes and digit order), two subjects
            # not known to be one are never merged by their wording - the adjudicator may
            # still be asked about them below
            apart = pairs[mem.memory_id].apart
            if sim >= self.cfg.dedup_lexical_threshold and not apart:
                if numbers(mem.content) != c_numbers or has_negation(mem.content) != c_neg:
                    continue
                decision = DedupDecision.MERGE if c_tokens - m_tokens else DedupDecision.REINFORCE
                outcome = ConsolidationOutcome(
                    decision=decision,
                    candidate=candidate,
                    target_memory_id=mem.memory_id,
                    score=sim,
                    reason=f"lexical similarity {sim:.2f}",
                )
                if best is None or outcome.score > best.score:
                    best = outcome
                continue
            if (
                adjudicating
                and numbers(mem.content) == c_numbers
                and has_negation(mem.content) == c_neg
            ):
                askable.append((mem, sim))
            # 3. dense similarity (only with a real embedding provider; the hash stand-in is
            #    excluded because it would merge unrelated sentences sharing a few tokens)
            if (
                self.embedding is not None
                and not self.embedding.fingerprint().startswith("hash-")
                and sim >= 0.5
                and not apart
            ):
                dense_band.append(mem)
        # 3. dense similarity, in one pass (only with a real embedding provider; the hash
        #    stand-in is excluded because it would merge unrelated sentences sharing a few
        #    tokens). Deferred to here so an early return above - a single-valued slot being
        #    superseded - never pays for an embedding, and so the whole band is one batch.
        if dense_band and self.embedding is not None:
            vectors = await self.embedding.embed_documents(
                [candidate.content, *(mem.content for mem in dense_band)]
            )
            dense, others = vectors[0], vectors[1:]
            for mem, other in zip(dense_band, others, strict=True):
                cos = sum(a * b for a, b in zip(dense, other, strict=True))
                if cos >= self.cfg.dedup_dense_threshold and (
                    numbers(mem.content) == c_numbers and has_negation(mem.content) == c_neg
                ):
                    outcome = ConsolidationOutcome(
                        decision=DedupDecision.MERGE,
                        candidate=candidate,
                        target_memory_id=mem.memory_id,
                        score=cos,
                        reason=f"dense similarity {cos:.2f}",
                    )
                    if best is None or outcome.score > best.score:
                        best = outcome
        create = ConsolidationOutcome(
            decision=DedupDecision.CREATE, candidate=candidate, reason="no match"
        )
        grey = await self._uncertain(candidate, pairs, askable) if best is None else None
        if grey is not None:
            # about the same subject but not a duplicate: the native answer is "keep both";
            # the model may recognise a paraphrase, an update or a contradiction
            mem, sim, pair = grey
            same = ConsolidationOutcome(
                decision=DedupDecision.MERGE
                if c_tokens - tokens(mem.content)
                else DedupDecision.REINFORCE,
                candidate=candidate,
                target_memory_id=mem.memory_id,
                score=sim,
                reason=f"subject {pair.statement.verdict.value}: {pair.statement.reason}; "
                f"lexical similarity {sim:.2f}",
            )
            return await self._adjudicate(candidate, mem, ctx, native=create, same=same)
        return best or create

    async def _uncertain(
        self,
        candidate: MemoryCandidate,
        pairs: dict[str, SubjectPair],
        askable: list[tuple[CanonicalMemory, float]],
    ) -> _Uncertain | None:
        """The one memory worth asking the adjudicator about: the closest of ``askable``
        whose statement the subject matcher does not call DIFFERENT (SAME before POSSIBLE,
        then the matcher's score, then the wording). When the words leave every pair
        undecided, the encoder is asked too - only here, so a write without the model never
        pays for it."""
        if not askable:
            return None

        def closest(by: dict[str, SubjectPair]) -> _Uncertain | None:
            ranked = [
                (mem, sim, by[mem.memory_id])
                for mem, sim in askable
                if by[mem.memory_id].statement.verdict is not SubjectVerdict.DIFFERENT
                or (
                    by[mem.memory_id].statement.reason == NO_SUBJECT
                    and sim >= _ASSIST_MIN_SIMILARITY
                )
            ]
            return max(
                ranked,
                key=lambda r: (
                    r[2].statement.verdict is SubjectVerdict.SAME,
                    r[2].statement.score,
                    r[1],
                ),
                default=None,
            )

        found = closest(pairs)
        if found is None:
            # only the memories that could be asked about are encoded
            asked = [mem for mem, _ in askable]
            found = closest(await self.subjects.with_vectors(candidate, asked, pairs))
        return found

    async def _adjudicate(
        self,
        candidate: MemoryCandidate,
        mem: CanonicalMemory,
        ctx: MemoryExecutionContext,
        *,
        native: ConsolidationOutcome,
        same: ConsolidationOutcome,
    ) -> ConsolidationOutcome:
        """Ask the model to settle a candidate against one existing memory. The hard blocks
        (differing numbers or negation) are checked first and are never overridden; any
        failure or a 'different' verdict keeps the native outcome."""
        if not self.assist.wants("conflict_adjudication"):
            return native
        if numbers(mem.content) != numbers(candidate.content) or has_negation(
            mem.content
        ) != has_negation(candidate.content):
            return native
        observed = candidate.evidence[0].observed_at if candidate.evidence else datetime.now(UTC)
        out = await self.assist.structured(
            "conflict_adjudication",
            system=_ADJUDICATION_SYSTEM,
            user=(
                f"NEW (by {ctx.principal_id}, observed {observed.isoformat()}):\n"
                f"{candidate.content[:_ASSIST_MAX_CHARS]}\n"
                f"triple: {candidate.subject or '-'} / {candidate.predicate or '-'} / "
                f"{candidate.object or '-'}\n\n"
                f"EXISTING (by {mem.owner_principal}, observed "
                f"{mem.temporal.observed_at.isoformat()}):\n{mem.content[:_ASSIST_MAX_CHARS]}\n"
                f"triple: {mem.subject or '-'} / {mem.predicate or '-'} / {mem.object or '-'}"
            ),
            schema=_ADJUDICATION_SCHEMA,
            max_tokens=50,
        )
        verdict = out.get("verdict") if out else None
        if verdict == "same":
            return same.model_copy(update={"reason": f"model: same ({same.reason})"})
        if verdict in ("update", "contradict"):
            return ConsolidationOutcome(
                decision=DedupDecision.SUPERSEDE
                if verdict == "update"
                else DedupDecision.CONTRADICT,
                candidate=candidate,
                target_memory_id=mem.memory_id,
                score=0.85,
                reason=f"model: {verdict} ({native.reason})",
            )
        return native
