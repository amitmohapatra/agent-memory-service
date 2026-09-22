"""Native memory intelligence: deterministic extraction, classification and consolidation.

The rules are complete on their own and conservative on purpose: when the service is not
sure a sentence is memory-worthy it stores nothing (the raw message is still in the thread
and the archive), and when it is not sure two memories are the same it keeps both. The
release gate for this module is the *false-merge rate*, so every merge/supersede decision
needs positive evidence (identical normalized text, identical subject+predicate, or an
explicit replacement signal), and disagreeing numbers or negation always block a merge.

An optional ``LLMAssist`` refines only the ambiguous decisions (low-confidence extractions,
sentences no rule matched, the grey band of consolidation) and every consultation falls back
to the native result; with assist disabled the module behaves exactly as without it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from memory_service.config.settings import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    DedupDecision,
    Lifetime,
    MemoryType,
    ObservationKind,
    Visibility,
)
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import content_hash
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.observation import Observation
from memory_service.modules.llm.assist import LLMAssist
from memory_service.ports.intelligence import ConsolidationOutcome, MemoryCandidate
from memory_service.ports.models import EmbeddingProvider, ProviderInfo

# --------------------------------------------------------------------------- text utils

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9](?:[a-z0-9'+\-./@]*[a-z0-9])?")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
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
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


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


def split_sentences(text: str, *, max_sentences: int = 40) -> list[str]:
    out = []
    for raw in _SENTENCE_SPLIT.split(text):
        s = raw.strip().strip("-•*# ").strip()
        if len(s) >= 8 and len(_WORD.findall(s.lower())) >= 3:
            out.append(s)
        if len(out) >= max_sentences:
            break
    return out


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
_IDENTITY = re.compile(
    r"\bi(?:'m| am)\s+(?:a|an|the)?\s*"
    r"([a-z][a-z\- ]{2,40}?)(?:\s+(?:at|for|in|with)\s+([A-Z][\w&.\- ]{1,40}))?[.!]?$",
    re.IGNORECASE,
)
_WORKS_AT = re.compile(r"\bi\s+work\s+(?:at|for)\s+([A-Z][\w&.\- ]{1,40}?)[.!]?$", re.IGNORECASE)
_LIVES_IN = re.compile(
    r"\bi(?:'m| am)?\s+(?:live|living|based)\s+in\s+([A-Z][\w,.\- ]{1,40}?)[.!]?$", re.IGNORECASE
)
_PREFERENCE = re.compile(
    r"\b(?:i|we)\s+(?:really\s+|strongly\s+|always\s+|usually\s+)?"
    r"(prefer|like|love|hate|dislike|avoid|want|need|don't like|do not like|don't want|"
    r"can't stand|never use|always use|only use)\s+(.+)",
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
_QUESTION = re.compile(r"\?\s*$|^(?:what|why|how|when|where|who|can you|could you|do you)\b", re.I)
_CHITCHAT = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|ok|okay|sure|great|cool|yes|no|got it|sounds good)\b[.!]?$",
    re.IGNORECASE,
)

_SINGLE_VALUED_SLOTS = """
name timezone time_zone role title team email location birthday manager company employer
city country language pronouns phone department working_hours handle username works_at
lives_in favourite
"""
_SINGLE_VALUED = frozenset(_SINGLE_VALUED_SLOTS.split())

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

_AMBIGUOUS_CONFIDENCE = 0.7
_ASSIST_MAX_SENTENCES = 8  # model consultations per observation and use
_ASSIST_MAX_CHARS = 2000
_ASSIST_MIN_SIMILARITY = 0.5
_MEMORY_TYPES = [t.value for t in MemoryType if t is not MemoryType.CUSTOM]

_EXTRACTION_SYSTEM = (
    "You refine a tentative memory extracted from one sentence by rules. Keep the meaning, "
    "write the content as one compact factual statement, and pick the memory_type: USER "
    "(stable attribute of the user), PREFERENCE (how the user wants things), SEMANTIC (a fact "
    "or decision), PROCEDURAL (how to do something), EPISODIC (something that happened), TASK "
    "(something to do). subject/predicate/object describe the fact as a triple when it has "
    "one. Never invent details that are not in the sentence."
)
_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["memory_type", "content"],
    "properties": {
        "memory_type": {"type": "string", "enum": _MEMORY_TYPES},
        "content": {"type": "string"},
        "subject": {"type": "string"},
        "predicate": {"type": "string"},
        "object": {"type": "string"},
    },
}
_WORTHINESS_SYSTEM = (
    "Decide whether one sentence from a conversation is worth remembering as a durable memory "
    "about the user, the agent or their work: a stable attribute, a preference, a decision, "
    "a fact about their project, how something is done, or a notable event. Small talk, "
    "questions, transient chatter and generic statements are not worthy. When worthy, give "
    "the memory_type (USER, PREFERENCE, SEMANTIC, PROCEDURAL, EPISODIC or TASK) and the "
    "content as one compact statement in the third person that keeps every detail."
)
_WORTHINESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["worthy"],
    "properties": {
        "worthy": {"type": "boolean"},
        "memory_type": {"type": "string", "enum": _MEMORY_TYPES},
        "content": {"type": "string"},
    },
}
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


def _bounded(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit] if text else None


def _memory_type(value: object) -> MemoryType | None:
    return MemoryType(value) if isinstance(value, str) and value in _MEMORY_TYPES else None


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
        ObservationKind.MESSAGE: "message",
        ObservationKind.FILE: "file",
        ObservationKind.AGENT_RESULT: "agent_result",
        ObservationKind.TOOL_RESULT: "tool_result",
        ObservationKind.IMPORT: "import",
    }.get(observation.kind, "observation")
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


_SHARED_LEVELS = frozenset({"AGENT_GROUP", "THREAD", "WORK", "WORKSPACE", "TENANT", "GROUP"})


def _other_principals_shared(mem: CanonicalMemory, ctx: MemoryExecutionContext) -> bool:
    """True when ``mem`` was written by a different principal into a shared scope."""
    return mem.owner_principal != ctx.principal_id and mem.scope.level.value in _SHARED_LEVELS


def _user_subject(ctx: MemoryExecutionContext) -> str:
    return f"user:{ctx.user_id}" if ctx.user_id else ctx.principal_id


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
    ) -> None:
        self.cfg = settings
        self.embedding = embedding
        self.assist = assist or LLMAssist.disabled()

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
        refine_left = worth_left = _ASSIST_MAX_SENTENCES
        for sentence in split_sentences(text):
            for clause in split_clauses(sentence):
                cand = self._from_sentence(clause, ctx, evidence, kind=kind)
                if cand is None:
                    if worth_left > 0 and self._worth_asking(clause):
                        worth_left -= 1
                        cand = await self._assist_worthiness(clause, ctx, evidence)
                elif (
                    cand.confidence <= _AMBIGUOUS_CONFIDENCE
                    and refine_left > 0
                    and self.assist.wants("ambiguous_extraction")
                ):
                    refine_left -= 1
                    cand = await self._assist_extraction(clause, cand, ctx)
                if cand is None:
                    continue
                key = normalized_hash(cand.content)
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
        verbatim = self._verbatim(text, ctx, evidence, kind)
        if verbatim is not None and normalized_hash(verbatim.content) not in seen:
            out.append(verbatim)
        return out

    def _verbatim(
        self,
        text: str,
        ctx: MemoryExecutionContext,
        evidence: list[EvidenceRef],
        kind: ObservationKind,
    ) -> MemoryCandidate | None:
        """The turn as it was said, so that what no rule matched is still retrievable.

        The rules above are first-person: "I work at X", "my favourite is Y". A conversation
        *about* someone is third-person, matches nothing, and used to be dropped entirely —
        measured on LoCoMo, 452 of 788 turns produced no candidate at all, which put the
        answer out of reach of every retriever before ranking was ever consulted. The
        platitude was kept and the fact was lost: "Unconditional love is so important" was
        stored while "camping at the beach", in the same turn, was not.

        Deliberately an OBSERVATION. That type is in DERIVED_MEMORY_TYPES, which landing.py
        excludes from supersession and reflection, so a verbatim turn can never be mistaken
        for an asserted fact or merged with one. Any other memory type here would quietly
        feed raw chatter into the consolidation machinery.
        """
        if not self.cfg.keep_verbatim_turns or kind is not ObservationKind.MESSAGE:
            return None
        body = text.strip()
        if not body:
            return None
        # Reuse the noise rules the extractor already trusts rather than inventing a second
        # notion of "not worth keeping". A bare question or a greeting carries no fact, so
        # storing it verbatim only dilutes retrieval — which is the one risk this whole
        # change runs. Anything that is neither is kept, including the third-person
        # narrative that no extraction rule can parse.
        if _QUESTION.search(body) or _CHITCHAT.match(body):
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

    def _worth_asking(self, s: str) -> bool:
        return (
            self.assist.wants("ambiguous_worthiness")
            and not _QUESTION.search(s)
            and not _CHITCHAT.match(s)
        )

    async def _assist_extraction(
        self, s: str, cand: MemoryCandidate, ctx: MemoryExecutionContext
    ) -> MemoryCandidate:
        """Let the model refine a weak rule match; the native candidate stays unless the
        answer validates. Evidence and temporal fields are never taken from the model."""
        out = await self.assist.structured(
            "ambiguous_extraction",
            system=_EXTRACTION_SYSTEM,
            user=(
                f"Sentence: {s[:_ASSIST_MAX_CHARS]}\n"
                f"Speaker: {ctx.principal_id}\n"
                f"Rule guess: memory_type={cand.memory_type.value} subject={cand.subject or '-'} "
                f"predicate={cand.predicate or '-'} object={cand.object or '-'}"
            ),
            schema=_EXTRACTION_SCHEMA,
            max_tokens=1024,
        )
        if out is None:
            return cand
        mt = _memory_type(out.get("memory_type"))
        content = _bounded(out.get("content"), 2000)
        if mt is None or content is None:
            return cand
        update: dict[str, Any] = {"content": content, "memory_type": mt}
        if mt is not cand.memory_type:
            update["lifetime"] = _LIFETIME_BY_TYPE.get(mt, cand.lifetime)
            update["importance"] = _IMPORTANCE_BY_TYPE.get(mt, cand.importance)
            update["category"] = mt.value.lower()
        if subject := _bounded(out.get("subject"), 300):
            update["subject"] = subject.lower()
        if predicate := _bounded(out.get("predicate"), 100):
            update["predicate"] = predicate.lower().replace(" ", "_")
        if obj := _bounded(out.get("object"), 300):
            update["object"] = _clean_object(obj)
        return cand.model_copy(update=update)

    async def _assist_worthiness(
        self, s: str, ctx: MemoryExecutionContext, evidence: list[EvidenceRef]
    ) -> MemoryCandidate | None:
        """No rule matched: the fast model may still find a durable memory in the sentence.
        Anything short of a validated 'worthy' answer keeps the native verdict (drop)."""
        out = await self.assist.structured(
            "ambiguous_worthiness",
            system=_WORTHINESS_SYSTEM,
            user=f"Sentence: {s[:_ASSIST_MAX_CHARS]}\nSpeaker: {ctx.principal_id}",
            schema=_WORTHINESS_SCHEMA,
            # Sized for a model that reasons before it writes: at 200, deepseek-flash
            # spent the whole budget thinking and 12% of calls came back empty (measured
            # on LoCoMo ingest). The answer itself is still one short JSON object.
            max_tokens=1024,
        )
        if out is None or out.get("worthy") is not True:
            return None
        mt = _memory_type(out.get("memory_type"))
        content = _bounded(out.get("content"), 1000)
        if mt is None or content is None:
            return None
        vf_m, vt_m = _VALID_FROM.search(s), _VALID_TO.search(s)
        user = _user_subject(ctx)
        return MemoryCandidate(
            content=content,
            memory_type=mt,
            lifetime=_LIFETIME_BY_TYPE.get(mt, Lifetime.LONG_TERM),
            subject=user
            if mt in (MemoryType.USER, MemoryType.PREFERENCE) or not ctx.thread_id
            else f"thread:{ctx.thread_id}",
            evidence=evidence,
            importance=_IMPORTANCE_BY_TYPE.get(mt, 0.5),
            confidence=0.6,
            category="assisted",
            negates_prior=bool(_REPLACEMENT.search(s)),
            valid_from=parse_date(vf_m.group(1)) if vf_m else None,
            valid_to=parse_date(vt_m.group(1)) if vt_m else None,
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
        if _QUESTION.search(s) or _CHITCHAT.match(s):
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
        if m := _LIVES_IN.search(s):
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
        if m := _PREF_PLEASE.match(s):
            return MemoryCandidate(
                content=s,
                memory_type=MemoryType.PREFERENCE,
                lifetime=Lifetime.LONG_TERM,
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
            if len(s) <= 300 and (subject[0].isupper() or bool(numbers(s))):
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
        importance = _IMPORTANCE_BY_TYPE.get(mt, candidate.importance)
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
                else Visibility.WORKSPACE
                if ctx.workspace_id
                else Visibility.TENANT
            )
        elif mt is MemoryType.WORK and ctx.work_id:
            visibility = Visibility.WORK
        elif ctx.thread_id:
            visibility = Visibility.THREAD
        elif ctx.workspace_id:
            visibility = Visibility.WORKSPACE
        elif ctx.user_id:
            visibility = Visibility.USER
        else:
            visibility = Visibility.TENANT
        if ctx.is_agent and mt not in (MemoryType.USER, MemoryType.PREFERENCE):
            # an agent's own working notes stay private unless the type is explicitly shared
            visibility = (
                Visibility.PRIVATE if mt in (MemoryType.TASK, MemoryType.EPISODIC) else visibility
            )
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
        best: ConsolidationOutcome | None = None
        grey: tuple[CanonicalMemory, float] | None = None
        dense: list[float] | None = None
        # a principal's own memories are matched first: "actually, X is now Y" corrects the
        # writer's own earlier finding before it is compared with anyone else's
        ordered = sorted(existing, key=lambda m: m.owner_principal != ctx.principal_id)
        for mem in ordered:
            if mem.temporal.status.value != "CURRENT" or mem.deleted_at is not None:
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
            same_slot = (
                candidate.subject
                and candidate.predicate
                and mem.subject == candidate.subject
                and mem.predicate == candidate.predicate
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
                pred = candidate.predicate or ""
                single = pred in _SINGLE_VALUED or pred.startswith("favourite_")
                if m_obj and c_obj and m_obj != c_obj and single:
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
            if sim >= self.cfg.dedup_lexical_threshold:
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
                sim >= _ASSIST_MIN_SIMILARITY
                and (grey is None or sim > grey[1])
                and self.assist.wants("conflict_adjudication")
                and numbers(mem.content) == c_numbers
                and has_negation(mem.content) == c_neg
            ):
                grey = (mem, sim)
            # 3. dense similarity (only with a real embedding provider; the hash stand-in is
            #    excluded because it would merge unrelated sentences sharing a few tokens)
            if (
                self.embedding is not None
                and not self.embedding.fingerprint().startswith("hash-")
                and sim >= 0.5
            ):
                if dense is None:
                    dense = await self.embedding.embed_query(candidate.content)
                other = await self.embedding.embed_query(mem.content)
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
        if best is None and grey is not None:
            # similar wording but below the merge threshold: the native answer is "keep both";
            # the model may recognise a paraphrase, an update or a contradiction
            mem, sim = grey
            same = ConsolidationOutcome(
                decision=DedupDecision.MERGE
                if c_tokens - tokens(mem.content)
                else DedupDecision.REINFORCE,
                candidate=candidate,
                target_memory_id=mem.memory_id,
                score=sim,
                reason=f"lexical similarity {sim:.2f}",
            )
            return await self._adjudicate(candidate, mem, ctx, native=create, same=same)
        return best or create

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

    async def search_features(self, query: str, ctx: MemoryExecutionContext) -> dict[str, Any]:
        return {}
