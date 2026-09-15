"""Native memory intelligence: deterministic extraction, classification and consolidation.

No LLM is involved. Rules are conservative on purpose: when the service is not sure a
sentence is memory-worthy it stores nothing (the raw message is still in the thread and the
archive), and when it is not sure two memories are the same it keeps both. The release gate
for this module is the *false-merge rate*, so every merge/supersede decision needs positive
evidence (identical normalized text, identical subject+predicate, or an explicit replacement
signal), and disagreeing numbers or negation always block a merge.
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


def _user_subject(ctx: MemoryExecutionContext) -> str:
    return f"user:{ctx.user_id}" if ctx.user_id else ctx.principal_id


class NativeMemoryIntelligence:
    """Rule-based provider. ``embedding`` (optional) adds a dense-similarity dedup signal."""

    info = ProviderInfo(
        name="native", license="Apache-2.0", origin="memory-service", locality="local"
    )

    def __init__(
        self, settings: MemoryIntelligenceSettings, embedding: EmbeddingProvider | None = None
    ) -> None:
        self.cfg = settings
        self.embedding = embedding

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
        for sentence in split_sentences(text):
            for clause in split_clauses(sentence):
                cand = self._from_sentence(clause, ctx, evidence, kind=kind)
                if cand is None:
                    continue
                key = normalized_hash(cand.content)
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
        return out

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
            visibility = Visibility.PRIVATE
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
        dense: list[float] | None = None
        for mem in existing:
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
        return best or ConsolidationOutcome(
            decision=DedupDecision.CREATE, candidate=candidate, reason="no match"
        )

    async def search_features(self, query: str, ctx: MemoryExecutionContext) -> dict[str, Any]:
        return {}
