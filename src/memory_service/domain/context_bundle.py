"""ContextBundle: what an application receives to answer the current turn.

Built by the thin ContextBuilder from scope-filtered retrieval results. Follows the
context-engineering discipline: write / select / compress / isolate. It never contains
everything; it contains bounded, ranked, provenance-carrying evidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.grounding import GroundingReport

#: What produced ``ContextItem.score``; the scales are not comparable across kinds.
ScoreKind = Literal["cross_encoder", "fusion", "exact"]


class ContextItem(BaseModel):
    """One ranked piece of context."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    representation: Representation
    text: str
    #: The raw number the ranking stage produced. Kept for debugging; its meaning depends on
    #: ``score_kind``, so it is *not* comparable between items. Use ``relevance``.
    score: float = 0.0
    #: Comparable across every item in the bundle, 0..1, higher is better. This is the field
    #: to threshold and to show a user.
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Where ``score`` came from: a cross-encoder probability, a fusion rank score, or an
    #: exact identifier hit.
    score_kind: ScoreKind = "fusion"
    retrievers: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    citation: str = Field(..., description="stable citation key")
    document_id: str | None = None
    page: int | None = None
    section_path: str | None = Field(default=None, description="e.g. 'Financial Results > EBITDA'")
    expanded_from: str | None = Field(default=None, description="item_id this was expanded from")
    expansion_edge: str | None = Field(default=None, description="PARENT | NEXT | DEFINED_BY | ...")
    token_estimate: int = 0
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="structured extras: predicate/subject/object for facts and memories, "
        "fact attributes (period, currency, amount...), contradicts/contributors",
    )


class ConversationWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    thread_id: str | None = None
    message_ids: list[str] = Field(default_factory=list)
    rendered: str = ""
    token_estimate: int = 0
    summary: str | None = Field(default=None, description="rolling summary of older messages")


class UnusedEvidence(BaseModel):
    """Retrieved but not packed (reranked out or over budget): the grounding cascade scans
    these for contradictions with the answer."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    kind: str
    text: str


class EvidenceReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: EvidenceStatus
    required_groups: list[str] = Field(default_factory=list)
    satisfied_groups: list[str] = Field(default_factory=list)
    missing_groups: list[str] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list, description="strategies attempted")
    notes: list[str] = Field(default_factory=list)
    unused: list[UnusedEvidence] = Field(
        default_factory=list, description="retrieved-but-unused evidence (bounded)"
    )
    grounding: GroundingReport | None = Field(
        default=None, description="per-claim verdicts when an answer was verified"
    )
    llm_tokens: int = Field(default=0, description="LLM tokens spent building this report")


#: How many of the highest-ranked memories are repeated above the chronological timeline.
#:
#: Attention over a long context is measurably U-shaped: with the gold document first,
#: in the middle, or last, accuracy is 75.8 / 53.8 / 63.2 per cent (arXiv 2307.03172).
#: A strictly chronological list of a hundred memories drops the best-ranked evidence
#: wherever its date happens to fall, which for most questions is the trough. Ten is the
#: largest block that stays inside the first screen of the prompt, and the timeline keeps
#: its full shape because the ten are replaced there by a one-line pointer rather than
#: deleted: the ranking is added at the cost of ten short lines, not of a second copy.
#: How many memories go above the timeline, where the model reads first.
#:
#: Measured on the full 1986-question LoCoMo set: of the gold evidence that any single memory
#: carries, 69.6% sits at rank 0-9, 12.3% at 10-19, 7.2% at 20-29, 10.9% at 30-49 and NOTHING
#: beyond 50 - ``final_k`` is the ceiling. So a head of 10 leaves 30.4% of the evidence that
#: WAS retrieved below the fold, and widening to the retrieval ceiling puts all of it above.
#:
#: 30 and not 50, for a structural reason: ``render`` only emits the block when it is SMALLER
#: than the bundle (otherwise it would print the whole timeline twice), and ``final_k`` is 50.
#: A head of 50 therefore DELETES the ranked block and leaves a chronological list - which is
#: the arrangement measured to make the model anchor on position and fail date arithmetic. 30
#: keeps the block and still lifts 89.1% of the findable evidence above the fold (69.6 + 12.3
#: + 7.2), against 69.6% at ten.
#:
#: The alternative was reranking, and it was measured and rejected: ettin-17m at k=10 over the
#: same 1986 questions moved evidence_in_head 0.5968 -> 0.5940 and p99 351 -> 2018 ms. Widening
#: costs tokens, not milliseconds, and we render ~5,300 against Hindsight's ~36,000.
MOST_RELEVANT_MAX = 30

#: Share of the bundle promoted out of the timeline, and whether the block is repeated at the
#: end as well as the head.
#:
#: BOTH OFF. A share of 0.0 leaves the floor above as the whole rule, which is the ten this
#: renderer always used. 0.35 was shipped on the paper's argument and then measured: it
#: costs about 10 per cent more rendered characters - every promoted memory leaves a
#: pointer line behind, which I had wrongly called token-neutral - and it cannot be shown
#: to help, because the only instrument available without an answerer is evidence recall,
#: and that is saturated at 0.9785 across every cell of the ablation. A knob that has a
#: measured cost and an unmeasurable benefit does not belong on by default.
#:
#: Ten was chosen to stay inside the first screen, against a bundle that is a hundred memories
#: at judged depth - so ninety per cent of the ranked set existed only inside a ~21,000
#: character chronological list, which is the trough the paper above measures at 53.8 per cent.
#: Read directly, that is what the failures look like: sixteen of fifty-two misses declined
#: with the evidence present, and two pairs prove it was reachable - "which events has Jon
#: participated in" answered "networking events, one on 20 June 2023" while "when did Jon
#: visit networking events" answered "I don't know", in the same run over the same corpus.
#:
#: Two changes, both from the same figures. The block is now a share of what is actually in
#: the bundle, so it scales with depth instead of shrinking to a tenth of it. And it is
#: repeated immediately before the end of the prompt, because the paper's own numbers put the
#: tail at 63.2 per cent against the middle's 53.8 - the best evidence now sits at both peaks
#: of the U rather than only the first.
MOST_RELEVANT_SHARE = 0.0
#: OFF by default, and an ablation knob rather than a decision.
#:
#: Repeating the top few at the tail would put the best evidence at both peaks of the U, and
#: the paper's own tail figure (63.2) is well above its middle (53.8). But it breaks an
#: invariant this renderer was built on and which a test pins: every memory body appears
#: exactly once, the promoted ones standing in the timeline as a pointer rather than a second
#: copy. Duplicating eight bodies is real tokens out of a budget that evicts memories when it
#: is exceeded, so it is worth measuring and not worth assuming. The proportional head block
#: above costs nothing and carries most of the same argument; this is the part that has to
#: earn its place.
REPEAT_MOST_RELEVANT_AT_END = False
#: How many of the ranked block the tail repeat carries. The tail is a reminder, not a second
#: copy of the bundle: past a handful it costs tokens the timeline needs.
MOST_RELEVANT_TAIL = 8

#: What stands in the timeline for a memory printed in full under "Most relevant". Keeps
#: the date, the weekday and the speaker in their chronological place - which is what the
#: timeline is for - without paying for the body twice.
SHOWN_ABOVE = "(see Most relevant)"


def _observed(m: Any) -> tuple[str, str]:
    """``("2023-05-08", "Mon")`` from the memory's observed_at; empty when it has none.

    The weekday is what the model is worst at deriving and what LoCoMo's temporal questions
    ask for ("the Sunday before 25 May 2023"), and it costs one token.
    """
    raw = str(m.attributes.get("observed_at") or "")
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return raw[:10], ""
    return when.date().isoformat(), when.strftime("%a")


def _most_relevant_count(total: int) -> int:
    """How many memories are promoted out of the timeline.

    A share rather than a fixed count, so the block does not shrink to a tenth of the bundle
    the moment the depth doubles. Floored at ``MOST_RELEVANT_MAX`` so a small bundle keeps the
    behaviour it already had.
    """
    return max(MOST_RELEVANT_MAX, math.ceil(total * MOST_RELEVANT_SHARE))


#: Pull the memories extracted from the same turn in alongside one that ranked. Proposition-
#: sized extraction means a gold turn often becomes several memories, none of which carries
#: enough of it alone: on the full set the best SINGLE memory matches 69.6% of gold evidence
#: while the whole bundle matches 93.3%. That 23.7-point gap is not extraction losing
#: information, it is the retrieval unit being smaller than the question's unit - and no
#: reordering can close it, because every fragment is individually a weak match. Reuniting a
#: turn's fragments in the block the model reads first is the thing that can.
GROUP_BY_SOURCE = True


def _source_ids(m: Any) -> set[str]:
    return {
        sid
        for ref in (getattr(m, "evidence", None) or [])
        if (sid := getattr(ref, "source_id", None))
    }


def _head_with_siblings(memories: Sequence[Any], count: int) -> list[Any]:
    """The top ``count``, each followed by the memories extracted from the same turn."""
    if not GROUP_BY_SOURCE:
        return list(memories[:count])
    by_source: dict[str, list[Any]] = {}
    for m in memories:
        for sid in _source_ids(m):
            by_source.setdefault(sid, []).append(m)
    head: list[Any] = []
    seen: set[str] = set()
    for m in memories:
        if len(head) >= count:
            break
        if m.item_id in seen:
            continue
        head.append(m)
        seen.add(m.item_id)
        for sid in _source_ids(m):
            for sib in by_source.get(sid, ()):
                if len(head) >= count:
                    break
                if sib.item_id not in seen:
                    head.append(sib)
                    seen.add(sib.item_id)
    return head


def _memory_line(m: Any, *, body: str | None = None) -> str:
    """One memory as one line: citation, date, weekday, speaker, text - each exactly once.

    The speaker is the subject's user id (``user:caroline`` -> ``caroline``). Its original
    casing is *not* recoverable here: the subject is written from the execution context's
    user id, which is an identifier, and nothing carries the display name the speaker was
    ingested under. Title-casing it would invent one and would mangle opaque ids, so the id
    is printed as it is stored.
    """
    day, weekday = _observed(m)
    subject = str(m.attributes.get("subject") or "")
    who = subject.split(":", 1)[1] if subject.startswith("user:") else ""
    parts = (
        f"- [{m.citation}]",
        day,
        weekday,
        f"{who}:" if who else "",
        m.text if body is None else body,
    )
    return " ".join(p for p in parts if p)


class ContextBundle(BaseModel):
    """Bounded, ranked context for one query in one execution context."""

    model_config = ConfigDict(frozen=True)

    query: str
    query_type: QueryType
    bundle_id: str = Field(
        default="", description="tenant-bound handle for /v1/verify while the bundle is cached"
    )
    conversation: ConversationWindow
    memories: list[ContextItem] = Field(default_factory=list)
    knowledge: list[ContextItem] = Field(default_factory=list)
    graph_facts: list[ContextItem] = Field(default_factory=list)
    summaries: list[ContextItem] = Field(default_factory=list)
    evidence: EvidenceReport
    token_budget: int
    token_estimate: int
    cache_hit: bool = False
    revision_fingerprint: str = Field(
        default="", description="revisions this bundle was built from"
    )
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        """Plain-text rendering suitable for a system prompt. Applications may ignore it."""
        parts: list[str] = []
        if self.conversation.summary:
            parts.append(f"## Conversation summary\n{self.conversation.summary}")
        if self.conversation.rendered:
            parts.append(f"## Recent conversation\n{self.conversation.rendered}")
        if self.memories:
            # ``self.memories`` is the retrieval ranking, best first; the timeline below is a
            # sorted copy, so both orders are available and neither is thrown away.
            #
            # Oldest first, each with its date and who it is about. Every system that scores
            # well on conversational memory renders this way; a rank-ordered list with no
            # time in it made the model anchor on position and fail date arithmetic. But
            # chronological alone discards the ranking entirely, and the position a memory
            # then lands in decides how well it is read, so the best-ranked few are repeated
            # above the timeline (see MOST_RELEVANT_MAX).
            ranked = _head_with_siblings(self.memories, _most_relevant_count(len(self.memories)))
            shown: set[str] = set()
            if len(ranked) < len(self.memories):  # otherwise the block is the whole timeline
                shown = {m.item_id for m in ranked}
                parts.append("## Most relevant\n" + "\n".join(_memory_line(m) for m in ranked))
            ordered = sorted(self.memories, key=lambda m: str(m.attributes.get("observed_at", "")))
            parts.append(
                "## Memories\n"
                + "\n".join(
                    _memory_line(m, body=SHOWN_ABOVE if m.item_id in shown else None)
                    for m in ordered
                )
            )
        if self.memories and REPEAT_MOST_RELEVANT_AT_END and len(self.memories) > MOST_RELEVANT_MAX:
            # The tail of the prompt is the second attention peak (63.2 against the middle's
            # 53.8 in arXiv 2307.03172), and it is the last thing read before the question.
            tail = self.memories[:MOST_RELEVANT_TAIL]
            parts.append("## Most relevant, again\n" + "\n".join(_memory_line(m) for m in tail))
        if self.graph_facts:
            parts.append(
                "## Facts\n" + "\n".join(f"- [{f.citation}] {f.text}" for f in self.graph_facts)
            )
        if self.summaries:
            parts.append(
                "## Summaries\n" + "\n".join(f"- [{s.citation}] {s.text}" for s in self.summaries)
            )
        if self.knowledge:
            parts.append(
                "## Knowledge\n"
                + "\n\n".join(
                    f"[{k.citation}]"
                    + (f" ({k.section_path})" if k.section_path else "")
                    + f"\n{k.text}"
                    for k in self.knowledge
                )
            )
        if self.evidence.status is not EvidenceStatus.COMPLETE:
            parts.append(f"## Evidence status\n{self.evidence.status}")
        return "\n\n".join(parts)
