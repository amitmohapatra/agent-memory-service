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
from memory_service.domain.memory import aggregate_statement, unverified_representation
from memory_service.domain.predicates import is_multi_valued
from memory_service.domain.tools import ToolCandidate, ToolHints

#: The native graph's link between a memory's speaker and an entity it names.
MENTIONS = "mentions"
#: Tool candidates the rendered Tools section shows (the next step is shown even past it).
TOOLS_SHOWN_MAX = 3

#: What produced ``ContextItem.score``; the scales are not comparable across kinds.
ScoreKind = Literal["fusion", "exact"]
#: Handle prefix -> the bundle list it numbers (``ContextBundle.handles``).
HANDLE_PREFIXES: dict[str, str] = {
    "m": "memories",
    "f": "graph_facts",
    "s": "summaries",
    "d": "knowledge",
}


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
    #: Where ``score`` came from: a fusion rank score or an exact identifier hit.
    score_kind: ScoreKind = Field(
        default="fusion",
        description=(
            "fusion is rank aggregation; exact is an identifier match."
            " Use relevance to compare across kinds."
        ),
    )
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


class WindowMessage(BaseModel):
    """One message of the conversation window."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    role: str
    text: str


class ConversationWindow(BaseModel):
    """The thread's messages after its durable summary, the most recent that fit."""

    model_config = ConfigDict(frozen=True)

    thread_id: str | None = None
    message_ids: list[str] = Field(default_factory=list)
    messages: list[WindowMessage] = Field(default_factory=list)
    rendered: str = ""
    token_estimate: int = 0


class ProfileBlockView(BaseModel):
    """A pinned profile block as the context carries it."""

    model_config = ConfigDict(frozen=True)

    block: str
    text: str
    version: int


class ThreadSummaryView(BaseModel):
    """The thread's durable summary: every message up to ``covers_to_sequence``."""

    model_config = ConfigDict(frozen=True)

    text: str
    covers_to_sequence: int
    version: int


class ProcedureView(BaseModel):
    """A procedure learned for the task."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)
    success_rate: float = 0.0
    support: int = 0


class UnusedEvidence(BaseModel):
    """Retrieved but not packed (ranked out or over budget): the grounding cascade scans
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


#: Keep retrieved fragments of a source together in the head. This changes rendering,
#: not candidate recall; its effect on answer accuracy must be measured with a reader.
GROUP_BY_SOURCE = True


def _source_ids(m: Any) -> list[tuple[str, str]]:
    # Stable order matters for memories with multiple sources. A set makes the rendered
    # head depend on PYTHONHASHSEED; source type prevents unrelated id namespaces joining.
    return list(
        dict.fromkeys(
            (getattr(ref, "source_type", ""), sid)
            for ref in (getattr(m, "evidence", None) or [])
            if (sid := getattr(ref, "source_id", None))
        )
    )


def _head_with_siblings(memories: Sequence[Any], count: int) -> list[Any]:
    """The top ``count``, each followed by the memories extracted from the same turn."""
    if not GROUP_BY_SOURCE:
        return list(memories[:count])
    by_source: dict[tuple[str, str], list[Any]] = {}
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


def _memory_line(m: Any, ref: str, *, body: str | None = None) -> str:
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
        f"- [{ref}]",
        "model-extracted, unverified; source speaker"
        if unverified_representation(m.attributes)
        else "",
        (
            "sources through"
            if m.attributes.get("source_observed_to")
            else "summary created"
            if m.attributes.get("derived")
            else ""
        ),
        day,
        weekday,
        f"{who}:" if who else "",
        m.text if body is None else body,
        _resolved_dates(m) if body is None else "",
    )
    return " ".join(p for p in parts if p)


def _resolved_dates(m: Any) -> str:
    """``(three days ago = 2023-05-05)``: the relative dates the text names, resolved at
    ingest against the day it was said, so the reader never does that arithmetic.

    Named and counted offsets only, never weekday phrases like "last Tuesday" - see
    ``modules.memory.temporal`` for why guessing those is worse than leaving them."""
    mentions = m.attributes.get("dated_mentions") or ()
    pairs = [
        f"{mention.get('text')} = {mention.get('date')}"
        for mention in mentions
        if isinstance(mention, dict) and mention.get("text") and mention.get("date")
    ]
    return f"({'; '.join(pairs)})" if pairs else ""


#: How many memories of one multi-valued slot it takes before they are gathered into a single
#: dated block rather than printed as separate lines.
#:
#: A bundle routinely holds several statements about the same slot - four "participated in"
#: memories about one person, said on four different days. Printed as four lines among forty
#: they read as four unrelated facts, and a question that needs all of them ("which events has
#: Jon been to") is answered from whichever line was read last. Fragmentation is the largest
#: loss bucket on LoCoMo: 226 of 416 misses (ADR 0024, D6 step 4). One block per (subject,
#: predicate) says these values are all current, and says it in the same words a consolidated
#: belief would use - so the read-side grouping and a write-path belief cannot disagree.
#:
#: Two is the floor because a "group" of one is just the memory's own line.
AGGREGATE_MIN_MEMBERS = 2


def _aggregate_key(m: Any) -> tuple[str, str] | None:
    """``(subject, predicate)`` when this memory is one value of a multi-valued slot.

    ``None`` for anything that must keep its own line: a memory with no subject or predicate,
    a single-valued slot (where the newest value is the answer and older ones are superseded
    history, not a set), and a derived memory, which is already an aggregate of its sources.
    """
    if m.attributes.get("derived"):
        return None
    subject = str(m.attributes.get("subject") or "")
    predicate = str(m.attributes.get("predicate") or "")
    if not subject or not is_multi_valued(predicate):
        return None
    return subject, predicate


def _aggregate_line(
    subject: str, predicate: str, members: Sequence[Any], refs: dict[str, str]
) -> str:
    """Every gathered value of one slot as a single citable block, oldest first.

    Carries every member's citation, so the reader can still attribute each statement, and
    every member's resolved relative dates. The unverified warning is kept if it applies to
    any member: an aggregate is no more trustworthy than its least trustworthy source.
    """
    citations = "; ".join(dict.fromkeys(refs[m.item_id] for m in members))
    warning = (
        "model-extracted, unverified; source speaker"
        if any(unverified_representation(m.attributes) for m in members)
        else ""
    )
    dated = [
        (_observed(m)[0], " ".join(filter(None, (m.text, _resolved_dates(m))))) for m in members
    ]
    head = " ".join(filter(None, (f"- [{citations}]", warning)))
    return f"{head} {aggregate_statement(subject, predicate, dated)}"


def _timeline_lines(ordered: Sequence[Any], shown: set[str], refs: dict[str, str]) -> list[str]:
    """The chronological block, with each multi-valued slot gathered into one dated block.

    ``ordered`` is oldest first, so a gathered block is oldest first too and lands at its
    oldest member's place in the timeline. ``shown`` are the ids already printed in full under
    "Most relevant"; those stand here as a pointer and are never gathered, which keeps the
    invariant this renderer is built on: every memory body appears exactly once.
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    for m in ordered:
        key = _aggregate_key(m)
        if key is not None and m.item_id not in shown:
            groups.setdefault(key, []).append(m)
    gathered = {
        m.item_id: key
        for key, members in groups.items()
        if len(members) >= AGGREGATE_MIN_MEMBERS
        for m in members
    }
    lines: list[str] = []
    done: set[tuple[str, str]] = set()
    for m in ordered:
        key = gathered.get(m.item_id)
        if key is None:
            body = SHOWN_ABOVE if m.item_id in shown else None
            lines.append(_memory_line(m, refs[m.item_id], body=body))
        elif key not in done:
            done.add(key)
            lines.append(_aggregate_line(*key, groups[key], refs))
    return lines


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
    profile: list[ProfileBlockView] = Field(default_factory=list)
    thread_summary: ThreadSummaryView | None = None
    procedures: list[ProcedureView] = Field(default_factory=list)
    tools: ToolHints | None = None

    def handles(self) -> dict[str, str]:
        """Short, per-bundle handles for what the bundle carries, handle -> item id: ``m1``..
        for memories, ``f1``.. for graph facts, ``s1``.. for summaries, ``d1``.. for
        document passages, in bundle order. The rendered prompt cites by them, and the
        service resolves them back within the bundle (update, forget, verify)."""
        return {
            f"{prefix}{i}": item.item_id
            for prefix, items in HANDLE_PREFIXES.items()
            for i, item in enumerate(getattr(self, items), start=1)
        }

    def redundant(self) -> frozenset[str]:
        """Items the context would only repeat, left out of what it shows: a memory every
        source of which is a message the recent conversation already shows, and a
        ``mentions`` fact whose object the shown text already names. They stay in the bundle
        (handles, verification); only the rendering and the wire form omit them."""
        shown = set(self.conversation.message_ids)
        out: set[str] = set()
        for m in self.memories:
            sources = {e.message_id for e in m.evidence if e.message_id}
            if shown and sources and sources <= shown:
                out.add(m.item_id)
        said = " ".join(
            [self.conversation.rendered, *(m.text for m in self.memories if m.item_id not in out)]
        ).casefold()
        for f in self.graph_facts:
            named = str(f.attributes.get("object") or "").casefold()
            if f.attributes.get("predicate") == MENTIONS and named and named in said:
                out.add(f.item_id)
        return frozenset(out)

    def render(self) -> str:
        """Plain-text rendering suitable for a system prompt, citing by handle. Nothing is
        shown twice: what ``redundant`` names is left out."""
        refs = {item_id: handle for handle, item_id in self.handles().items()}
        parts = pinned_sections(self.profile, self.thread_summary, self.procedures, self.tools)
        if self.conversation.rendered:
            parts.append(f"## Recent conversation\n{self.conversation.rendered}")
        repeated = self.redundant()
        memories = [m for m in self.memories if m.item_id not in repeated]
        facts = [f for f in self.graph_facts if f.item_id not in repeated]
        if memories:
            # ``memories`` is the retrieval ranking, best first; the timeline below is a
            # sorted copy, so both orders are available and neither is thrown away.
            #
            # Oldest first, each with its date and who it is about. Every system that scores
            # well on conversational memory renders this way; a rank-ordered list with no
            # time in it made the model anchor on position and fail date arithmetic. But
            # chronological alone discards the ranking entirely, and the position a memory
            # then lands in decides how well it is read, so the best-ranked few are repeated
            # above the timeline (see MOST_RELEVANT_MAX).
            ranked = _head_with_siblings(memories, _most_relevant_count(len(memories)))
            shown: set[str] = set()
            if len(ranked) < len(memories):  # otherwise the block is the whole timeline
                shown = {m.item_id for m in ranked}
                parts.append(
                    "## Most relevant\n"
                    + "\n".join(_memory_line(m, refs[m.item_id]) for m in ranked)
                )
            ordered = sorted(memories, key=lambda m: str(m.attributes.get("observed_at", "")))
            parts.append("## Memories\n" + "\n".join(_timeline_lines(ordered, shown, refs)))
        if facts:
            parts.append("## Facts\n" + "\n".join(f"- [{refs[f.item_id]}] {f.text}" for f in facts))
        if self.summaries:
            parts.append(
                "## Summaries\n"
                + "\n".join(f"- [{refs[s.item_id]}] {s.text}" for s in self.summaries)
            )
        if self.knowledge:
            parts.append(
                "## Knowledge\n"
                + "\n\n".join(
                    f"[{refs[k.item_id]}]"
                    + (f" ({k.section_path})" if k.section_path else "")
                    + f"\n{k.text}"
                    for k in self.knowledge
                )
            )
        if self.evidence.status is not EvidenceStatus.COMPLETE:
            parts.append(f"## Evidence status\n{self.evidence.status}")
        return "\n\n".join(parts)


def profile_section(profile: Sequence[ProfileBlockView]) -> str | None:
    if not profile:
        return None
    return "## Profile\n" + "\n".join(f"### {b.block}\n{b.text}" for b in profile)


def summary_section(summary: ThreadSummaryView | None) -> str | None:
    return f"## Conversation summary\n{summary.text}" if summary else None


def procedures_section(procedures: Sequence[ProcedureView]) -> str | None:
    if not procedures:
        return None
    lines = []
    for p in procedures:
        steps = " -> ".join(str(step.get("tool")) for step in p.steps)
        title = f"{p.title}: " if p.title else ""
        lines.append(f"- {title}{steps} (worked {p.success_rate:.0%} of {p.support} runs)")
    return "## Procedures that worked for this task\n" + "\n".join(lines)


def tools_section(hints: ToolHints | None) -> str | None:
    """The tools that fit, best first, each with its confidence, the arguments already found
    and the required ones nothing found - what the model acts on and what it must ask."""
    if hints is None or not hints.candidates:
        return None
    shown = [
        c for i, c in enumerate(hints.candidates) if i < TOOLS_SHOWN_MAX or c.name == hints.next
    ]
    return "## Tools\n" + "\n".join(_tool_line(c, hints) for c in shown)


def _tool_line(candidate: ToolCandidate, hints: ToolHints) -> str:
    prefix = f"{candidate.name}."
    found = {
        k.removeprefix(prefix): p.value for k, p in hints.prefill.items() if k.startswith(prefix)
    }
    absent = [m for m in hints.missing if m.tool == candidate.name]
    head = f"- {candidate.name} (confidence {candidate.confidence:.2f}"
    head += ", next step)" if candidate.name == hints.next else ")"
    details = [", ".join(f"{arg} = {value!r}" for arg, value in found.items())] if found else []
    details += [f"missing {m.arg}: {m.question}" for m in absent]
    return head + (": " + "; ".join(details) if details else "")


def pinned_sections(
    profile: Sequence[ProfileBlockView],
    summary: ThreadSummaryView | None,
    procedures: Sequence[ProcedureView],
    tools: ToolHints | None,
) -> list[str]:
    """What every prompt starts from: the profile, the thread summary, the procedures and
    the tool hints, in that order."""
    sections = (
        profile_section(profile),
        summary_section(summary),
        procedures_section(procedures),
        tools_section(tools),
    )
    return [s for s in sections if s]
