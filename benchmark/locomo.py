"""Conversational memory accuracy on LoCoMo — a corpus nobody here wrote.

Every other gate in ``benchmark/results`` scores against fixtures authored in this repository
and reads 1.0, which measures self-consistency rather than quality. This measures the path
that is actually the product: turns go in as observations, the pipeline extracts and
consolidates them, and questions are answered from what retrieval surfaces.

    uv run python -m benchmark.locomo --conversations 2       # a quick pass
    make bench-locomo                                         # the full set, real models

**What is measured, and what is not.** The service supplies evidence; a model writes the
prose. With ``models.llm.enabled=false`` there is no generation here, so scoring generated
text would be measuring nothing. What is scored instead is *evidence recall*: LoCoMo annotates
the dialogue turns that support each answer, and the question is whether retrieval surfaces
them. That is the half this service is responsible for.

Category 5 is scored differently and matters most. Those questions are **adversarial — they
have no answer in the conversation**. For them the correct behaviour is to abstain, so the
metric is the abstention rate, not recall. A system that answers them confidently is exactly
the failure mode this product exists to avoid.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.common import provenance, reset_store, write_result
from benchmark.env import bench_overrides, bench_retrieval
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.constants import FROZEN_MODELS
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import _memory_line, _most_relevant_count
from memory_service.domain.enums import ObservationKind, Visibility
from memory_service.domain.ids import new_id
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark" / "data" / "locomo10.json"

#: LoCoMo's own category numbering.
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
#: This benchmark's own tenant in the shared vector store; see benchmark/external_retrieval.
TENANT = "bench_conv"

ADVERSARIAL = 5
_SESSION = re.compile(r"^session_(\d+)$")


#: A run is only comparable to another run if both answered the *same* questions.
#: ``--questions N`` took the first N, and LoCoMo does not interleave its categories evenly,
#: so a head-slice silently drops whole categories — a 40-question cap on conversation 1
#: contained no adversarial questions at all, which is the category that matters most here.
#: This takes a seeded, per-category share instead, so a sample is representative and the
#: ablation arms see an identical question set.
SAMPLE_SEED = 20240919


def _stratified(questions: list[dict], size: int) -> list[dict]:
    """``size`` questions drawn proportionally from each category, deterministically."""
    if size <= 0 or size >= len(questions):
        return questions
    buckets: dict[int, list[dict]] = defaultdict(list)
    for q in questions:
        buckets[int(q.get("category", 0))].append(q)
    rng = random.Random(SAMPLE_SEED)
    picked: list[dict] = []
    for category in sorted(buckets):
        group = buckets[category]
        # proportional share, but never zero: a category present in the corpus is present in
        # the sample, because "we didn't measure it" and "it scored 0" are different answers.
        share = max(1, round(size * len(group) / len(questions)))
        picked.extend(rng.sample(group, min(share, len(group))))
    return picked


def _sessions(conversation: dict) -> list[tuple[str, list[dict]]]:
    keys = sorted(
        (k for k in conversation if _SESSION.match(k)),
        key=lambda k: int(_SESSION.match(k).group(1)),  # type: ignore[union-attr]
    )
    return [(k, conversation[k]) for k in keys if isinstance(conversation[k], list)]


_SESSION_TIME = "%I:%M %p on %d %B, %Y"  # LoCoMo: "1:56 pm on 8 May, 2023"


def _session_time(when: str) -> datetime | None:
    """The session's own timestamp, as a datetime, or None when the dataset gives none."""
    try:
        return datetime.strptime(when.strip(), _SESSION_TIME).replace(tzinfo=UTC)
    except ValueError:
        return None


#: Ingest every turn inside a conversation thread, the way `/v1/messages` does.
#:
#: This benchmark has always ingested with no thread_id, and that is not a detail: the
#: verbatim catch-all in native.py - the rule that keeps a turn no extraction pattern matched,
#: and which its own docstring credits with rescuing 452 of 788 LoCoMo turns - returns None
#: for any turn inside a thread. Every real chat message has one, because append_message
#: raises without it. So the scores in this repository describe a configuration production
#: cannot run, and the gap has never appeared in a number.
#:
#: Off by default, so the default run stays comparable with every result already recorded.
THREADED_INGEST = os.environ.get("BENCH_THREADED_INGEST", "").strip().lower() in {"1", "true"}


def _speaker_ctx(ctx: MemoryExecutionContext, speaker: str) -> MemoryExecutionContext:
    """The ingest context for one speaker: same tenant and workspace, their own user id."""
    update: dict[str, object] = {"user_id": speaker.strip().lower() or ctx.user_id}
    if THREADED_INGEST:
        # One thread for the whole conversation, which is what a chat client would do. The
        # session and turn ids are what append_message would require alongside it.
        update |= {
            "thread_id": f"thr_{ctx.workspace_id}_bench",
            "session_id": f"ses_{ctx.workspace_id}_bench",
            "turn_id": new_id("turn"),
        }
    return ctx.model_copy(update=update)


async def _ingest_conversation(container, ctx, conversation: dict) -> dict[str, str]:
    """Every turn becomes an observation. Returns dia_id -> the text that was submitted."""
    uow_factory = container.services["uow_factory"]
    memory = container.services["memory"]
    turns: dict[str, str] = {}
    for session_key, session in _sessions(conversation):
        when = conversation.get(f"{session_key}_date_time", "")
        occurred_at = _session_time(when)
        for turn in session:
            dia_id, speaker = turn.get("dia_id"), turn.get("speaker", "")
            body = turn.get("text") or turn.get("clean_text") or ""
            # Shared images are turns too. 95 of 233 answerable questions in the first two
            # conversations touch one, and for ten the answer exists only in the caption;
            # Mem0's harness appends it the same way (memory-benchmarks locomo/run.py).
            if caption := (turn.get("blip_caption") or "").strip():
                query = (turn.get("query") or "").strip()
                body = (
                    f"{body} [Shared an image{f' of {query}' if query else ''}: {caption}]".strip()
                )
            if not dia_id or not body:
                continue
            turns[dia_id] = body
            # The turn is submitted verbatim. It used to be submitted as
            # "[1:56 pm on 8 May, 2023] Caroline: <body>", and that 35-character stamp cost
            # on three fronts. The renderer prints the date and the speaker itself, from
            # observed_at and from the subject, so every bundle line carried both twice, in
            # two formats and two casings. Ingest splits sentences on .!? only
            # (native._SENTENCE_SPLIT), so the stamp stayed glued to the first sentence of
            # every turn and each start-anchored rule - _FACT's ^(?P<subject>...),
            # _PREF_PLEASE, _DECISION_PREFIX - and the _CHITCHAT.match noise filter could
            # never fire on it: the first sentence of a turn was unparseable by
            # construction and a greeting-only turn was stored verbatim as noise. And the
            # dense vector of a short turn was dominated by a prefix shared with every
            # other turn in the corpus.
            #
            # Nothing is lost. occurred_at carries the session date into the memory's
            # temporal fields, which is where the renderer and the temporal questions read
            # it from, and each speaker is their own user, so the fact "I moved from
            # Sweden" gets the subject user:caroline rather than a shared id that makes
            # Caroline's and Melanie's facts indistinguishable - which is what the graph,
            # the subject check and the answerer all key on. WORKSPACE visibility is
            # anchored on the workspace, not the user, so the questioner - a third context,
            # made a member of the workspace above - still sees every turn.
            async with uow_factory() as uow:
                await memory.submit_observation(
                    uow,
                    _speaker_ctx(ctx, speaker),
                    kind=ObservationKind.MESSAGE,
                    content=body,
                    # TENANT, because the benchmark's two speakers share one corpus and
                    # every question may draw on either. This was WORKSPACE, which meant the
                    # same thing here and no longer exists as an audience.
                    hints=ProcessingHints(visibility=Visibility.TENANT),
                    occurred_at=occurred_at,
                )
                await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return turns


#: Words that carry no retrieval signal. Keeping them in the overlap inflates every score:
#: "The sunday before 25 May 2023" would match any chunk containing "the" and "before".
_STOPWORDS = frozenset(
    """a an the and or but if of in on at to for from by with was were is are am be been
    being do does did done have has had it its this that these those there here what when
    where who whom which how why not no nor as so than then too very can will just about
    into over under again further once s t""".split()
)


def _normalise(value: object) -> str:
    # LoCoMo answers are sometimes numbers, not strings
    value = "" if value is None else str(value)
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", value or "").casefold()).strip()


def _content_tokens(text: object) -> set[str]:
    """Normalised tokens with the stopwords removed."""
    return {t for t in _normalise(text).split() if t and t not in _STOPWORDS}


def _overlap(needle: set[str], haystack: set[str]) -> float:
    """Fraction of ``needle`` present in ``haystack``. 0.0 when there is nothing to find."""
    return round(len(needle & haystack) / len(needle), 4) if needle else 0.0


def _evidence_ranks(memories: Sequence[Any], evidence: Sequence[str]) -> list[int | None]:
    """Where each gold evidence turn landed in the retrieval ranking, or ``None`` if absent.

    ``evidence_all_hit`` flattens the whole bundle into one token set before scoring
    (``found = _content_tokens(bundle.render())``), so it cannot see order at all. Every
    ordering change measured so far - the promotion share, the repeated tail, subsumption
    collapse - was invisible to it *by construction* and read as an exact null: a knob that
    only permutes the bundle cannot move a metric computed over the bundle's union. This is
    the same overlap test applied per memory instead of over the render, which is the
    smallest change that makes an ordering experiment falsifiable at all.

    Scored against ``_memory_line`` rather than the raw body, because that is the text the
    prompt actually shows - date, speaker and all.
    """
    ranked = [_content_tokens(_memory_line(m)) for m in memories]
    out: list[int | None] = []
    for text in evidence:
        needle = _content_tokens(text)
        best: int | None = None
        best_score = 0.0
        if needle:
            for position, haystack in enumerate(ranked):
                score = _overlap(needle, haystack)
                if score > best_score:
                    best, best_score = position, score
        out.append(best if best_score >= EVIDENCE_OVERLAP_HIT else None)
    return out


def _evidence_ranks_by_arm(
    memories: Sequence[Any], evidence: Sequence[str]
) -> dict[str, list[int | None]]:
    """Where each gold turn landed within EACH arm's own contribution, or ``None`` if that
    arm never carried it.

    ``Candidate.retrievers`` survives fusion (``rrf_fuse`` returns a per-record arm list) and
    survives dedup by set-union rather than being dropped, and it reaches this harness as the
    bundle's ``retrievers``. So the arm that produced a memory is still readable at scoring
    time, and two failures that ``evidence_ranks`` cannot tell apart become separable:

        gold carried by a ``graph``-only memory that never reached the head
            -> FUSION suppressed it; the evidence was retrieved
        gold carried by no memory of any arm
            -> RETRIEVAL never found it; no amount of re-ranking will help

    Those have opposite fixes, and choosing between candidate-seeded expansion and fusion
    work without separating them is guessing. The rank recorded is the position within that
    arm's own sublist - "where would this have ranked if only this arm ran" - which is what
    an arm-alone column in an oracle table means.
    """
    lines = [
        (_content_tokens(_memory_line(m)), set(getattr(m, "retrievers", None) or []))
        for m in memories
    ]
    arms = sorted({a for _, names in lines for a in names})
    out: dict[str, list[int | None]] = {}
    for arm in arms:
        sub = [tokens for tokens, names in lines if arm in names]
        ranks: list[int | None] = []
        for text in evidence:
            needle = _content_tokens(text)
            best: int | None = None
            best_score = 0.0
            if needle:
                for position, haystack in enumerate(sub):
                    score = _overlap(needle, haystack)
                    if score > best_score:
                        best, best_score = position, score
            ranks.append(best if best_score >= EVIDENCE_OVERLAP_HIT else None)
        out[arm] = ranks
    return out


def _evidence_reconstructed(memories: Sequence[Any], evidence: Sequence[str]) -> float | None:
    """Coverage when a gold turn is matched against the GROUP of memories it produced.

    ``_evidence_ranks`` asks whether any ONE memory carries a gold turn. On the full set that
    is true for 69.6% of gold items while ``evidence_all_hit`` - the same test over the whole
    flattened bundle - is 93.3%. The 23.7-point difference is a gold turn whose content is
    spread across several memories, none of which carries enough of it alone:

        turn: "Amit joined in May 2025 as Principal Engineer and managed teams in
               India, the Netherlands and Budapest."
        ->  M1 joined in May 2025 | M2 was Principal Engineer | M3 managed India | ...

    That is proposition-sized extraction working as intended, not extraction losing
    information - and no amount of reordering can fix it, because the unit being ranked is
    smaller than the unit being asked for. The fix is to reconstruct the parent.

    Every memory carries ``EvidenceRef.source_id`` back to the turn it came from, so the
    group is already expressible: match the gold turn against the union of the memories that
    share its source. This measures the ceiling that reconstruction would reach.
    """
    groups: dict[str, set[str]] = {}
    for m in memories:
        for ref in getattr(m, "evidence", None) or []:
            sid = getattr(ref, "source_id", None)
            if sid:
                groups.setdefault(sid, set()).update(_content_tokens(_memory_line(m)))
    if not evidence:
        return None
    hits = 0
    for text in evidence:
        needle = _content_tokens(text)
        if not needle:
            continue
        best = max((_overlap(needle, g) for g in groups.values()), default=0.0)
        hits += best >= EVIDENCE_OVERLAP_HIT
    return round(hits / len(evidence), 4)


def _rank_metrics(ranks: Sequence[int | None], head: int) -> dict[str, float | None]:
    """Mean reciprocal rank, and whether the evidence reached the block the model reads first.

    ``head`` is ``_most_relevant_count(len(memories))`` - the size of the "## Most relevant"
    block that ``ContextBundle.render`` puts above the chronological timeline. Evidence below
    it is still *present*, which is all ``evidence_all_hit`` ever asked; whether being below
    it costs anything is the question these two numbers exist to answer.
    """
    if not ranks:
        return {
            "evidence_mrr": None,
            "evidence_in_head": None,
            "evidence_worst_rank": None,
            "complete_in_candidates": None,
            "complete_in_head": None,
            "evidence_head_size": head,
        }
    found = [r for r in ranks if r is not None]
    return {
        "evidence_mrr": round(sum(1.0 / (r + 1) for r in found) / len(ranks), 4),
        "evidence_in_head": round(sum(1 for r in found if r < head) / len(ranks), 4),
        # a multi-hop question is bounded by its LAST evidence item, the same way
        # evidence_min_overlap bounds recall
        "evidence_worst_rank": max(found) if len(found) == len(ranks) else None,
        # The two that separate a retrieval failure from a selection failure. Both were
        # computed by hand from ``evidence_ranks`` after every run so far, which meant the
        # number that actually drove the roadmap lived in a throwaway script instead of in
        # the result file. ``complete_in_candidates`` is the ceiling - the fraction of
        # questions whose EVERY gold turn was retrieved at all - and the distance from it
        # down to ``complete_in_head`` is everything selection is costing.
        "complete_in_candidates": len(found) == len(ranks),
        "complete_in_head": len(found) == len(ranks) and all(r < head for r in found),
        "evidence_head_size": head,
    }


def _rank_summary(records: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """``evidence_mrr`` and ``evidence_in_head`` averaged over the answerable rows.

    Adversarial rows are excluded for the same reason they are excluded from
    ``evidence_all_recall``: they have no gold evidence, so a rank over them is undefined
    rather than zero.
    """
    scored = [
        r for r in records if r["category"] != "adversarial" and r.get("evidence_mrr") is not None
    ]
    if not scored:
        return {
            "evidence_mrr": None,
            "evidence_in_head": None,
            "evidence_reconstructed": None,
            "complete_evidence_in_candidates": None,
            "complete_evidence_in_head": None,
        }
    rec = [
        r["evidence_reconstructed"] for r in scored if r.get("evidence_reconstructed") is not None
    ]
    return {
        "evidence_mrr": round(sum(r["evidence_mrr"] for r in scored) / len(scored), 4),
        "evidence_in_head": round(sum(r["evidence_in_head"] for r in scored) / len(scored), 4),
        "evidence_reconstructed": round(sum(rec) / len(rec), 4) if rec else None,
        "complete_evidence_in_candidates": round(
            sum(1 for r in scored if r.get("complete_in_candidates")) / len(scored), 4
        ),
        "complete_evidence_in_head": round(
            sum(1 for r in scored if r.get("complete_in_head")) / len(scored), 4
        ),
    }


def _answer_present(bundle_text: str, answer: str) -> bool:
    """Strict containment: the whole gold answer as a contiguous substring.

    Kept, but **not** used as the headline metric. It reads near zero on a system that is
    working, because a gold answer like ``"The sunday before 25 May 2023"`` is a phrasing the
    corpus never uses — the turn says "I ran the charity race last Sunday" and the pipeline
    rewrites even that. A run scored this way reported 4.35% recall and 0.0 on single-hop,
    which is the signature of a broken ruler, not a broken system. It stays as a floor: when
    strict containment *does* fire, the evidence is unambiguously there.
    """
    haystack, needle = _normalise(bundle_text), _normalise(answer)
    if not needle:
        return False
    if len(needle) <= 4:
        return needle in haystack.split()
    return needle in haystack


#: Share of the gold answer's content words that must appear in the bundle to count as
#: recalled. LoCoMo's own authors score with an LLM judge; with generation disabled the
#: closest honest proxy is token recall, and 0.6 is the point where "7 May 2023" still matches
#: "May 7th" but a bundle about an unrelated topic does not. Every raw overlap is persisted in
#: the result file, so this threshold can be moved without paying for another run.
ANSWER_OVERLAP_HIT = 0.6

#: The same idea applied to LoCoMo's annotated supporting turns. Lower, because a whole
#: dialogue turn carries far more words than its answer-bearing part.
EVIDENCE_OVERLAP_HIT = 0.5

_EVIDENCE_IDS = re.compile(r"[A-Za-z]+\d+:\d+")


def _evidence_ids(item: dict) -> list[str]:
    """LoCoMo stores evidence as the *string* ``"['D1:3', 'D1:5']"``, not as a list - and one
    list item can itself hold two ids joined by a semicolon (``"D8:6; D9:17"``)."""
    raw = item.get("evidence")
    text = " ".join(str(x) for x in raw) if isinstance(raw, list) else str(raw or "")
    return _EVIDENCE_IDS.findall(text)


def judged_hit(category: str, judged: dict) -> bool:
    """One rule for the run and for a rescore: an adversarial question is answered correctly
    by abstaining; every other category by matching the gold answer. The rescorer once
    compared the category *name* with the numeric constant, so adversarial rows were graded
    on ``correct`` there and on ``abstained`` here, and the two files disagreed by design."""
    if category == CATEGORY_NAMES[ADVERSARIAL]:
        return bool(judged["abstained"])
    return bool(judged["correct"])


#: Generation + judging, the way LoCoMo itself is scored.
#:
#: Token overlap is a proxy: it cannot tell "the bundle contains the answer" from "the bundle
#: contains those words". LoCoMo's authors score with a model, and with a gateway configured
#: this harness can do the same — ask a model to answer *only* from the bundle, then ask a
#: model whether that answer matches the gold one.
#:
#: It also makes category 5 measurable for the first time. Adversarial questions have no
#: answer in the conversation; the correct behaviour is to say so. Nothing in the retrieval
#: path can express that — the abstention gate is lexical and these questions share every
#: term with the conversation — so until something *generates*, "did it abstain?" has no
#: answer to read. With generation on, it is simply whether the model said it did not know.
#: Measured against deepseek-flash on 304 questions: the previous wording ("never guess,
#: as few words as possible") abstained on 58 questions whose evidence was in the bundle
#: (69% of open_domain, 30% of temporal) and shortened 16 correct answers into ones the
#: judge rejected ("Transgender." for "Transgender woman"). Inference from the context and
#: keeping the qualifiers are what the categories ask for; abstaining is still the only
#: correct answer when nothing in the context bears on the question.
async def _build_timed(builder, ctx, question: str, incidents: list[str]) -> tuple[Any, float]:
    """One bundle and the wall time of the attempt that produced it.

    A connection refused by the vector store for a single call is not a measurement of the
    system; it is the benchmark box hiccuping. Three attempts with a short pause, every one
    of them recorded in ``incidents`` so the result says how often it happened, and the
    latency of the attempt that succeeded, not of the retries.
    """
    delay = 2.0
    for attempt in range(3):
        call = time.perf_counter()
        try:
            bundle = await builder.build(ctx, question)
        except Exception as exc:  # noqa: BLE001 - retried and reported, then re-raised
            if attempt == 2:
                raise
            incidents.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            print(
                f"[locomo] store failure, retrying in {delay:.0f}s: {exc!s:.120}", file=sys.stderr
            )
            await asyncio.sleep(delay)
            delay *= 2
            continue
        return bundle, (time.perf_counter() - call) * 1000
    raise AssertionError("unreachable")


#: Abstention is kept for exactly one case, the premise check. Mem0's answer prompt forbids
#: it outright ("NEVER say not specified"), which is part of how their headline is scored;
#: ours had, for one run, also abstained whenever the bundle's evidence status was not
#: COMPLETE - simulated over the first two conversations, that status fired on 63 of 233
#: answerable questions (half of the temporal ones), so the rule turned a retrieval
#: diagnostic into a quarter of the answers being "I don't know". The status is now a cue
#: to re-check, never a reason to refuse.
ANSWER_SYSTEM = (
    "You answer questions about a person's life using only the CONTEXT: dated memories "
    "from their conversations. Read EVERY entry from first to last before answering - do "
    "not stop at the first relevant one; the answer is often spread across several entries "
    "or has to be reasoned from them. "
    # Completeness, unscoped. This used to say \"for counting or listing questions\", and
    # \"What do Melanie's kids like?\" does not read as one, so the rule never fired where
    # it was needed: 44 of 304 graded misses gave one item of a multi-item gold answer.
    "WHENEVER the answer is more than one thing, give ALL of them, separated by commas - "
    "every distinct instance the context supports, not the first one you find, and do not "
    "estimate. Answer directly and "
    "specifically, keeping "
    "the names, dates, places and qualifiers the context gives - do not shorten them. "
    "When asked when, work out the actual date from the dated entries and state it. When "
    "asked what someone would likely do, feel or choose, infer it from what the context "
    "shows about them. "
    # Abstention, narrowed. 115 of 304 graded misses declined with the evidence present.
    # The instruction to check the premise was being read as licence to decline whenever
    # the context was not a verbatim match.
    "Check the question's premise first, and answer the premise before the detail. If the "
    "context NEVER MENTIONS the person, thing or event the question is about - not merely "
    "in less detail, but not at all - or attributes it to someone else, reply with exactly: "
    "I don't know. Many questions are about things that never happened, and saying so is the "
    "correct answer to them, not a failure to find one. "
    "That test is about ABSENCE, never about detail: once the context does record the thing, "
    "partial, indirect or differently worded entries are NOT a reason to decline - answer "
    "from what is there. If the context ends with an Evidence status of INCOMPLETE, "
    "re-check that premise against the named person, then still answer whenever the entries "
    "support an answer. Never use outside knowledge. "
    # Brevity last and explicitly subordinate. It used to be the final instruction and read
    # as the governing one, which is what truncated the lists above.
    "Be brief - a phrase or a single sentence - but never drop part of the answer to be "
    "shorter: completeness comes first."
)

#: Two rulers, reported separately and never mixed.
#:
#: STRICT is ours: the answer must convey the gold fact. LENIENT reproduces the rules in
#: mem0ai/memory-benchmarks (benchmarks/locomo/prompts.py), the judge behind Mem0's
#: published 92.5: at least one correct item from a list answer is CORRECT, paraphrases
#: count, extra detail is fine, dates within 14 days are CORRECT, and WRONG only when zero
#: correct content appears. "Transgender." for "Transgender woman" is WRONG under the first
#: and CORRECT under the second. Their answer prompt also forbids abstaining ("NEVER say
#: 'not specified'"), which is why the adversarial category is absent from their headline;
#: ours keeps abstention, and reports it as its own number under either ruler.
JUDGE_RULERS = {
    "strict": (
        "You grade one answer. `correct` is true when the ANSWER conveys the same "
        "fact as the GOLD answer, allowing different wording, formatting or extra "
        "detail. `abstained` is true when the ANSWER declines to answer or says it "
        "does not know. An empty GOLD means the question is unanswerable: then "
        "`correct` is true only if the answer abstained."
    ),
    # LoCoMo-Refined (mem-eval-suite/LoCoMo_refined src/llm_judge.py): the strictest public
    # ruler, under which the best published system scores 82.65 and Mem0's platform 48.91.
    "refined": (
        "You grade one answer against a GOLD answer. Mark `correct` true only if the ANSWER "
        "clearly includes the GOLD's key content and does not contradict it. Time granularity "
        "must match exactly (hour with hour, day with day, month with month, year with year); "
        "do not treat a relative and an absolute time expression as equivalent; harmless "
        "modifiers like last/previous are fine. When GOLD lists several distinct facts, ALL "
        "must be covered, and extra non-contradictory items generally make it wrong. For a "
        "preference question, any one reason from GOLD's list suffices. `abstained` is true "
        "when the ANSWER declines to answer or says it does not know. An empty GOLD means the "
        "question is unanswerable: then `correct` is true only if the answer abstained."
    ),
    "lenient": (
        "You grade one answer against a GOLD answer. Mark `correct` true if the ANSWER "
        "includes AT LEAST ONE correct item from the GOLD answer (when GOLD lists several "
        "things, one is enough). Paraphrases count. Extra detail is fine. Dates within 14 "
        "days of each other are correct. Mark `correct` false ONLY if zero correct items "
        "appear or the ANSWER addresses a completely different topic. `abstained` is true "
        "when the ANSWER declines to answer or says it does not know. An empty GOLD means "
        "the question is unanswerable: then `correct` is true only if the answer abstained."
    ),
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "abstained": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "abstained"],
}

_ABSTAIN = re.compile(
    r"\b(i don'?t know|cannot|can not|no information|not (?:in|mentioned))\b", re.I
)


class _Pacer:
    """Keeps a judged run under the gateway's quota.

    A free Gemini key allows roughly ten calls a minute; a judged question costs two, so an
    unpaced run fails from the seventh question onward and the result is a list of 429s rather
    than a measurement. Retrying does not help — the quota is per minute, not per request —
    so the harness has to not exceed it in the first place.
    """

    #: Never crawl below this. Halving on every rate limit is right for a quota that is
    #: merely tight and wrong for one that is exhausted: four rate limits took a run from 14
    #: calls a minute to 0.4, where it would have needed six hours to finish and produced
    #: nothing usable in the meantime. Below this floor the run stops and says why, which is
    #: information; crawling is not.
    FLOOR_CALLS_PER_MINUTE = 3.0

    def __init__(self, calls_per_minute: float) -> None:
        self.interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0.0
        self._next = 0.0
        self.rate_limits = 0

    async def wait(self) -> None:
        if not self.interval:
            return
        now = time.monotonic()
        if now < self._next:
            await asyncio.sleep(self._next - now)
        self._next = max(now, self._next) + self.interval

    async def rate_limited(self) -> None:
        """A quota was hit anyway: wait out the window and pace more slowly from here.

        Two things made a fixed rate insufficient. The provider's limit is counted in *wire*
        requests, and a client configured with retries sends up to three of those per logical
        call — so pacing at half the documented limit still exceeded it. And the retry delay
        does not arrive in a `Retry-After` header: Gemini puts it in the error body ("Please
        retry in 59.18s"), where a client honouring the header sees nothing and falls back to
        a backoff measured in milliseconds against a window measured in a minute.
        """
        self.rate_limits += 1
        self.interval *= 2
        if 60.0 / self.interval < self.FLOOR_CALLS_PER_MINUTE:
            raise SystemExit(
                f"the gateway rate-limited this run {self.rate_limits} times and backing off "
                f"has reached {60.0 / self.interval:.1f} calls/minute, which cannot finish. "
                "The quota is exhausted rather than tight — wait for it to reset, raise it, "
                "or run with a smaller --sample."
            )
        await asyncio.sleep(60.0)
        self._next = time.monotonic() + self.interval


async def _answer(llm, bundle_text: str, question: str, *, reference_date: str = "") -> str:
    """One answer, grounded only in the bundle. ``reference_date`` is the last session's
    date, so "last week" and "two years ago" resolve against the conversation, not today."""
    from memory_service.ports.models import LLMMessage

    user = f"CONTEXT:\n{bundle_text}\n\n"
    if reference_date:
        user += f"REFERENCE DATE (the conversation's last session): {reference_date}\n\n"
    user += f"QUESTION: {question}"
    completion = await llm.complete(
        [
            LLMMessage(role="system", content=ANSWER_SYSTEM),
            LLMMessage(role="user", content=user),
        ],
        # Generous on purpose: a model that reasons before it answers spends the output
        # budget on thinking first and returns an empty string when it runs out. Measured
        # here: 16 of 304 answers came back empty at 1024 against deepseek-flash.
        max_tokens=16384,
        use="grounding_judge",
    )
    return (completion.text or "").strip()


async def _judge(llm, question: str, gold: str, got: str, *, ruler: str = "strict") -> dict:
    """Whether the produced answer matches the gold one, and whether it abstained.

    ``abstained`` is asked of the judge rather than pattern-matched because a model has many
    ways to decline. The regex below is only a fallback for when the judge itself fails.
    ``ruler`` picks the grading rules; see JUDGE_RULERS.
    """
    from memory_service.ports.models import LLMMessage

    messages = [
        LLMMessage(role="system", content=JUDGE_RULERS[ruler]),
        LLMMessage(
            role="user",
            content=f"QUESTION: {question}\nGOLD: {gold or '(unanswerable)'}\nANSWER: {got}",
        ),
    ]
    # A reasoning model sometimes spends the whole budget thinking and emits nothing, even on
    # a prompt this small: 19 of 233 answerable rows in v6, every one of them a row whose
    # answer had already been retrieved and generated, and every one scored wrong for it. The
    # loop is stochastic, so asking again clears it. The retry belongs here because a judged
    # run sets max_retries=0 on the gateway, and because only an empty completion is worth
    # repeating - a 429 is the pacer's to handle, and anything else is a real failure.
    last: Exception | None = None
    for attempt in range(3):
        try:
            return await llm.structured(
                messages, schema=JUDGE_SCHEMA, max_tokens=16384, use="grounding_judge"
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it is the empty one
            if "no content" not in str(exc):
                raise
            last = exc
            await asyncio.sleep(0.5 * (attempt + 1))
    raise last if last else RuntimeError("unreachable")


async def run(
    conversations: int | None,
    limit_questions: int | None,
    k: int,
    *,
    sample: int | None = None,
    ablate: dict[str, bool] | None = None,
    judge: bool = False,
    calls_per_minute: float = 10.0,
    ruler: str = "strict",
) -> dict:
    if not DATASET.is_file():
        raise SystemExit(f"{DATASET} is missing — run `make bench-locomo-prepare`")
    data = (
        json.loads(DATASET.read_text())[:conversations]
        if conversations
        else json.loads(DATASET.read_text())
    )

    settings = _settings()
    overrides = bench_overrides()
    if ablate:
        # An ablation answers "is this component earning its cost?" the only way that means
        # anything: turn it off, measure the same questions, compare. The reranker is ~87% of
        # per-request model cost, so its ablation is the one that decides real money.
        overrides = replace(
            overrides, retrieval=bench_retrieval(overrides).model_copy(update=ablate)
        )
    container = await build_container(settings, __version__, overrides=overrides)
    llm = container.llm if judge else None
    pacer = _Pacer(calls_per_minute)
    if judge and not getattr(llm, "enabled", False):
        raise SystemExit(
            "--judge needs a generative model: set models.llm.enabled=true, "
            "models.llm.enabled=true and a models.llm.model the gateway serves."
        )
    per_category: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "hit": 0})
    records: list[dict] = []
    latencies: list[float] = []
    incidents: list[str] = []
    aborted: str | None = None
    started = time.perf_counter()
    try:
        for index, conversation in enumerate(data):
            await reset_store(container, TENANT)
            register_handlers(container)
            ctx = MemoryExecutionContext(
                tenant_id=TENANT, user_id=f"locomo-{index}", workspace_id="ws"
            )
            # The turns are written under each speaker's own user id with TENANT visibility,
            # so the questioner reads them by being in the tenant and nothing has to be
            # granted. This used to be a WORKSPACE ingest, which the questioner could only
            # read through workspace membership - a key it carried only once granted, and
            # without the grant every search matched nothing and a whole judged run scored
            # 0.0 on 304 questions while reporting INSUFFICIENT evidence. WORKSPACE is no
            # longer an audience; the grant that made it work went with it.
            turns = await _ingest_conversation(container, ctx, conversation["conversation"])
            sessions = _sessions(conversation["conversation"])
            reference_date = (
                conversation["conversation"].get(f"{sessions[-1][0]}_date_time", "")
                if sessions
                else ""
            )

            questions = conversation.get("qa", [])
            if limit_questions:
                questions = questions[:limit_questions]
            if sample:
                questions = _stratified(questions, sample)
            builder = container.services["context_builder"]
            # A run of this costs hours on CPU. Without a progress line it is indistinguishable
            # from a hang — an earlier ablation sat silent for five hours and the only way to
            # tell it was alive was to read /proc from inside the container. Progress goes to
            # stderr so stdout stays a clean JSON document.
            total_q = len(questions)
            print(
                f"[locomo] conversation {index + 1}/{len(data)}: {total_q} questions",
                file=sys.stderr,
                flush=True,
            )
            for asked, item in enumerate(questions, start=1):
                category = int(item.get("category", 0))
                question = item.get("question") or ""
                if not question:
                    continue
                bundle, latency_ms = await _build_timed(builder, ctx, question, incidents)
                latencies.append(latency_ms)
                if asked % 10 == 0 or asked == total_q:
                    mean = sum(latencies) / len(latencies) / 1000
                    left = (total_q - asked) * mean
                    print(
                        f"[locomo] conv {index + 1} {asked}/{total_q} "
                        f"mean={mean:.1f}s eta={left / 60:.0f}m",
                        file=sys.stderr,
                        flush=True,
                    )
                bucket = per_category[category]
                bucket["n"] += 1
                # `evidence.status` lives on the bundle, not on a raw retrieval result —
                # reading it off the wrong object scored abstention 0.0 by construction.
                # `str(...)` above already resolved the enum, so the value is a string here
                status = str(getattr(getattr(bundle, "evidence", None), "status", "") or "").upper()

                rendered = bundle.render() or ""
                found = _content_tokens(rendered)
                answer = item.get("answer") or ""
                answer_overlap = _overlap(_content_tokens(answer), found)
                # The supporting turns LoCoMo annotates, scored the same way. This is the
                # phrasing-independent question - "did the right source surface?" - and it is
                # the one that does not move when the pipeline rewrites what it stored.
                overlaps = [
                    _overlap(_content_tokens(turns.get(e, "")), found) for e in _evidence_ids(item)
                ]
                evidence_overlap = max(overlaps, default=0.0)
                # The max says "at least one source surfaced". A multi-hop question needs
                # every one of them (40 of 43 in the first two conversations cite two or
                # more turns), so the minimum is the recall that actually bounds it.
                evidence_min_overlap = min(overlaps, default=0.0)
                # ...and the same question asked of the ORDER rather than the union, which is
                # the only thing a reordering can move (see _evidence_ranks).
                evidence_ranks = _evidence_ranks(
                    bundle.memories, [turns.get(e, "") for e in _evidence_ids(item)]
                )
                rank_metrics = _rank_metrics(
                    evidence_ranks, _most_relevant_count(len(bundle.memories))
                )
                rank_metrics["evidence_reconstructed"] = _evidence_reconstructed(
                    bundle.memories, [turns.get(e, "") for e in _evidence_ids(item)]
                )
                evidence_ranks_by_arm = _evidence_ranks_by_arm(
                    bundle.memories, [turns.get(e, "") for e in _evidence_ids(item)]
                )
                strict = _answer_present(rendered, answer)
                abstained = "INSUFFICIENT" in status

                judged: dict | None = None
                if llm is not None:
                    produced: str | None = None
                    try:
                        await pacer.wait()
                        produced = await _answer(
                            llm, rendered, question, reference_date=reference_date
                        )
                        await pacer.wait()
                        verdict = await _judge(llm, question, answer, produced, ruler=ruler)
                    except Exception as exc:  # noqa: BLE001 - a judged run must say it failed
                        # the gateway's own message lives in `details`; without it every
                        # failure reads "returned 400" and says nothing about why
                        detail = str(getattr(exc, "details", ""))[:400]
                        judged = {
                            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                            "detail": detail,
                        }
                        # The row still scores WRONG - a judged run may not fall back to the
                        # heuristic - but the answer cost a retrieval and a generation, and
                        # throwing it away made the failure permanent: locomo_rescore rebuilds
                        # a verdict from `produced`, so without it the only repair is re-running
                        # the whole question. Keeping it turns a lost row into one judge call.
                        if produced is not None:
                            judged["produced"] = produced[:400]
                        if "429" in detail or "429" in str(exc):
                            print(
                                f"[locomo] rate limited; slowing to "
                                f"{60.0 / (pacer.interval * 2):.1f} calls/min",
                                file=sys.stderr,
                                flush=True,
                            )
                            await pacer.rate_limited()
                    else:
                        judged = {
                            "produced": produced[:400],
                            "correct": bool(verdict.get("correct")),
                            # the schema asks for it; keeping it makes a disputed verdict
                            # auditable without re-running the judge
                            "reason": (verdict.get("reason") or "")[:300] or None,
                            # the judge is asked directly; the regex is only a fallback for a
                            # judge that answered the first question but not the second
                            "abstained": bool(
                                verdict.get("abstained", _ABSTAIN.search(produced) is not None)
                            ),
                        }

                # adversarial questions have no answer: abstaining *is* the correct answer
                if judged and "error" not in judged:
                    # A judged run scores what the caller would actually receive, which is the
                    # only place the adversarial category means anything.
                    hit = judged_hit(CATEGORY_NAMES.get(category, str(category)), judged)
                elif judge:
                    # A judged run whose judge failed on this row scores the row WRONG. It used
                    # to fall through to the token-overlap heuristic, so a run with a broken
                    # judge could still post hits - which is exactly how 79/79 failures once
                    # read as 0.197. Failures stay counted in n and in judge_failures.
                    hit = False
                else:
                    hit = (
                        abstained
                        if category == ADVERSARIAL
                        else answer_overlap >= ANSWER_OVERLAP_HIT
                    )
                if hit:
                    bucket["hit"] += 1
                # Persisted so the score can be re-derived - a different threshold, a stricter
                # rule, an LLM judge later - without paying for the run again. Every number
                # this harness has reported so far was wrong because of how it was scored, not
                # because of what was retrieved; keeping the raw material makes that
                # recoverable instead of a re-run.
                records.append(
                    {
                        "conversation": index,
                        "category": CATEGORY_NAMES.get(category, str(category)),
                        "question": question,
                        "answer": answer,
                        "answer_overlap": answer_overlap,
                        "evidence_overlap": evidence_overlap,
                        "evidence_hit": evidence_overlap >= EVIDENCE_OVERLAP_HIT,
                        "evidence_min_overlap": evidence_min_overlap,
                        "evidence_all_hit": bool(overlaps)
                        and evidence_min_overlap >= EVIDENCE_OVERLAP_HIT,
                        # The FULL per-turn overlap list, not just its max and min. Without
                        # it ``evidence_all_recall`` (question-level: every turn clears the
                        # bar) cannot be compared against ``evidence_reconstructed``
                        # (item-level: the fraction of turns that clear it), and the two were
                        # set side by side once already as though they measured the same
                        # thing. Keeping the list makes either level derivable afterwards.
                        # What the ROUTER decided this question was, beside what LoCoMo
                        # says it is. These are different claims - ``category`` is ground
                        # truth about the question, ``query_type`` is our classification of
                        # it - and the confusion matrix between them has never been taken on
                        # the full set, because this field was computed at retrieval time
                        # (engine sets diagnostics["query_type"]) and then dropped here. It
                        # decides whether a per-type execution plan is worth building: a
                        # deep path routed to 6% of the multi-hop questions is a deep path
                        # that does almost nothing, and the widened patterns in router.py
                        # were never re-measured after the rates in their own comments.
                        "query_type": bundle.diagnostics.get("query_type"),
                        "query_signals": bundle.diagnostics.get("signals"),
                        "evidence_overlaps": [round(o, 4) for o in overlaps],
                        "evidence_ranks": evidence_ranks,
                        "evidence_ranks_by_arm": evidence_ranks_by_arm,
                        **rank_metrics,
                        # per question, so p99 and a stage split can be derived from the
                        # file instead of from a summary computed once and never checkable
                        "latency_ms": round(latency_ms, 1),
                        "timings_ms": bundle.diagnostics.get("timings_ms"),
                        "strict_substring": strict,
                        "status": status,
                        "abstained": abstained,
                        # ContextBundle has no `chunks`: it carries memories, knowledge,
                        # graph facts and summaries separately, and LoCoMo goes in as
                        # observations, so everything lands in `memories`. The old field read
                        # 0 on every row of a run that was retrieving fine.
                        "bundle": {
                            "memories": len(bundle.memories),
                            "knowledge": len(bundle.knowledge),
                            "graph_facts": len(bundle.graph_facts),
                            "summaries": len(bundle.summaries),
                        },
                        "rendered_chars": len(rendered),
                        "judged": judged,
                        "judge_ruler": ruler if judged else None,
                        "hit": hit,
                    }
                )
    except Exception as exc:  # noqa: BLE001 - keep what was measured
        # A store that was unreachable for one call once took a 33-minute run with it: the
        # exception left run(), nothing was written, and 300 judged answers were lost. The
        # partial result is written with the failure named in it; the exit code says so.
        aborted = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(f"[locomo] ABORTED after {len(records)} questions: {aborted}", file=sys.stderr)
    finally:
        await container.close()

    latencies.sort()
    answerable = {c: v for c, v in per_category.items() if c != ADVERSARIAL}
    total_n = sum(v["n"] for v in answerable.values())
    total_hit = sum(v["hit"] for v in answerable.values())
    embedding = "hash" if overrides.embedding == "hash" else FROZEN_MODELS.dense.id

    # Abstention is the headline claim of this product, and this harness cannot measure it
    # in this configuration. The NLI cascade that decides "the evidence does not support this"
    # scores *claims*, which only exist once something has generated an answer; with
    # `models.llm.enabled=false` it never runs. What is left is the retrieval-level gate in
    # VerificationStage, which abstains only when nothing retrieved shares a content term with
    # the question — and LoCoMo's adversarial questions are deliberately about the same people
    # and topics as the conversation, so they always share terms. The gate cannot fire on
    # them by construction. A 0.0 here is a statement about the setup, not about the system,
    # and reporting it as a score would be the third time this harness published its own
    # artifact as a measurement.
    caveats: list[str] = []
    if aborted:
        caveats.append(
            f"run aborted after {len(records)} questions ({aborted}); every score is over the "
            "questions answered before the failure and the run must be repeated."
        )
    if incidents:
        caveats.append(
            f"{len(incidents)} transient store failures were retried; per-question latency "
            "records the successful attempt only."
        )
    # A judged run only measures abstention if the judge actually answered. The first one did
    # not: every call failed (rate limit, then an open circuit) and every score silently fell
    # back to token overlap — while the result still reported `abstention_measurable: true`
    # and an abstention rate of 0.0, which is precisely the false confidence this file exists
    # to avoid.
    judged_rows = [r for r in records if r.get("judged")]
    judged_ok = [r for r in judged_rows if "error" not in (r["judged"] or {})]
    llm_on = bool(getattr(settings.models.llm, "enabled", False)) and bool(judged_ok)
    if judge and len(judged_ok) < len(judged_rows):
        caveats.append(
            f"{len(judged_rows) - len(judged_ok)} of {len(judged_rows)} judged calls failed; "
            "those questions fell back to token overlap. Scores are a mixture of two "
            "different metrics and should not be compared with a clean run."
        )
    nli_representative = overrides.nli is None
    if not llm_on:
        caveats.append(
            "abstention_rate_on_adversarial is NOT a measurement: the NLI cascade scores "
            "generated claims and models.llm.enabled=false, so only the lexical overlap gate "
            "can abstain, and LoCoMo adversarial questions always share terms with the "
            "conversation. Re-run with generation enabled to score category 5."
        )
    if not nli_representative:
        caveats.append(
            f"nli={overrides.nli!r} is a stand-in, not the "
            "entailment model; grounding results are not representative."
        )
    if embedding == "hash":
        caveats.append("embedding='hash' is a stand-in; recall is not real.")

    return {
        "dataset": {
            "name": "LoCoMo (snap-research/locomo)",
            "conversations": len(data),
            "questions": sum(v["n"] for v in per_category.values()),
        },
        # the depth that was actually used; `--k` only names the file, retrieval reads
        # settings, and a run that said "depth 50" once carried 40 memories in every bundle
        "k": container.tuning.retrieval.final_k,
        "k_requested": k,
        "settings": {
            "retrieval": container.tuning.retrieval.model_dump(mode="json"),
            "context": container.tuning.context.model_dump(mode="json"),
            "llm_model": settings.models.llm.model,
        },
        "answer_recall_at_k": round(total_hit / total_n, 4) if total_n else 0.0,
        "caveats": caveats,
        "abstention_measurable": llm_on,
        "abstention_rate_on_adversarial": (
            round(per_category[ADVERSARIAL]["hit"] / per_category[ADVERSARIAL]["n"], 4)
            if per_category[ADVERSARIAL]["n"]
            else None
        ),
        "by_category": {
            CATEGORY_NAMES.get(c, str(c)): {
                "n": v["n"],
                "score": round(v["hit"] / v["n"], 4) if v["n"] else 0.0,
            }
            for c, v in sorted(per_category.items())
        },
        # Secondary metrics, on the same questions. `evidence_recall` is the one that does
        # not depend on how the gold answer happens to be phrased; `strict_recall` is the
        # old contiguous-substring rule, reported so its gap to the headline number stays
        # visible rather than being quietly swapped out.
        "judged": judge,
        "judge_failures": sum(
            1 for r in records if r.get("judged") and "error" in (r["judged"] or {})
        ),
        "evidence_recall": (
            round(
                sum(1 for r in records if r["category"] != "adversarial" and r["evidence_hit"])
                / total_n,
                4,
            )
            if total_n
            else 0.0
        ),
        "evidence_all_recall": (
            round(
                sum(1 for r in records if r["category"] != "adversarial" and r["evidence_all_hit"])
                / total_n,
                4,
            )
            if total_n
            else 0.0
        ),
        # The order-aware companions to evidence_all_recall, over the same answerable rows.
        # That number has been saturated at 0.9785 across every ablation taken, which is what
        # a union metric does once retrieval works; these two can still move when nothing but
        # the ranking changes, and are the instrument any promotion/reordering arm is read on.
        **_rank_summary(records),
        "strict_recall": (
            round(
                sum(1 for r in records if r["category"] != "adversarial" and r["strict_substring"])
                / total_n,
                4,
            )
            if total_n
            else 0.0
        ),
        "scoring": {
            "answer_overlap_hit": ANSWER_OVERLAP_HIT,
            "evidence_overlap_hit": EVIDENCE_OVERLAP_HIT,
            "note": (
                (
                    "answers were generated from each bundle and graded by a model, the way "
                    "LoCoMo is scored; the overlap figures below are kept as a second opinion."
                )
                if judge
                else (
                    "generation is disabled, so answers are not judged - these are "
                    "token-recall proxies over the retrieved bundle."
                )
            )
            + " Per-question detail is in `records`; rescore from it rather than re-running.",
        },
        "records": records,
        "aborted": aborted,
        "store_incidents": incidents,
        "query_p50_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0.0,
        "query_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0.0,
        "query_p99_ms": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))], 1)
        if latencies
        else 0.0,
        "total_seconds": round(time.perf_counter() - started, 1),
        "ablation": ablate or {},
        "embedding_provider": embedding,
        "representative": embedding != "hash",
        "provenance": provenance(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=int, default=None)
    parser.add_argument("--questions", type=int, default=None, help="cap per conversation")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="seeded per-category sample of this many questions per conversation",
    )
    parser.add_argument(
        "--off",
        nargs="*",
        default=[],
        metavar="FLAG",
        help="retrieval flags to disable for this run, e.g. --off rerank graph",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="generate an answer from each bundle and grade it with a model (LoCoMo's own "
        "method); makes the adversarial category measurable. Needs models.llm configured.",
    )
    parser.add_argument(
        "--calls-per-minute",
        type=float,
        default=120.0,
        help=(
            "gateway call budget for --judge. A free Gemini key allowed about ten; "
            "DeepSeek limits by concurrency rather than a daily quota, so 120 is safe and "
            "turns a multi-hour judged run into a bounded one."
        ),
    )
    parser.add_argument(
        "--judge-ruler",
        choices=sorted(JUDGE_RULERS),
        default="strict",
        help="grading rules for --judge: strict (ours) or lenient (Mem0's published ruler)",
    )
    parser.add_argument("--out", default="locomo.json", help="result filename")
    args = parser.parse_args()
    ablate = dict.fromkeys(args.off, False)
    result = asyncio.run(
        run(
            args.conversations,
            args.questions,
            args.k,
            sample=args.sample,
            ablate=ablate,
            judge=args.judge,
            calls_per_minute=args.calls_per_minute,
            ruler=args.judge_ruler,
        )
    )
    write_result(args.out, result)
    if result.get("aborted"):
        sys.stderr.write(f"[locomo] result is PARTIAL: {result['aborted']}\n")
    # `records` is hundreds of rows; it belongs in the file, not on the terminal.
    summary = {k: v for k, v in result.items() if k not in ("provenance", "records")}
    sys.stdout.write(json.dumps(summary, indent=2))
    sys.stdout.write("\n")
    return 1 if result.get("aborted") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
