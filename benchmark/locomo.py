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
import random
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from benchmark.common import provenance, reset_store, write_result
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind, Visibility
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


def _speaker_ctx(ctx: MemoryExecutionContext, speaker: str) -> MemoryExecutionContext:
    """The ingest context for one speaker: same tenant and workspace, their own user id."""
    return ctx.model_copy(update={"user_id": speaker.strip().lower() or ctx.user_id})


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
            # the date is part of the record: LoCoMo's temporal questions depend on it
            content = f"[{when}] {speaker}: {body}"
            turns[dia_id] = content
            # Each speaker is their own user, so the fact "I moved from Sweden" gets the
            # subject user:Caroline rather than a shared id that makes Caroline's and
            # Melanie's facts indistinguishable - which is what the graph, the subject
            # check and the answerer all key on. WORKSPACE visibility is anchored on the
            # workspace, not the user, so the questioner (a third context in the same
            # workspace) still sees every turn. occurred_at carries the session date into
            # the memory's temporal fields instead of leaving it as ingest time.
            async with uow_factory() as uow:
                await memory.submit_observation(
                    uow,
                    _speaker_ctx(ctx, speaker),
                    kind=ObservationKind.MESSAGE,
                    content=content,
                    hints=ProcessingHints(visibility=Visibility.WORKSPACE),
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
    """LoCoMo stores evidence as the *string* ``"['D1:3', 'D1:5']"``, not as a list."""
    raw = item.get("evidence")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return _EVIDENCE_IDS.findall(str(raw or ""))


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
ANSWER_SYSTEM = (
    "You answer questions about a person's life using only the CONTEXT: dated memories "
    "from their conversations. Read EVERY entry from first to last before answering - do "
    "not stop at the first relevant one; the answer is often spread across several entries "
    "or has to be reasoned from them. For counting or listing questions, enumerate each "
    "distinct instance the context supports; do not estimate. Answer directly and "
    "specifically, keeping "
    "the names, dates, places and qualifiers the context gives - do not shorten them. "
    "When asked when, work out the actual date from the dated entries and state it. When "
    "asked what someone would likely do, feel or choose, infer it from what the context "
    "shows about them. Check the question's premise against the context: if it attributes "
    "something to the wrong person, or asks about an event the context never records, reply "
    "with exactly: I don't know. If the context ends with an Evidence status other than "
    "COMPLETE, answer only when the context explicitly states the fact for the person asked "
    "about; otherwise reply with exactly: I don't know. Never use outside knowledge. One "
    "short phrase or sentence."
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

    return await llm.structured(
        [
            LLMMessage(role="system", content=JUDGE_RULERS[ruler]),
            LLMMessage(
                role="user",
                content=f"QUESTION: {question}\nGOLD: {gold or '(unanswerable)'}\nANSWER: {got}",
            ),
        ],
        schema=JUDGE_SCHEMA,
        max_tokens=16384,
        use="grounding_judge",
    )


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
    if ablate:
        # An ablation answers "is this component earning its cost?" the only way that means
        # anything: turn it off, measure the same questions, compare. The reranker is ~87% of
        # per-request model cost, so its ablation is the one that decides real money.
        settings = settings.model_copy(
            update={"retrieval": settings.retrieval.model_copy(update=ablate)}
        )
    container = await build_container(settings, __version__)
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
    started = time.perf_counter()
    try:
        for index, conversation in enumerate(data):
            await reset_store(container, TENANT)
            register_handlers(container)
            ctx = MemoryExecutionContext(
                tenant_id=TENANT, user_id=f"locomo-{index}", workspace_id="ws"
            )
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
                call = time.perf_counter()
                bundle = await builder.build(ctx, question)
                latencies.append((time.perf_counter() - call) * 1000)
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
                status = str(getattr(getattr(bundle, "evidence", None), "status", "") or "")
                if hasattr(status, "value"):  # pragma: no cover - enum or str
                    status = status.value
                status = str(status).upper()

                rendered = bundle.render() or ""
                found = _content_tokens(rendered)
                answer = item.get("answer") or ""
                answer_overlap = _overlap(_content_tokens(answer), found)
                # The supporting turns LoCoMo annotates, scored the same way. This is the
                # phrasing-independent question - "did the right source surface?" - and it is
                # the one that does not move when the pipeline rewrites what it stored.
                evidence_overlap = max(
                    (
                        _overlap(_content_tokens(turns.get(e, "")), found)
                        for e in _evidence_ids(item)
                    ),
                    default=0.0,
                )
                strict = _answer_present(rendered, answer)
                abstained = "INSUFFICIENT" in status

                judged: dict | None = None
                if llm is not None:
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
                    hit = judged["abstained"] if category == ADVERSARIAL else judged["correct"]
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
    finally:
        await container.close()

    latencies.sort()
    answerable = {c: v for c, v in per_category.items() if c != ADVERSARIAL}
    total_n = sum(v["n"] for v in answerable.values())
    total_hit = sum(v["hit"] for v in answerable.values())
    embedding = settings.models.embedding.provider

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
    nli_representative = settings.models.nli.provider not in ("lexical", "hash")
    if not llm_on:
        caveats.append(
            "abstention_rate_on_adversarial is NOT a measurement: the NLI cascade scores "
            "generated claims and models.llm.enabled=false, so only the lexical overlap gate "
            "can abstain, and LoCoMo adversarial questions always share terms with the "
            "conversation. Re-run with generation enabled to score category 5."
        )
    if not nli_representative:
        caveats.append(
            f"models.nli.provider={settings.models.nli.provider!r} is a stand-in, not the "
            "entailment model; grounding results are not representative."
        )
    if embedding == "hash":
        caveats.append("models.embedding.provider='hash' is a stand-in; recall is not real.")

    return {
        "dataset": {
            "name": "LoCoMo (snap-research/locomo)",
            "conversations": len(data),
            "questions": sum(v["n"] for v in per_category.values()),
        },
        "k": k,
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
    # `records` is hundreds of rows; it belongs in the file, not on the terminal.
    summary = {k: v for k, v in result.items() if k not in ("provenance", "records")}
    sys.stdout.write(json.dumps(summary, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
