"""GroundingCascade: per-claim verification of an answer against evidence.

    decompose -> validate citations -> NLI support -> judge (borderline band only)
              -> contradiction scan over retrieved-but-unused evidence -> GroundingReport

Deterministic first: sentences/clauses are split by rules, citation markers (``[1]``,
``[chunk_id:...]``, ``(source: ...)``) must resolve to the evidence they point at and that
evidence must mention the claim, and every claim is scored by the NLI provider. The LLM is
consulted through ``LLMAssist`` only for claims whose entailment lies in the borderline band
and only when ``grounding_judge`` is enabled; when it cannot help the claim stays
``borderline``. Claims that are not supported and collide with evidence the retriever found
but the bundle did not pack are reported as ``contradicted``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from memory_service.config.settings import NLISettings
from memory_service.domain.context_bundle import ContextBundle
from memory_service.domain.grounding import ClaimReport, ClaimVerdict, GroundingReport
from memory_service.modules.grounding.lexical import content_tokens, coverage, words
from memory_service.modules.llm.assist import LLMAssist
from memory_service.observability.metrics import grounding_claims_total, stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.models import NLIProvider, NLIScore

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r";\s+|,\s+(?:and|but|while|whereas)\s+", re.IGNORECASE)
_BRACKET_CITE = re.compile(r"\s*\[([^\[\]]{1,120})\]")
_SOURCE_CITE = re.compile(r"\s*\((?:source|src|ref|see)\s*:?\s*([^()]{1,120})\)", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^(?:[-•*>]+|\d{1,2}[.)]|\(\d{1,2}\))\s+")
_HEDGE = re.compile(
    r"^(?:maybe|perhaps|possibly|probably|i think|i believe|i guess|i suppose|it seems|"
    r"it might|it may|it could|i'?m not sure|i am not sure|not sure)\b",
    re.IGNORECASE,
)
_DISCOURSE = re.compile(
    r"^(?:in summary|in short|to summari[sz]e|overall|here is|here's|here are|let me know|"
    r"i hope this helps|sure|certainly|of course|as requested|based on the (?:evidence|context))\b",
    re.IGNORECASE,
)
_ID_LIKE = re.compile(
    r"^(?:\d{1,3}|[a-z_]+:.+|(?:chk|mem|sum|rel|wm|fact|doc)_\S+)$", re.IGNORECASE
)

HEDGE_MAX_WORDS = 8
DISCOURSE_MAX_WORDS = 8
CLAUSE_MIN_TOKENS = 4
CLAIM_MIN_TOKENS = 3

_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
    "additionalProperties": False,
}
_JUDGE_SYSTEM = (
    "You judge whether a claim is fully supported by evidence passages. Supported means every "
    "fact in the claim (entities, numbers, direction of change, negation, dates) is stated in "
    "or follows directly from the evidence; anything the evidence does not say is unsupported. "
    'Use only the evidence. Return JSON only: {"supported": true|false, "reason": "..."}.'
)
_JUDGE_PASSAGE_CHARS = 1200


@dataclass(frozen=True)
class Evidence:
    item_id: str
    text: str
    kind: str = "chunk"
    citation: str = ""


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[str, ...] = ()


# ------------------------------------------------------------------ decomposition


def _strip_citations(sentence: str) -> tuple[str, list[str]]:
    cites: list[str] = []

    def take(match: re.Match[str]) -> str:
        cites.extend(p.strip() for p in match.group(1).split(",") if p.strip())
        return " "

    text = _BRACKET_CITE.sub(take, sentence)
    text = _SOURCE_CITE.sub(take, text)
    return " ".join(text.split()).strip(" ,;"), cites


# ``kind`` is a Representation value for packed evidence (CHUNK, TABLE, RELATION, SUMMARY...)
# and a record kind for unused evidence. Higher is a better thing to cite. This is the closed
# set /v1/verify accepts: every kind the service itself emits in a bundle is here, so a
# bundle's evidence_items() and unused list always round-trip. MEMORY (the packed
# representation of a memory) and fact (the record kind of an unused graph fact) were never
# ranked, which the 0 keeps; they are listed so the contract names them.
EvidenceKind = Literal[
    "CHUNK",
    "TABLE",
    "PARAGRAPH",
    "SECTION",
    "SUBSECTION",
    "CODE_BLOCK",
    "chunk",
    "SUMMARY",
    "summary",
    "ENTITY",
    "RELATION",
    "MEMORY",
    "memory",
    "fact",
]
_SOURCE_RANK: dict[str, int] = {
    "CHUNK": 3,
    "TABLE": 3,
    "PARAGRAPH": 3,
    "SECTION": 3,
    "SUBSECTION": 3,
    "CODE_BLOCK": 3,
    "chunk": 3,
    "SUMMARY": 2,
    "summary": 2,
    "ENTITY": 1,
    "RELATION": 1,
    "memory": 1,
    "MEMORY": 0,
    "fact": 0,
}


def _source_rank(evidence: Evidence) -> int:
    """A verifiable passage beats a derived fact: a chunk carries a document, page and offsets
    the reader can check, while a graph relation distilled from that chunk carries none."""
    return _SOURCE_RANK.get(evidence.kind, 0)


_DOT_SENTINEL = "\x00dot\x00"


def _protect_citations(answer: str) -> str:
    """Hide full stops inside citation markers from the sentence splitter."""

    def hide(match: re.Match[str]) -> str:
        return match.group(0).replace(".", _DOT_SENTINEL)

    return _SOURCE_CITE.sub(hide, _BRACKET_CITE.sub(hide, answer))


def _restore_citations(text: str) -> str:
    return text.replace(_DOT_SENTINEL, ".")


def _assertive(text: str) -> bool:
    if not text or text.endswith("?"):
        return False
    count = len(words(text))
    if len(content_tokens(text)) < CLAIM_MIN_TOKENS:
        return False
    if _HEDGE.match(text) and count < HEDGE_MAX_WORDS:
        return False
    return not (_DISCOURSE.match(text) and (count < DISCOURSE_MAX_WORDS or text.endswith(":")))


def _clauses(sentence: str) -> list[str]:
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(sentence) if p.strip()]
    if len(parts) > 1 and all(len(content_tokens(p)) >= CLAUSE_MIN_TOKENS for p in parts):
        return parts
    return [sentence]


def decompose(answer: str, *, max_claims: int = 40) -> list[Claim]:
    """Deterministic sentence/clause split; questions, short hedges and discourse-only
    fragments are dropped; citation markers are lifted off the claim text.

    Citations are lifted *before* the sentence split, because a citation can contain a full
    stop ("(source: annual report p. 11)") that would otherwise split the sentence through the
    middle of the marker — leaving its words in the claim, where they count as uncovered
    content and push a well-supported claim down into the borderline band.
    """
    out: list[Claim] = []
    for raw in _SENTENCE_SPLIT.split(_protect_citations(answer)):
        sentence = _LIST_MARKER.sub("", raw.strip())
        if not sentence:
            continue
        text, cites = _strip_citations(_restore_citations(sentence))
        for raw_clause in _clauses(text):
            clause = raw_clause.rstrip(".!").strip()
            if _assertive(clause):
                out.append(Claim(text=clause, citations=tuple(cites)))
            if len(out) >= max_claims:
                return out
    return out


# ------------------------------------------------------------------ evidence


def bundle_evidence(bundle: ContextBundle) -> tuple[list[Evidence], list[Evidence]]:
    """(packed, unused) evidence of a bundle; ordinal citations (``[1]``) count through the
    packed list in this order: memories, facts, summaries, knowledge."""
    packed = [
        Evidence(item_id=i.item_id, text=i.text, kind=i.representation.value, citation=i.citation)
        for group in (bundle.memories, bundle.graph_facts, bundle.summaries, bundle.knowledge)
        for i in group
    ]
    unused = [Evidence(item_id=u.item_id, text=u.text, kind=u.kind) for u in bundle.evidence.unused]
    return packed, unused


def resolve_citation(cite: str, evidence: Sequence[Evidence]) -> Evidence | None:
    key = cite.strip()
    if key.isdigit():
        index = int(key) - 1
        return evidence[index] if 0 <= index < len(evidence) else None
    tail = key.split(":")[-1].strip()
    for e in evidence:
        if key in (e.item_id, e.citation) or tail == e.item_id:
            return e
    return None


def attach(bundle: ContextBundle, report: GroundingReport) -> ContextBundle:
    return bundle.model_copy(
        update={"evidence": bundle.evidence.model_copy(update={"grounding": report})}
    )


# ------------------------------------------------------------------ cascade


class GroundingCascade:
    def __init__(
        self,
        nli: NLIProvider,
        *,
        settings: NLISettings,
        assist: LLMAssist | None = None,
    ) -> None:
        self.nli = nli
        self.cfg = settings
        self.assist = assist or LLMAssist.disabled()

    async def verify_bundle(self, bundle: ContextBundle, answer: str) -> GroundingReport:
        packed, unused = bundle_evidence(bundle)
        return await self.verify(answer, packed, unused=unused)

    async def verify(
        self,
        answer: str,
        evidence: Sequence[Evidence],
        *,
        unused: Sequence[Evidence] = (),
    ) -> GroundingReport:
        tokens_before = self.assist.tokens_used()
        claims = decompose(answer, max_claims=self.cfg.max_claims)
        with (
            span("grounding.verify", claims=len(claims), evidence=len(evidence)),
            stage_seconds.labels("grounding.verify").time(),
        ):
            reports: list[ClaimReport] = []
            judged = 0
            for claim in claims:
                report = await self._verify_claim(claim, evidence)
                if report.method == "judge":
                    judged += 1
                report = await self._scan_unused(claim, report, unused)
                grounding_claims_total.labels(report.verdict).inc()
                reports.append(report)
        counts = {v: sum(1 for r in reports if r.verdict == v) for v in _VERDICTS}
        total = len(reports)
        notes: list[str] = []
        if not total:
            notes.append("no assertive claims found in the answer")
        if not self.nli.representative:
            notes.append(f"{self.nli.info.name} is a deterministic stand-in; not representative")
        return GroundingReport(
            claims=reports,
            supported=counts["supported"],
            unsupported=counts["unsupported"],
            contradicted=counts["contradicted"],
            borderline=counts["borderline"],
            per_claim_hallucination_rate=(
                round((counts["unsupported"] + counts["contradicted"]) / total, 4) if total else 0.0
            ),
            nli_provider=self.nli.fingerprint(),
            representative=self.nli.representative,
            judge_consulted=judged,
            llm_tokens=self.assist.tokens_used() - tokens_before,
            evidence_count=len(evidence),
            unused_count=len(unused),
            notes=notes,
        )

    # ---------------------------------------------------------------- per claim
    async def _verify_claim(self, claim: Claim, evidence: Sequence[Evidence]) -> ClaimReport:
        notes: list[str] = []
        premises: list[Evidence] = []
        for cite in claim.citations:
            found = resolve_citation(cite, evidence)
            if found is not None:
                if found not in premises:
                    premises.append(found)
                continue
            if _ID_LIKE.match(cite):
                return ClaimReport(
                    claim=claim.text,
                    verdict="unsupported",
                    citations=list(claim.citations),
                    method="citation",
                    notes=[f"citation [{cite}] does not resolve to any evidence item"],
                )
            notes.append(f"citation ({cite}) is not an evidence id; treated as uncited")
        cited = bool(premises)
        if premises:
            mismatched = [e.item_id for e in premises if coverage(claim.text, e.text) == 0.0]
            if mismatched:
                return ClaimReport(
                    claim=claim.text,
                    verdict="unsupported",
                    evidence_ids=mismatched,
                    citations=list(claim.citations),
                    method="citation",
                    notes=[*notes, "cited evidence does not mention the claim"],
                )
        else:
            premises = self._closest(claim.text, evidence)
        if not premises:
            return ClaimReport(
                claim=claim.text,
                verdict="unsupported",
                citations=list(claim.citations),
                notes=[*notes, "no evidence shares a content term with the claim"],
            )
        scores = await self.nli.entail([e.text for e in premises], claim.text)
        # On equal support, cite the primary source. A graph fact derived from a chunk renders
        # as the same sentence and ties with it, but only the chunk carries a document, page and
        # offsets the reader can check, so it is the more useful citation.
        best_e = max(
            range(len(scores)),
            key=lambda i: (scores[i].entailment, _source_rank(premises[i]), -i),
        )
        best_c = max(
            range(len(scores)),
            key=lambda i: (scores[i].contradiction, _source_rank(premises[i]), -i),
        )
        support = scores[best_e].entailment
        contradiction = scores[best_c].contradiction
        threshold = self.cfg.supported_threshold
        low, high = self.cfg.borderline_band
        method = "nli"
        verdict: ClaimVerdict
        ids = [premises[best_e].item_id]
        # the best-supporting premise decides; a contradiction only wins when nothing supports
        # the claim above the band (a second passage with other figures is not a refutation)
        supported = support >= threshold and support > scores[best_e].contradiction
        if not supported and _contradicts(scores[best_c], threshold) and support <= high:
            verdict = "contradicted"
            ids = [premises[best_c].item_id]
        elif cited:
            # The claim named its source, so the question is only "does that item say this?" —
            # answerable from the premise itself. A middling score means the citation does not
            # carry the claim, which is a failed citation rather than something to hedge on or
            # spend a judge call arguing about.
            verdict = "supported" if supported else "unsupported"
            method = "citation"
            if not supported:
                notes.append("cited evidence does not support the claim")
        elif low <= support <= high:
            verdict = "borderline"
            decision = await self._judge(claim.text, premises)
            if decision is None:
                notes.append("entailment in the borderline band; no judge decision")
            else:
                verdict = "supported" if decision["supported"] else "unsupported"
                method = "judge"
                notes.append(f"judge: {decision['reason']}")
        elif support >= threshold:
            verdict = "supported"
        else:
            verdict = "unsupported"
        return ClaimReport(
            claim=claim.text,
            verdict=verdict,
            support=round(support, 4),
            contradiction=round(contradiction, 4),
            evidence_ids=ids,
            citations=list(claim.citations),
            method=method,
            notes=notes,
        )

    def _closest(self, claim: str, evidence: Sequence[Evidence]) -> list[Evidence]:
        ranked = sorted(
            ((coverage(claim, e.text), i, e) for i, e in enumerate(evidence)),
            key=lambda t: (-t[0], t[1]),
        )
        return [e for cov, _, e in ranked if cov > 0.0][: self.cfg.premises_per_claim]

    async def _judge(self, claim: str, premises: Sequence[Evidence]) -> dict[str, Any] | None:
        if not self.assist.wants("grounding_judge"):
            return None
        passages = "\n".join(
            f"[{i + 1}] {' '.join(e.text.split())[:_JUDGE_PASSAGE_CHARS]}"
            for i, e in enumerate(premises)
        )
        out = await self.assist.structured(
            "grounding_judge",
            system=_JUDGE_SYSTEM,
            user=f"Claim: {claim}\n\nEvidence:\n{passages}",
            schema=_JUDGE_SCHEMA,
            max_tokens=200,
        )
        if out is None:
            return None
        return {"supported": bool(out["supported"]), "reason": str(out.get("reason", ""))[:300]}

    async def _scan_unused(
        self, claim: Claim, report: ClaimReport, unused: Sequence[Evidence]
    ) -> ClaimReport:
        candidates = self._closest(claim.text, unused)
        if not candidates:
            return report
        scores = await self.nli.entail([e.text for e in candidates], claim.text)
        against = [
            e.item_id
            for e, s in zip(candidates, scores, strict=True)
            if _contradicts(s, self.cfg.supported_threshold)
        ]
        if not against:
            return report
        update: dict[str, Any] = {
            "contradicted_by": against,
            "notes": [*report.notes, f"contradicted by unused evidence {', '.join(against)}"],
        }
        if report.verdict in ("unsupported", "borderline"):
            update["verdict"] = "contradicted"
            update["contradiction"] = round(
                max(s.contradiction for s in scores if s.contradiction), 4
            )
        return report.model_copy(update=update)


_VERDICTS: tuple[ClaimVerdict, ...] = ("supported", "unsupported", "contradicted", "borderline")


def _contradicts(score: NLIScore, threshold: float) -> bool:
    return score.contradiction >= threshold and score.contradiction > score.entailment
