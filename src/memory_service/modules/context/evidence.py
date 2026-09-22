"""VerificationStage: evidence-group verification, escalation and honest abstention.

Required evidence groups are derived deterministically from the Document Context Graph of
the best-ranked chunks: a chunk that *uses a defined term* requires the definition, a chunk
with a *footnote marker* requires the footnote, a chunk that says "see Section 8" requires
that section. A group is satisfied when any candidate comes from the target node.

    verify -> escalate (fetch the missing companions directly, up to
    ``escalation_max_rounds``) -> verify again -> report

The report says COMPLETE, INCOMPLETE (some required companions could not be produced) or
INSUFFICIENT (no evidence at all, or nothing that shares a content term with the question —
the abstention case). The ContextBuilder re-checks the report against what actually fit in
the token budget, so a bundle never claims completeness for evidence it dropped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from memory_service.config.constants import RetrievalSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ContextGraphEdge, EvidenceStatus, QueryType
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.context.expansion import ExpansionStage, edges_to_groups
from memory_service.modules.memory.native import _STOP as STOP_WORDS
from memory_service.modules.retrieval.engine import Candidate
from memory_service.modules.retrieval.router import RoutedQuery
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.uow import UnitOfWorkFactory

#: Word characters in *any* script. The original pattern was ``[a-z][a-z0-9\-]+`` — ASCII
#: only — so ``content_terms`` returned the empty set for every Japanese, Chinese, Korean,
#: Greek, Cyrillic, Hebrew and Arabic query. Combined with the rule below (an empty term set
#: used to mean "overlaps everything") the abstention gate could not fire for those languages
#: at all: measured on the degenerate-input benchmark, a Japanese question against an
#: English-only corpus came back COMPLETE with ten items attached.
_WORD = re.compile(r"[^\W\d_][\w\-]*", re.UNICODE)

#: Scripts written without spaces between words. Tokenising those by "word" produces one
#: giant token that matches nothing but itself, so the standard treatment — Lucene's
#: CJKAnalyzer, and essentially every engine that indexes Japanese — is overlapping character
#: bigrams. Ranges: hiragana/katakana, CJK ideographs (incl. extension A and compatibility),
#: and hangul syllables.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+")
_REQUIRED_EDGES = (
    ContextGraphEdge.DEFINED_BY,
    ContextGraphEdge.FOOTNOTE,
    ContextGraphEdge.CROSS_REFERENCE,
)


def content_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = {w for w in _WORD.findall(lowered) if w not in STOP_WORDS and len(w) > 2}
    for run in _CJK.findall(lowered):
        # a lone ideograph is too common to carry signal; bigrams are the usual unit
        terms.update(run[i : i + 2] for i in range(len(run) - 1))
    return terms


class TermCache:
    """``content_terms`` per record id, for the life of one request.

    The rules below walk the same candidates several times over - ``overlaps`` once,
    ``unsupported_subject`` once per question name and again for the other-terms check - and
    each walk re-tokenised the full text of every item. On a bundle of 50 memories with a
    two-name question that is a few hundred regex passes over text that has not changed since
    the first one. The id is the key: two candidates with the same record id are the same
    record, and a candidate's text does not change within a request.
    """

    def __init__(self) -> None:
        self._terms: dict[str, set[str]] = {}

    def of(self, candidate: Candidate) -> set[str]:
        terms = self._terms.get(candidate.record_id)
        if terms is None:
            terms = content_terms(candidate.text)
            self._terms[candidate.record_id] = terms
        return terms


def overlaps(
    query: str, candidates: Sequence[Candidate], *, terms: TermCache | None = None
) -> bool:
    """Abstention rule: at least one evidence item shares a content term with the query."""
    cache = terms or TermCache()
    q = content_terms(query)
    if not q:
        # Nothing to ground: an empty query, bare punctuation, emoji, or nothing but
        # stopwords. This used to return ``bool(candidates)`` — vacuously "yes, it overlaps"
        # — so ``query=""`` was answered with a COMPLETE bundle carrying ten memories. A
        # question that asks nothing cannot be supported by evidence, and saying COMPLETE
        # about it is exactly the dishonesty the evidence report exists to prevent.
        return False
    return any(c.kind in ("chunk", "memory", "summary") and cache.of(c) & q for c in candidates)


def _conflicting_memories(candidates: Sequence[Candidate]) -> list[tuple[str, str]]:
    """Pairs of retrieved memories that are linked as contradicting (multi-agent conflicts
    the consolidator refused to resolve silently)."""
    ids = {c.record_id for c in candidates if c.kind == "memory"}
    out: list[tuple[str, str]] = []
    for c in candidates:
        if c.kind != "memory":
            continue
        for other in c.payload.get("contradicts", []) or []:
            if other in ids and (other, c.record_id) not in out:
                out.append((c.record_id, other))
    return out


_CAPITALISED = re.compile(r"\b([A-Z][a-z]{2,})(?:'s)?\b")
#: Interrogatives and auxiliaries are not content: "what" is not in the stop list, and a
#: memory containing it would have counted as supporting the question.
_FUNCTION_WORDS = frozenset(
    {
        "what",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "which",
        "why",
        "how",
        "did",
        "does",
        "do",
        "was",
        "were",
        "is",
        "are",
        "has",
        "have",
        "had",
        "would",
        "could",
        "should",
        "will",
        "can",
        "may",
        "might",
    }
)


def _question_names(query: str, evidence_text: str) -> list[str]:
    """Capitalised words in the question, taken as names. A sentence-initial one counts
    only when the evidence itself uses it as a name ("What" opens a question; "Melanie"
    opens a question about Melanie)."""
    names: list[str] = []
    for i, name in enumerate(_CAPITALISED.findall(query)):
        # A question word is never a name, whatever the evidence contains. With verbatim
        # turns in the store the evidence nearly always holds a capitalised "What", so the
        # sentence-initial heuristic below promoted "What was..." to a person and flagged
        # ~28% of answerable questions INCOMPLETE - which the answerer read as "I don't know".
        if name.lower() in _FUNCTION_WORDS or name.lower() in STOP_WORDS:
            continue
        if i == 0 and query.startswith(name) and name not in evidence_text:
            continue
        if name not in names:
            names.append(name)
    return names


def _is_about(candidate: Candidate, name: str, cache: TermCache) -> bool:
    key = name.lower()
    subject = str(candidate.payload.get("subject") or "").lower()
    return key in cache.of(candidate) or key in subject


def unsupported_subject(
    query: str, candidates: Sequence[Candidate], *, terms: TermCache | None = None
) -> str | None:
    """A note when the question is about a named person the evidence does not support.

    ``overlaps`` asks whether *any* evidence shares a term with the question. On a memory of
    a conversation between two people that is always true, for any question about either of
    them - so it never fires on the one failure conversational memory actually produces: a
    question whose premise is about the wrong person, or an event that never happened.
    "What was grandma's gift to Melanie?" retrieves Caroline's grandma and Caroline's
    necklace, shares every term, and reports COMPLETE.

    Deterministic and cheap: for each name the question carries, is there a retrieved
    memory *about* that name - by subject or by mention - that shares one of the question's
    other content terms? When none does, the bundle does not establish what was asked, and
    says so. Returns the note, or None when the question is supported or names nobody.
    """
    cache = terms or TermCache()
    memories = [c for c in candidates if c.kind == "memory"]
    if not memories:
        return None
    names = _question_names(query, " ".join(c.text for c in memories))
    if not names:
        return None
    others = content_terms(query) - {n.lower() for n in names} - _FUNCTION_WORDS
    for name in names:
        about = [c for c in memories if _is_about(c, name, cache)]
        if not about:
            return f"no retrieved memory is about {name}"
        if others and not any(cache.of(c) & others for c in about):
            return f"no retrieved memory about {name} mentions {', '.join(sorted(others)[:4])}"
    return None


class VerificationStage:
    name = "verify"

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        expansion: ExpansionStage,
        *,
        settings: RetrievalSettings,
        seeds: int = 3,
    ) -> None:
        self.uow_factory = uow_factory
        self.expansion = expansion
        self.cfg = settings
        self.seeds = seeds
        self._last_seed_groups: dict[str, list[str]] = {}
        #: node ids of the seeds the last call derived requirements from; empty means the
        #: seeds had nothing to derive from, which is not the same as "nothing required"
        self._last_seed_nodes: list[str] = []

    async def required_groups(
        self, ctx: MemoryExecutionContext, seeds: Sequence[Candidate]
    ) -> dict[str, set[str]]:
        """Group name -> node ids that satisfy it (the target and its descendants, so a
        'see Section 8' reference is satisfied by any paragraph of Section 8)."""
        node_ids = [str(c.payload["node_id"]) for c in seeds if c.payload.get("node_id")]
        self._last_seed_nodes = list(node_ids)
        if not node_ids:
            return {}
        async with self.uow_factory() as uow:
            edges = await uow.documents.edges_from(
                ctx.tenant_id, node_ids, kinds=list(_REQUIRED_EDGES)
            )
            targets = edges_to_groups(edges)
            groups: dict[str, set[str]] = {name: {t} for name, t in targets.items()}
            # which seed node needs which group (the bundle packs a seed with its companions)
            seed_groups: dict[str, list[str]] = {}
            for e in edges:
                name = f"{e.edge.value.lower()}:{e.label or e.target_id}"
                if name in groups:
                    seed_groups.setdefault(e.source_id, []).append(name)
            self._last_seed_groups = seed_groups
            frontier = sorted(set(targets.values()))
            for _ in range(3):  # section > subsection > paragraph
                if not frontier:
                    break
                children = await uow.documents.edges_from(
                    ctx.tenant_id, frontier, kinds=[ContextGraphEdge.CHILD]
                )
                if not children:
                    break
                parent_of = {e.target_id: e.source_id for e in children}
                for ids in groups.values():
                    for child, parent in parent_of.items():
                        if parent in ids:
                            ids.add(child)
                frontier = sorted(set(parent_of))
        return groups

    @staticmethod
    def satisfied(groups: dict[str, set[str]], candidates: Sequence[Candidate]) -> set[str]:
        nodes = {c.payload.get("node_id") for c in candidates if c.kind in ("chunk", "summary")}
        return {name for name, ids in groups.items() if ids & nodes}

    async def __call__(
        self,
        ctx: MemoryExecutionContext,
        routed: RoutedQuery,
        candidates: list[Candidate],
        visibility: VisibilitySpecification,
        diagnostics: dict[str, Any],
    ) -> list[Candidate]:
        with span("retrieval.verify"), stage_seconds.labels("retrieval.verify").time():
            terms = TermCache()
            evidence = [c for c in candidates if c.kind in ("chunk", "memory", "summary")]
            report: dict[str, Any] = {
                "status": EvidenceStatus.COMPLETE.value,
                "required_groups": [],
                "satisfied_groups": [],
                "missing_groups": [],
                "escalations": [],
                "notes": [],
            }
            if not evidence:
                report["status"] = EvidenceStatus.INSUFFICIENT.value
                report["notes"].append("no evidence retrieved")
                diagnostics["evidence"] = report
                return candidates
            if routed.query_type is QueryType.EXACT_IDENTIFIER:
                diagnostics["evidence"] = report
                return candidates
            if self.cfg.abstain_when_insufficient and not overlaps(
                routed.query, evidence, terms=terms
            ):
                report["status"] = EvidenceStatus.INSUFFICIENT.value
                report["notes"].append("no retrieved evidence shares a content term with the query")
                diagnostics["evidence"] = report
                return candidates
            seeds = [c for c in candidates if c.kind == "chunk" and c.expansion_edge is None][
                : self.seeds
            ]
            if not seeds and self.cfg.subject_evidence_check:
                # Conversational bundle: no document seeds to derive companions from, so the
                # subject rule is the only structural check available. INCOMPLETE, not
                # INSUFFICIENT - evidence exists; it just does not establish what was asked.
                unsupported = unsupported_subject(routed.query, candidates, terms=terms)
                if unsupported:
                    report["status"] = EvidenceStatus.INCOMPLETE.value
                    report["notes"].append(unsupported)
            groups = await self.required_groups(ctx, seeds)
            report["required_groups"] = sorted(groups)
            # COMPLETE has two very different meanings, and they were indistinguishable: the
            # companions were checked and found, or nothing was ever checked. Measured live,
            # the second was reported as the first for every memory-only turn and for every
            # bundle whose seeds were orphaned index entries. Say which happened.
            if not groups:
                report["notes"].append(
                    "no companion evidence was required"
                    if seeds
                    else "no document seeds to derive requirements from"
                )
            if seeds and not self._last_seed_nodes:
                report["notes"].append(
                    "seed chunks carry no context-graph node, so requirements could not be "
                    "derived from them"
                )
            diagnostics["evidence_targets"] = {name: sorted(ids) for name, ids in groups.items()}
            diagnostics["evidence_seed_groups"] = {
                node: sorted(set(names)) for node, names in self._last_seed_groups.items()
            }
            done = self.satisfied(groups, candidates)
            rounds = 0
            while len(done) < len(groups) and rounds < self.cfg.escalation_max_rounds:
                rounds += 1
                missing_targets = sorted(
                    {t for name, ids in groups.items() if name not in done for t in ids}
                )
                added = await self._escalate(ctx, seeds, candidates, missing_targets)
                report["escalations"].append(f"round {rounds}: fetched {len(added)} companion(s)")
                if not added:
                    break
                first_fact = next(
                    (i for i, c in enumerate(candidates) if c.kind == "fact"), len(candidates)
                )
                candidates[first_fact:first_fact] = added
                done = self.satisfied(groups, candidates)
            report["satisfied_groups"] = sorted(done)
            report["missing_groups"] = sorted(set(groups) - done)
            conflicts = _conflicting_memories(candidates)
            if conflicts:
                report["notes"].append(
                    "conflicting memories from different principals: "
                    + "; ".join(f"{a} vs {b}" for a, b in conflicts)
                )
            if report["missing_groups"]:
                report["status"] = EvidenceStatus.INCOMPLETE.value
                report["notes"].append("required companion evidence could not be retrieved")
            diagnostics["evidence"] = report
        return candidates

    async def _escalate(
        self,
        ctx: MemoryExecutionContext,
        seeds: Sequence[Candidate],
        candidates: Sequence[Candidate],
        targets: Sequence[str],
    ) -> list[Candidate]:
        """Fetch the companions directly: the first chunk under any acceptable node."""
        from memory_service.modules.context.expansion import chunk_candidate

        present = {c.record_id for c in candidates}
        out: list[Candidate] = []
        async with self.uow_factory() as uow:
            chunks = await uow.documents.chunks_for_nodes(ctx.tenant_id, targets)
        seen_nodes: set[str] = set()
        source = seeds[0].record_id if seeds else "verify"
        for c in chunks:
            if c.chunk_id in present or c.node_id in seen_nodes:
                continue
            seen_nodes.add(c.node_id)
            out.append(chunk_candidate(c, score=0.5, edge="ESCALATION", source=source))
            present.add(c.chunk_id)
        return out
