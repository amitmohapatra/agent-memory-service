"""Hierarchical summaries without an LLM: deterministic extractive summaries for every
section/subsection node and for the document, built from the chunks beneath each node.

Sentence scoring = lead bonus + keyword density (document-level term frequencies, stop words
removed) + a small bonus for sentences carrying numbers or defined terms. The result is
stable for the same input, cites nothing it did not contain, and is bounded in length so it
fits the ``summaries`` bucket of a ContextBundle. Summaries are indexed as ``kind="summary"``
records (``sum_<node_id>``) and answer GLOBAL_SUMMARY questions; they never replace chunks
as evidence.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from memory_service.domain.documents import Chunk, DocumentNode
from memory_service.domain.enums import Representation
from memory_service.modules.memory.native import _STOP as STOP_WORDS

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z][a-z0-9\-]+")
_TABLE_LINE = re.compile(r"^\s*\|")


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _TABLE_LINE.match(line) or line.startswith("<!--"):
            continue
        for segment in re.split(r"\s+(?=\|)", line):
            if segment.startswith("|"):
                continue  # inline table fragment
            for piece in _SENT.split(segment):
                s = piece.strip().strip("*#- ")
                if 25 <= len(s) <= 400:
                    out.append(s)
    return out


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP_WORDS and len(w) > 2]


def summarize(text: str, *, max_sentences: int = 3, max_chars: int = 600) -> str:
    """Extractive summary: top sentences by keyword density, returned in original order."""
    sents = sentences(text)
    if not sents:
        return ""
    if len(sents) <= max_sentences and sum(len(s) for s in sents) <= max_chars:
        return " ".join(sents)
    freq = Counter(_terms(text))
    scored: list[tuple[float, int, str]] = []
    for i, s in enumerate(sents):
        terms = _terms(s)
        if not terms:
            continue
        density = sum(freq[t] for t in terms) / (len(terms) + 4)
        lead = 1.5 if i == 0 else (0.5 if i == 1 else 0.0)
        numbers = 0.4 if re.search(r"\d", s) else 0.0
        defined = 0.6 if re.search(r"\b(means|refers to|is defined as)\b", s) else 0.0
        scored.append((density + lead + numbers + defined, i, s))
    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen: list[tuple[int, str]] = []
    used = 0
    for _, i, s in scored:
        if len(chosen) >= max_sentences or used + len(s) > max_chars:
            continue
        chosen.append((i, s))
        used += len(s) + 1
    chosen.sort()
    return " ".join(s for _, s in chosen)


def build_summaries(
    nodes: Sequence[DocumentNode], chunks: Sequence[Chunk], *, title: str
) -> dict[str, str]:
    """node_id -> summary for DOCUMENT/SECTION/SUBSECTION nodes that have text beneath them."""
    by_node: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_node.setdefault(c.node_id, []).append(c)
    # text under a node = chunks of the node and of every descendant
    children: dict[str, list[str]] = {}
    for n in nodes:
        if n.parent_id:
            children.setdefault(n.parent_id, []).append(n.node_id)

    def descendants(node_id: str) -> list[str]:
        out, stack = [], [node_id]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(children.get(cur, []))
        return out

    out: dict[str, str] = {}
    for n in nodes:
        if n.representation not in (
            Representation.DOCUMENT,
            Representation.SECTION,
            Representation.SUBSECTION,
        ):
            continue
        texts = []
        for nid in descendants(n.node_id):
            texts.extend(c.text for c in by_node.get(nid, []))
        joined = "\n".join(texts)
        if not joined.strip():
            continue
        is_doc = n.representation is Representation.DOCUMENT
        summary = summarize(
            joined, max_sentences=5 if is_doc else 3, max_chars=900 if is_doc else 500
        )
        if summary:
            label = title if is_doc else (n.title or n.node_id)
            out[n.node_id] = f"{label}: {summary}"
    return out
