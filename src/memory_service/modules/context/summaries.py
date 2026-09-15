"""Hierarchical summaries without an LLM: deterministic extractive summaries for every
section/subsection node and for the document, built from the chunks beneath each node.

Sentence scoring = lead bonus + keyword density (document-level term frequencies, stop words
removed) + a small bonus for sentences carrying numbers or defined terms. The result is
stable for the same input, cites nothing it did not contain, and is bounded in length so it
fits the ``summaries`` bucket of a ContextBundle. Summaries are indexed as ``kind="summary"``
records (``sum_<node_id>``) and answer GLOBAL_SUMMARY questions; they never replace chunks
as evidence. With the ``summaries`` LLM use enabled, a bounded number of them is rewritten
abstractively from the same source text; the extractive summary stays the fallback.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from memory_service.domain.documents import Chunk, DocumentNode
from memory_service.domain.enums import Representation
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.native import _STOP as STOP_WORDS

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z][a-z0-9\-]+")
_TABLE_LINE = re.compile(r"^\s*\|")

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
_SUMMARY_SYSTEM = (
    "You summarise a section of a document for a retrieval index. Write one abstractive "
    "summary of at most {max_chars} characters covering the key facts, figures, names and "
    "defined terms of the source text. State only what the source says; no preamble. "
    'Return JSON only: {{"summary": "..."}}.'
)
SOURCE_CHARS = 6000


def summary_limits(representation: Representation) -> tuple[int, int]:
    """(max_sentences, max_chars) of a node summary; the document gets the larger budget."""
    is_doc = representation is Representation.DOCUMENT
    return (5, 900) if is_doc else (3, 500)


def summary_label(node: DocumentNode, *, title: str) -> str:
    if node.representation is Representation.DOCUMENT:
        return title
    return node.title or node.node_id


def accept_abstractive(result: dict[str, Any] | None, *, max_chars: int) -> str | None:
    """The model's summary when it is non-empty and within 1.5x the budget, else ``None``."""
    if result is None:
        return None
    text = " ".join(str(result.get("summary", "")).split())
    if not text or len(text) > int(max_chars * 1.5):
        return None
    return text


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


def node_texts(nodes: Sequence[DocumentNode], chunks: Sequence[Chunk]) -> dict[str, str]:
    """node_id -> text beneath each DOCUMENT/SECTION/SUBSECTION node (its chunks and every
    descendant's), in document order; nodes with nothing beneath them are left out."""
    by_node: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_node.setdefault(c.node_id, []).append(c)
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
        if joined.strip():
            out[n.node_id] = joined
    return out


def build_summaries(
    nodes: Sequence[DocumentNode], chunks: Sequence[Chunk], *, title: str
) -> dict[str, str]:
    """node_id -> summary for DOCUMENT/SECTION/SUBSECTION nodes that have text beneath them."""
    by_id = {n.node_id: n for n in nodes}
    out: dict[str, str] = {}
    for nid, joined in node_texts(nodes, chunks).items():
        n = by_id[nid]
        max_sentences, max_chars = summary_limits(n.representation)
        summary = summarize(joined, max_sentences=max_sentences, max_chars=max_chars)
        if summary:
            out[nid] = f"{summary_label(n, title=title)}: {summary}"
    return out


async def abstractive_summaries(
    assist: LLMAssist,
    nodes: Sequence[DocumentNode],
    chunks: Sequence[Chunk],
    summaries: dict[str, str],
    *,
    title: str,
    max_nodes: int = 24,
) -> dict[str, str]:
    """Rewrite up to ``max_nodes`` extractive summaries (largest sources first) with the
    model; every other node, and every node the model cannot help with, keeps its extractive
    summary."""
    out = dict(summaries)
    if not assist.wants("summaries") or not summaries:
        return out
    by_id = {n.node_id: n for n in nodes}
    texts = node_texts(nodes, chunks)
    chosen = sorted(
        (nid for nid in summaries if nid in by_id and nid in texts),
        key=lambda nid: (-len(texts[nid]), nid),
    )[:max_nodes]
    for nid in chosen:
        node = by_id[nid]
        _, max_chars = summary_limits(node.representation)
        label = summary_label(node, title=title)
        extractive = summaries[nid][len(label) + 2 :]
        result = await assist.structured(
            "summaries",
            system=_SUMMARY_SYSTEM.format(max_chars=max_chars),
            user=(
                f"Document: {title}\nSection: {label}\n\nExtractive summary:\n{extractive}\n\n"
                f"Source text:\n{texts[nid][:SOURCE_CHARS]}"
            ),
            schema=SUMMARY_SCHEMA,
            max_tokens=max(128, max_chars // 2),
        )
        summary = accept_abstractive(result, max_chars=max_chars)
        if summary is not None:
            out[nid] = f"{label}: {summary}"
    return out
