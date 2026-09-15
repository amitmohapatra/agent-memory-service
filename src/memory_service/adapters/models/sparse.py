"""BM25 sparse encoder for Qdrant's server-side IDF modifier.

The client emits term frequencies with BM25 saturation (k1, b); Qdrant multiplies by IDF
computed over the collection (``Modifier.IDF``). Tokenisation is deterministic (lowercase,
alnum tokens, English stop words, light suffix stemming) so it needs no model download; the
same tokeniser is used for queries and documents. Term ids are 32-bit hashes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from memory_service.ports.models import ProviderInfo
from memory_service.ports.search import SparseVector

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOP_WORDS = """
a an the and or but if then of to in on at by for with from as is are was were
be been being it its this that these those there here we you they he she i me my
our your their his her them do does did done have has had having not no nor so
than too very can will would should could may might must shall into onto over
under about above below between during before after also
"""
_STOP = frozenset(_STOP_WORDS.split())
_SUFFIXES = (
    "ization",
    "isation",
    "ations",
    "ation",
    "ments",
    "ment",
    "ness",
    "ings",
    "ing",
    "ies",
    "ied",
    "ers",
    "er",
    "ed",
    "es",
    "s",
    "ly",
)


def stem(token: str) -> str:
    if len(token) <= 4:
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            base = token[: -len(suf)]
            if suf in ("ies", "ied"):
                return base + "y"
            return base
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def term_id(token: str) -> int:
    return int(hashlib.blake2b(token.encode(), digest_size=4).hexdigest(), 16)


class Bm25SparseEncoder:
    info = ProviderInfo(
        name="bm25-sparse", version="1", license="Apache-2.0", origin="internal", locality="local"
    )

    def __init__(self, *, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 256.0) -> None:
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._encode(t, query=False) for t in texts]

    def encode_query(self, text: str) -> SparseVector:
        return self._encode(text, query=True)

    def _encode(self, text: str, *, query: bool) -> SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])
        counts: dict[int, float] = {}
        for tok in tokens:
            counts[term_id(tok)] = counts.get(term_id(tok), 0.0) + 1.0
        if query:
            items = sorted((idx, 1.0) for idx in counts)
        else:
            dl = len(tokens)
            norm = self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
            items = sorted((idx, (tf * (self.k1 + 1)) / (tf + norm)) for idx, tf in counts.items())
        return SparseVector(indices=[i for i, _ in items], values=[v for _, v in items])

    def fingerprint(self) -> str:
        return f"bm25-v1-k{self.k1}-b{self.b}"
