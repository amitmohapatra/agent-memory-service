"""Conservative semantic reuse of source-backed context, never generated answers.

The lookup bucket must already bind authorization/content revisions, execution scope,
model/configuration, document filters and token budget. Ordered lexical guards prevent
nearby embeddings from confusing names, negation, quantities, dates or relation direction.
One representative per guarded signature bounds lookup to O(query length + dimension).
"""

from __future__ import annotations

import contextlib
import math
import re
import unicodedata

import orjson

from memory_service.domain.ids import stable_key
from memory_service.ports.cache import CacheProvider, CacheUnavailable

_POLITE = re.compile(r"^(?:please\s+|could you\s+|can you\s+)", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"[\"'`<>]|\b(?:now|today|yesterday|tomorrow|latest|current)\b", re.IGNORECASE
)
_MAX_QUERY = 512
_MAX_ENTRY = 32768


def guarded_signature(query: str) -> str | None:
    """Deliberately narrow equivalence; broad paraphrase reuse needs measured calibration."""
    if not query or len(query) > _MAX_QUERY or _FORBIDDEN.search(query):
        return None
    text = unicodedata.normalize("NFC", query).strip()
    text = _POLITE.sub("", text).rstrip("?.!\u3002\uff1f\uff01").strip()
    # Preserve order, case, negation, numbers, diacritics and all content words. Removing
    # only polite framing/punctuation gives a useful safe starting point for calibration.
    return " ".join(text.split()) or None


def semantic_key(namespace: str, query: str) -> str | None:
    signature = guarded_signature(query)
    return f"semantic-context:{stable_key(namespace, signature)}" if signature else None


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    norm = math.hypot(*left) * math.hypot(*right)
    if not norm or not math.isfinite(norm):
        return 0.0
    return math.fsum(a * b for a, b in zip(left, right, strict=True)) / norm


class SemanticBundleCache:
    def __init__(self, cache: CacheProvider, *, ttl_seconds: int, similarity: float = 0.98):
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.similarity = similarity

    async def lookup(self, raw: bytes, vector: list[float]) -> bytes | None:
        if len(raw) > _MAX_ENTRY:
            return None
        try:
            entry = orjson.loads(raw)
            if cosine(entry["vector"], vector) < self.similarity:
                return None
            return await self.cache.get(entry["bundle_key"])
        except (ValueError, TypeError, KeyError, CacheUnavailable):
            return None

    async def store(self, key: str, bundle_key: str, vector: list[float]) -> None:
        raw = orjson.dumps({"bundle_key": bundle_key, "vector": vector})
        if len(raw) <= _MAX_ENTRY:
            with contextlib.suppress(CacheUnavailable):
                await self.cache.set(key, raw, ttl_seconds=self.ttl_seconds)
