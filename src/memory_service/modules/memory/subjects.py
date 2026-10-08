"""The write path's one question about subjects: is this new statement about the same
thing as that stored memory?

``domain.subjects`` compares two subject *strings*. A statement is more than its subject
string, so this service decides what to compare:

* a memory about a **named thing** ("Forklift 4", "SKU-1001", "Acme Logistics", from the
  fact rule or a model) is compared by that name;
* a memory about a **principal or anchor** ("user:u1", "thread:thr_1") is about that
  identity, so two such memories share a subject only when the identity is the same - and
  then the *slot* decides: a single-valued predicate (a person has one city) is the same
  subject outright; otherwise the two topics ("tabs over spaces", "spaces instead of tabs")
  must share words.

The vocabulary is the packs plus what the memories at hand define ("hazardous materials
(hazmat)"), and the names those memories mention are the tenant's known names - a short
form two of them extend is no evidence of either. Both come from the rows consolidation has
already loaded: nothing is configured and nothing extra is read.

With the conflict adjudicator enabled, a pair the words leave undecided can also be scored by
the multilingual encoder (subject against subject), the encoder the service already runs;
its vectors are kept (float32, by encoder) in a bounded in-process cache, so a recurring
subject is encoded once.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.predicates import is_single_valued
from memory_service.domain.subjects import (
    MAX_DEFINITION_CHARS,
    Subject,
    SubjectMatch,
    SubjectVerdict,
    Vocabulary,
    compare,
    defined_aliases,
    is_identity,
    parse,
    spellings,
    vocabulary,
)
from memory_service.ports.intelligence import MemoryCandidate
from memory_service.ports.models import EmbeddingProvider

#: Least share of words two topics of one identity's slot must share to be worth asking
#: about ("tea in the morning" and "coffee in the morning" share one word of three).
TOPIC_OVERLAP = 1 / 3
_VECTOR_CACHE = 4096


@dataclass(frozen=True, slots=True)
class SubjectPair:
    """A candidate against one stored memory."""

    #: the two subjects themselves: SAME is what a slot rule may rely on
    subject: SubjectMatch
    #: the two statements: the subject, and for an identity its slot and topic. What the
    #: conflict adjudicator is gated by.
    statement: SubjectMatch
    #: two named subjects that are not known to be one ("Tower A" / "Tower B", "Tower A" /
    #: "Tower", "Bank of China" / "China Bank"): no rule may merge the two statements,
    #: however alike their wording
    apart: bool = False


@lru_cache(maxsize=4096)
def _defined(text: str) -> tuple[tuple[str, str], ...]:
    return tuple(defined_aliases(text))


def _different(reason: str) -> SubjectMatch:
    return SubjectMatch(SubjectVerdict.DIFFERENT, 0.0, reason)


def written(subject: str | None, entities: Iterable[object] = ()) -> str:
    """The subject as it was written. The fact rule stores it lower-cased ("store la") and
    keeps the original among the entities ("Store LA"), and case is what tells a code from
    a word ("LA", "la"), so the original is compared when there is one."""
    if not subject:
        return ""
    for entity in entities:
        text = str(entity).strip()
        if _THE.sub("", text.lower()) == subject:
            return text
    return subject


_THE = re.compile(r"^the\s+")


class SubjectMatcher:
    """Same-subject decisions for consolidation (see the module docstring)."""

    def __init__(self, embedding: EmbeddingProvider | None = None) -> None:
        # the hash stand-in would call any two strings that share a few letters similar
        usable = embedding is not None and not embedding.fingerprint().startswith("hash-")
        self.embedding = embedding if usable else None
        #: (encoder fingerprint, subject) -> unit float32 vector, least recently used first
        self._vectors: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    # -- candidate generation ---------------------------------------------------------
    @staticmethod
    def spellings(subject: str | None) -> list[str]:
        """The stored spellings an exact subject lookup should try for ``subject``."""
        return spellings(subject) if subject else []

    # -- comparison -------------------------------------------------------------------
    def pairs(
        self, candidate: MemoryCandidate, existing: Sequence[CanonicalMemory]
    ) -> dict[str, SubjectPair]:
        """Every memory of ``existing`` (by id) against ``candidate``, by their words."""
        vocab = self._vocabulary(candidate, existing)
        names = self._names(candidate, existing, vocab)
        mine = parse(written(candidate.subject, candidate.entities), vocab)
        return {
            mem.memory_id: self._pair(candidate, mine, mem, vocab, names, cosine=None)
            for mem in existing
        }

    async def with_vectors(
        self,
        candidate: MemoryCandidate,
        existing: Sequence[CanonicalMemory],
        pairs: dict[str, SubjectPair],
    ) -> dict[str, SubjectPair]:
        """``pairs`` with the encoder's say on two named subjects whose words differ: a
        cosine of at least ``DENSE_POSSIBLE`` makes such a pair POSSIBLE. Unchanged without
        an encoder; a pair the words decided (or blocked) is never re-scored. ``existing``
        should be only the memories the answer matters for: each is encoded."""
        if self.embedding is None or not candidate.subject or is_identity(candidate.subject):
            return pairs
        soft = [
            mem
            for mem in existing
            if mem.subject
            and not is_identity(mem.subject)
            and pairs[mem.memory_id].subject.verdict is SubjectVerdict.DIFFERENT
            and pairs[mem.memory_id].subject.reason.startswith("names differ")
        ]
        if not soft:
            return pairs
        mine_text = written(candidate.subject, candidate.entities)
        theirs = {m.memory_id: written(m.subject, _entities(m)) for m in soft}
        vectors = await self._encode([mine_text, *theirs.values()])
        vocab = self._vocabulary(candidate, existing)
        names = self._names(candidate, existing, vocab)
        mine = parse(mine_text, vocab)
        out = dict(pairs)
        for mem in soft:
            cosine = float(np.dot(vectors[mine_text], vectors[theirs[mem.memory_id]]))
            out[mem.memory_id] = self._pair(candidate, mine, mem, vocab, names, cosine=cosine)
        return out

    def _pair(
        self,
        candidate: MemoryCandidate,
        mine: Subject,
        mem: CanonicalMemory,
        vocab: Vocabulary,
        names: Sequence[Subject],
        *,
        cosine: float | None,
    ) -> SubjectPair:
        theirs = parse(written(mem.subject, _entities(mem)), vocab)
        if mine.empty or theirs.empty:
            nothing = _different(NO_SUBJECT)
            return SubjectPair(nothing, nothing)
        subject = compare(mine, theirs, cosine=cosine, names=names)
        if mine.identity is None and theirs.identity is None:
            # only SAME lets the wording decide: "Bank of China" / "China Bank" share every
            # word, "Tower A" / "Tower" every word but a code
            apart = subject.verdict is not SubjectVerdict.SAME
            return SubjectPair(subject, subject, apart=apart)
        if subject.verdict is SubjectVerdict.DIFFERENT:
            return SubjectPair(subject, subject)
        return SubjectPair(subject, self._slot(candidate, mem, vocab))

    @staticmethod
    def _slot(candidate: MemoryCandidate, mem: CanonicalMemory, vocab: Vocabulary) -> SubjectMatch:
        """Two statements about one identity: the same single-valued slot, or topics that
        share words (a differing predicate can at most make the pair POSSIBLE: "I avoid
        meetings before 10am" and "I prefer meetings before 10am")."""
        same_predicate = candidate.predicate == mem.predicate
        if same_predicate and is_single_valued(candidate.predicate):
            return SubjectMatch(SubjectVerdict.SAME, 1.0, f"same slot {candidate.predicate}")
        mine = parse(candidate.object or candidate.content, vocab)
        theirs = parse(mem.object or mem.content, vocab)
        if mine.empty or theirs.empty:
            return _different("no topic")
        topic = compare(mine, theirs, topic=True)
        if topic.verdict is SubjectVerdict.POSSIBLE and topic.reason == "shared topic words":
            shared = len(mine.tokens & theirs.tokens) / len(mine.tokens | theirs.tokens)
            if shared < TOPIC_OVERLAP:
                return _different(f"topics share {shared:.2f} of their words")
        if topic.verdict is SubjectVerdict.SAME and not same_predicate:
            return SubjectMatch(SubjectVerdict.POSSIBLE, 0.8, "same topic, another predicate")
        if topic.verdict is not SubjectVerdict.DIFFERENT and not same_predicate:
            return SubjectMatch(SubjectVerdict.DIFFERENT, 0.0, "another predicate and topic")
        return topic

    # -- what the memories at hand teach ----------------------------------------------
    @staticmethod
    def _vocabulary(candidate: MemoryCandidate, existing: Sequence[CanonicalMemory]) -> Vocabulary:
        learned = [
            pair
            for text in _texts(candidate, existing)
            for pair in _defined(text[:MAX_DEFINITION_CHARS])
        ]
        return vocabulary().with_aliases(learned)

    @staticmethod
    def _names(
        candidate: MemoryCandidate, existing: Sequence[CanonicalMemory], vocab: Vocabulary
    ) -> tuple[Subject, ...]:
        raw: set[str] = set(candidate.entities)
        for mem in existing:
            if mem.subject and not is_identity(mem.subject):
                raw.add(written(mem.subject, _entities(mem)))
            raw.update(str(e) for e in _entities(mem))
        parsed = (parse(name, vocab) for name in sorted(raw))
        return tuple(p for p in parsed if not p.empty and p.identity is None)

    # -- vectors ----------------------------------------------------------------------
    async def _encode(self, texts: Iterable[str]) -> dict[str, np.ndarray]:
        """Unit float32 vectors of ``texts``; the cached ones are refreshed before any new
        one is inserted, so a call never evicts what it is about to read."""
        assert self.embedding is not None
        model = self.embedding.fingerprint()
        wanted = list(dict.fromkeys(texts))
        found: dict[str, np.ndarray] = {}
        for text in wanted:
            vector = self._vectors.get((model, text))
            if vector is not None:
                self._vectors.move_to_end((model, text))
                found[text] = vector
        missing = [t for t in wanted if t not in found]
        if missing:
            encoded = await self.embedding.embed_documents(missing)
            for text, raw in zip(missing, encoded, strict=True):
                vector = np.asarray(raw, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                found[text] = vector / norm if norm else vector
                self._vectors[(model, text)] = found[text]
            while len(self._vectors) > _VECTOR_CACHE:
                self._vectors.popitem(last=False)
        return found


#: the reason a pair without a subject on one side carries
NO_SUBJECT = "no subject"


def _entities(mem: CanonicalMemory) -> list[object]:
    return list(mem.system_metadata.get("entities") or ())


def _texts(candidate: MemoryCandidate, existing: Sequence[CanonicalMemory]) -> list[str]:
    return [candidate.content, *(m.content for m in existing)]
