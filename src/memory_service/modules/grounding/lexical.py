"""Deterministic lexical signals shared by the cascade and the stand-in NLI: content-token
coverage, number agreement, negation and polarity (antonym) conflicts."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9](?:[a-z0-9'\-/]*[a-z0-9])?")
_NUMBER = re.compile(r"(?<![A-Za-z\d])\d+(?:[.,]\d+)*")
_STOP_WORDS = """
a an the and or but if then of to in on at by for with from as is are was were be been
being it its this that these those there here i me my we our you your they them he she
his her do does did done have has had having so than too very can will would should could
may might must shall into onto about please which who whom whose what when where while
also both each such per via any some all
"""
_STOP = frozenset(_STOP_WORDS.split())
_NEGATION = frozenset({"not", "no", "never", "none", "nobody", "nothing", "neither", "nor"})
_ANTONYMS: dict[str, frozenset[str]] = {}


def _pair(left: str, right: str) -> None:
    lefts, rights = frozenset(left.split()), frozenset(right.split())
    for w in lefts:
        _ANTONYMS[w] = _ANTONYMS.get(w, frozenset()) | rights
    for w in rights:
        _ANTONYMS[w] = _ANTONYMS.get(w, frozenset()) | lefts


_pair(
    "increase increased increases increasing rise rises rose risen rising grew grow grows "
    "growth higher up gain gains gained improved improve improves improvement",
    "decrease decreased decreases decreasing fall falls fell fallen falling decline declined "
    "declines drop dropped drops shrank shrink lower down loss losses lost worsened worsen "
    "deteriorated",
)
_pair("profit profits profitable", "loss losses unprofitable")
_pair("approved approve approves approval accepted accept", "rejected reject rejects denied")
_pair("more", "less fewer")
_pair("above exceeded exceed exceeds", "below missed miss misses")
_pair("won win wins", "lost lose loses")
_pair("opened open opens", "closed close closes")
_pair("started start starts began begin", "ended end ends")
_pair("true correct", "false incorrect")
_pair("always", "never")
_pair("positive", "negative")
_pair("faster", "slower")
_pair("expanded expand expands", "contracted contract contracts")


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """Stemmed content words plus normalised numbers."""
    out = {stem(w) for w in words(text) if w not in _STOP and len(w) > 1 and not w.isdigit()}
    return out | numbers(text)


def numbers(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "")
        try:
            out.add(str(float(cleaned)))
        except ValueError:
            out.add(cleaned)
    return out


def has_negation(text: str) -> bool:
    lowered = text.lower()
    return bool(set(re.findall(r"[a-z']+", lowered)) & _NEGATION) or "n't" in lowered


def coverage(hypothesis: str, premise: str) -> float:
    """Share of the hypothesis' content tokens that the premise contains."""
    h = content_tokens(hypothesis)
    if not h:
        return 0.0
    return len(h & content_tokens(premise)) / len(h)


def number_conflict(hypothesis: str, premise: str) -> bool:
    """Both sides carry numbers and the hypothesis states one the premise does not.

    Two failure modes have to be told apart. Requiring *no* overlap misses the commonest
    fabrication — quoting one real figure and altering another ("increased to EUR 150 million
    from EUR 81 million" where the premise says 98 and 81): the shared 81 makes the claim look
    anchored while the load-bearing number is invented. But treating any unmatched number as a
    clash would flag a premise that is simply shorter ("Adjusted EBITDA increased to EUR 98
    million" against a claim that also mentions the 81 it rose from), which omits rather than
    disagrees.

    So a conflict needs *divergence*: each side carries a number the other does not. Omission
    alone is not disagreement.
    """
    h, p = numbers(hypothesis), numbers(premise)
    return bool(h - p) and bool(p - h)


def polarity_conflict(hypothesis: str, premise: str) -> bool:
    """A hypothesis word whose antonym (and not the word itself) is in the premise."""
    h, p = set(words(hypothesis)), set(words(premise))
    for w in h:
        opposites = _ANTONYMS.get(w)
        if opposites and (opposites & p) and w not in p and not (opposites & h):
            return True
    return False


def conflicts(hypothesis: str, premise: str) -> list[str]:
    """Named deterministic conflicts between a claim and an evidence text."""
    out: list[str] = []
    if number_conflict(hypothesis, premise):
        out.append("number")
    if has_negation(hypothesis) != has_negation(premise):
        out.append("negation")
    if polarity_conflict(hypothesis, premise):
        out.append("polarity")
    return out
