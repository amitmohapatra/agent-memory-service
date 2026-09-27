"""Text that is safe to store.

PostgreSQL rejects NUL (0x00) in a ``text`` column outright — psycopg raises ``DataError``
and the whole transaction fails. That is not a theoretical concern: the document path hit it
first (one NUL in an uploaded file destroyed the entire document), and the observation path
hit it again from an agent writing back a tool result that contained a NUL byte, where it
surfaced as a 500 rather than as a stored observation.

Both paths now go through here. Sanitising at the two edges where external bytes become text
is deliberate — doing it once in the repository layer would hide the fact that the input was
malformed, and doing it in each caller is how it came to be missing from one of them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterator
from itertools import pairwise

#: C0 controls except tab (\x09), newline (\x0a) and carriage return (\x0d), plus DEL and the
#: C1 block. Whitespace is meaningful and kept; the rest is never intentional in text and is
#: either an encoding accident or a probe.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# A small explicit exclusion, not a language detector. Unknown short statements must
# remain eligible for retention and verification: one word can express a whole claim.
ACKNOWLEDGEMENT = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|ok|okay|sure|great|cool|yes|no|got it|sounds good)\b[.!]?$",
    re.IGNORECASE,
)


def sanitise(text: str) -> str:
    """Strip control characters that cannot be stored, keeping tabs and newlines."""
    return _CONTROL.sub("", text)


def token_units(character: str) -> int:
    """Quarter-token planning units: English heuristic, non-ASCII UTF-8 byte allowance.

    This is a conservative packing estimate, not a tokenizer or a billed-token count.
    Non-ASCII text must not inherit the English four-characters-per-token assumption.
    """
    if character.isascii():
        return 5 if character == "\n" else 1
    return 4 * len(character.encode("utf-8"))


def normalise_number(raw: str) -> str:
    """Normalize decimal digits and explicit Arabic separators without guessing a locale.

    Preserve the existing convention that comma groups thousands. A locale is required
    to distinguish European decimal commas; this helper deliberately does not infer one.
    """
    separators = {",": "", "\u066c": "", "\uff0c": "", "\u066b": ".", "\uff0e": "."}
    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else separators.get(char, char)
        for char in raw
    )


# Scripts commonly written without spaces. Character bigrams need no language detector
# or dictionary; combining marks remain attached to their base characters.
_UNSEGMENTED = re.compile(
    r"[\u0e00-\u0fff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u9fff\U00020000-\U0003134f]"
)
SENTENCE_BREAK = re.compile(
    r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f\u0964\u0965\u061f\u06d4])\s*|\n+"
)


def _lexical_unit(
    parts: list[list[str]],
    unsegmented: bool,
    ascii_tokens: Callable[[str], list[str]],
    max_ngram: int,
) -> list[str]:
    clusters = ["".join(part) for part in parts]
    if unsegmented:
        grams = clusters + [a + b for a, b in pairwise(clusters)]
        for width in range(3, max_ngram + 1):
            grams.extend("".join(clusters[i : i + width]) for i in range(len(clusters) - width + 1))
        return grams
    word = "".join(clusters)
    return ascii_tokens(word) if word.isascii() else [word]


def unicode_tokens(
    text: str, ascii_tokens: Callable[[str], list[str]], *, max_ngram: int = 2
) -> Iterator[str]:
    """O(n) time/output, with caller-owned ASCII stemming and stop-word policy."""
    if not 2 <= max_ngram <= 4:
        raise ValueError("max_ngram must be between two and four")
    clusters: list[list[str]] = []
    unsegmented = False
    for char in unicodedata.normalize("NFC", text).casefold():
        category = unicodedata.category(char)[0]
        if category == "M" and clusters:
            clusters[-1].append(char)
            continue
        is_word = category in {"L", "N"} or char == "'"
        next_unsegmented = bool(_UNSEGMENTED.fullmatch(char))
        if clusters and (not is_word or next_unsegmented != unsegmented):
            yield from _lexical_unit(clusters, unsegmented, ascii_tokens, max_ngram)
            clusters = []
        if is_word:
            clusters.append([char])
            unsegmented = next_unsegmented
    if clusters:
        yield from _lexical_unit(clusters, unsegmented, ascii_tokens, max_ngram)
