"""A conversation turn restated so it stands on its own (``memory_restatement``, ADR 0027).

A turn is written for the person it answers: "Yeah, we went last weekend, the kids loved
it." Months later nothing in it says who went where or when, so a question about it shares
no words with it and lands far from it in every embedding space. At ingest, when the bound
key and the tenant's policy allow, the model restates the turn with the turn before it for
context - who by name, what, where, with whom, and when as a date - plus up to three short
facts it states. The restatement is appended to the turn's own index key (the turn itself is
kept verbatim and is what every reader sees): measured in LongMemEval's own study, the turn
with its facts appended raised round-level recall@10 from 0.692 to 0.784, where the facts as
separate keys lowered it.

Model output is untrusted. A restatement or fact is kept only when every number in it, and
every capitalised name, occurs in what the model was shown (the turn, the turn before it, the
speakers) - a date the model computed from the session date is the one exception - so a
hallucinated person, place or quantity never enters the index. It goes through the Bifrost
gateway like every other use: which model (an OpenAI, Gemini or local one) is the tenant's
policy, and whose key pays is the bound identity's. It is opt-in (``OPT_IN_LLM_USES``): a
tenant's policy names it, because it costs a model call per conversation message.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.temporal import dated_mentions

USE: Final = "memory_restatement"
MAX_TURN_CHARS: Final = 1500
MAX_CONTEXT_CHARS: Final = 600
MAX_RESTATEMENT_CHARS: Final = 400
MAX_FACTS: Final = 3
MAX_FACT_CHARS: Final = 200
MAX_RELATIONS: Final = 4
#: a line this much of whose words are the turn's own is an echo, not a restatement
ECHO_SHARE: Final = 0.9
MAX_RELATION_CHARS: Final = 80

SYSTEM: Final = (
    "You restate one turn of a conversation so it makes sense on its own months later. "
    "The turn's speaker is 'turn.speaker': I, me and my mean that person; you and your mean "
    "'turn.addressee', the speaker of the previous turn. Write every person by name, never I, "
    "you, he, she or they. Replace each relative time with the absolute date given for it in "
    "'dates' (else compute it from 'said_on'). Keep who, what, where, with whom, when and why. "
    "Do not add anything the turn does not say. Then list up to three short facts the turn "
    "states, each a full sentence that starts with the person's name, and up to four "
    "relations: a person's name, a short verb phrase in snake_case (went_to, adopted, likes, "
    "works_at), and a short object copied from the turn. If the turn says nothing worth "
    "remembering (a greeting, thanks, a question back), return an empty restatement and no "
    "facts or relations."
)
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "restatement": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["restatement", "facts", "relations"],
    "additionalProperties": False,
}
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
#: a date the model computed: ISO, a month with or without its day ("July 14", "14th of
#: July"), a weekday, or a four-digit year
_DATE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b(19|20)\d{{2}}\b"
    rf"|\b(\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?)?({_MONTHS})(\s+\d{{1,2}}(st|nd|rd|th)?\b)?"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_PREDICATE = re.compile(r"[a-z][a-z_]{1,39}")
_WORD = re.compile(r"[\w'-]{3,}")
#: words a relation may use that the turn need not: the model writes names for pronouns
#: what a model writes when it has no object (or subject) for a relation
_PLACEHOLDERS = frozenset({"n/a", "na", "none", "unknown", "null", "nothing", "-", "?"})
_FUNCTION = frozenset(
    {"the", "and", "her", "his", "their", "its", "our", "your", "with", "for", "from", "about"}
)
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z'-]+")


@dataclass(frozen=True)
class Restated:
    """What the model said a turn states, after the checks: the restatement with its facts
    (appended to the turn's index key) and the relations the graph links the turn by."""

    text: str
    relations: list[tuple[str, str, str]] = field(default_factory=list)


async def restate(
    assist: LLMAssist,
    *,
    text: str,
    speaker: str,
    said_at: datetime,
    before: dict[str, str] | None,
) -> Restated | None:
    """The turn restated, ready to append to its index key, with its relations; ``None`` when
    the model was not consulted or said nothing that passed the checks."""
    if not assist.wants(USE) or not text.strip():
        return None
    prior = before or {}
    said_before = prior.get("text", "")[:MAX_CONTEXT_CHARS]
    addressee = prior.get("speaker", "")
    payload = {
        "said_on": said_at.date().isoformat(),
        "weekday": said_at.strftime("%A"),
        # the relative dates the turn names, as the service resolved them at ingest: the
        # model substitutes them rather than doing calendar arithmetic it gets wrong
        "dates": {m["text"]: m["date"] for m in dated_mentions(text, base=said_at)},
        "previous_turn": {"speaker": addressee, "text": said_before} if said_before else None,
        "turn": {"speaker": speaker, "addressee": addressee, "text": text[:MAX_TURN_CHARS]},
    }
    output = await assist.structured(
        USE,
        system=SYSTEM,
        user=json.dumps(payload, ensure_ascii=False),
        schema=SCHEMA,
        max_tokens=500,
    )
    if not isinstance(output, dict):
        return None
    source = " ".join([text, said_before, speaker, prior.get("speaker", "")])
    kept = [_clean(output.get("restatement"), MAX_RESTATEMENT_CHARS)]
    facts = output.get("facts")
    if isinstance(facts, list):
        kept += [_clean(f, MAX_FACT_CHARS) for f in facts[:MAX_FACTS]]
    lines = list(
        dict.fromkeys(
            line for line in kept if line and grounded(line, source) and not _echo(line, text)
        )
    )
    relations = _relations(output.get("relations"), source)
    if not lines and not relations:
        return None
    return Restated(" ".join(lines), relations)


def _echo(line: str, turn: str) -> bool:
    """A line that only repeats the turn's own words adds nothing to its key."""
    said = set(_WORD.findall(turn.casefold()))
    words = _WORD.findall(line.casefold())
    return bool(words) and sum(w in said for w in words) / len(words) >= ECHO_SHARE


def _relations(value: object, source: str) -> list[tuple[str, str, str]]:
    """Relations whose person and object words all occur in what the model was shown (a
    computed date excepted), with a short snake_case predicate."""
    if not isinstance(value, list):
        return []
    lowered = source.casefold()
    out: list[tuple[str, str, str]] = []
    for item in value[: 2 * MAX_RELATIONS]:
        if len(out) == MAX_RELATIONS:
            break
        if not isinstance(item, dict):
            continue
        subject = _clean(item.get("subject"), MAX_RELATION_CHARS)
        predicate = "_".join(str(item.get("predicate", "")).casefold().split())
        obj = _clean(item.get("object"), MAX_RELATION_CHARS)
        if not (subject and obj and _PREDICATE.fullmatch(predicate)):
            continue
        if obj.casefold().strip(".") in _PLACEHOLDERS or subject.casefold() in _PLACEHOLDERS:
            continue
        words = _WORD.findall(_DATE.sub(" ", f"{subject} {obj}").casefold())
        if (subject, predicate, obj) not in out and all(
            word in lowered for word in words if word not in _FUNCTION
        ):
            out.append((subject, predicate, obj))
    return out


def grounded(line: str, source: str) -> bool:
    """Every number and every capitalised name in ``line`` occurs in ``source``, apart from
    the dates the model was asked to compute."""
    lowered = source.casefold()
    undated = _DATE.sub(" ", line)
    if any(n not in source for n in _NUMBER.findall(undated)):
        return False
    return all(name.casefold() in lowered for name in _NAME.findall(undated))


def _clean(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text if 0 < len(text) <= limit else ""
