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
policy, and whose key pays is the bound identity's.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Final

from memory_service.modules.llm.assist import LLMAssist

USE: Final = "memory_restatement"
MAX_TURN_CHARS: Final = 1500
MAX_CONTEXT_CHARS: Final = 600
MAX_RESTATEMENT_CHARS: Final = 400
MAX_FACTS: Final = 3
MAX_FACT_CHARS: Final = 200

SYSTEM: Final = (
    "You restate one turn of a conversation so it makes sense on its own months later. "
    "Use people's names instead of I, you, he, she or they. Replace relative times "
    "(yesterday, last week, next month, two days ago) with absolute dates computed from the "
    "date the turn was said. Keep who, what, where, with whom, when and why. Do not add "
    "anything the turn does not say. Then list up to three short facts the turn states. If the "
    "turn says nothing worth remembering (a greeting, thanks), return an empty restatement."
)
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "restatement": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["restatement", "facts"],
    "additionalProperties": False,
}
#: a date the model computed: ISO, or a month or weekday name, or a four-digit year
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july"
    r"|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday"
    r"|saturday|sunday)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z'-]+")


async def restate(
    assist: LLMAssist,
    *,
    text: str,
    speaker: str,
    said_at: datetime,
    before: dict[str, str] | None,
) -> str | None:
    """The turn's restatement with its facts, ready to append to its index key; ``None`` when
    the model was not consulted or said nothing that passed the checks."""
    if not assist.wants(USE) or not text.strip():
        return None
    prior = before or {}
    said_before = prior.get("text", "")[:MAX_CONTEXT_CHARS]
    payload = {
        "said_on": said_at.date().isoformat(),
        "weekday": said_at.strftime("%A"),
        "previous_turn": {"speaker": prior.get("speaker", ""), "text": said_before}
        if said_before
        else None,
        "turn": {"speaker": speaker, "text": text[:MAX_TURN_CHARS]},
    }
    output = await assist.structured(
        USE,
        system=SYSTEM,
        user=json.dumps(payload, ensure_ascii=False),
        schema=SCHEMA,
        max_tokens=400,
    )
    if not isinstance(output, dict):
        return None
    source = " ".join([text, said_before, speaker, prior.get("speaker", "")])
    kept = [_clean(output.get("restatement"), MAX_RESTATEMENT_CHARS)]
    facts = output.get("facts")
    if isinstance(facts, list):
        kept += [_clean(f, MAX_FACT_CHARS) for f in facts[:MAX_FACTS]]
    lines = list(dict.fromkeys(line for line in kept if line and grounded(line, source)))
    return " ".join(lines) or None


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
