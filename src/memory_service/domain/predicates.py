"""How many values a predicate may hold at once.

One closed vocabulary, one rule, three readers: consolidation supersedes a *single-valued*
slot when a new value arrives (``modules.memory.native``), belief derivation only aggregates
*multi-valued* ones (``modules.memory.landing``), and the renderer groups the multi-valued
ones into a single dated line (``domain.context_bundle``). The rule lived twice as a private
in the write path; the renderer is in the domain and cannot import the write path, so the
vocabulary belongs here.

A slot is single-valued when a person has exactly one of it at a time: you live in one city
and hold one job title, so a new value replaces the old. ``likes``, ``visited`` or
``participated_in`` accumulate, and replacing them would delete history.
"""

from __future__ import annotations

#: Slots a subject holds exactly one of at a time. Whitespace-separated for editing; the
#: frozenset below is the vocabulary.
_SINGLE_VALUED_SLOTS = """
name timezone time_zone role title team email location birthday manager company employer
city country language pronouns phone department working_hours handle username works_at
lives_in favourite
"""

SINGLE_VALUED: frozenset[str] = frozenset(_SINGLE_VALUED_SLOTS.split())

#: ``favourite_colour``, ``favourite_food``: one favourite per category, so the whole family
#: is single-valued without enumerating it.
_SINGLE_VALUED_PREFIX = "favourite_"


def is_single_valued(predicate: str | None) -> bool:
    """True when a new value for this predicate replaces the previous one."""
    return bool(predicate) and (
        predicate in SINGLE_VALUED or predicate.startswith(_SINGLE_VALUED_PREFIX)
    )


def is_multi_valued(predicate: str | None) -> bool:
    """True when values accumulate, so several memories on this predicate are all current.

    An absent predicate is neither: there is nothing to accumulate.
    """
    return bool(predicate) and not is_single_valued(predicate)


def predicate_label(predicate: str) -> str:
    """``works_at`` -> ``works at``: the slot as prose, for a line a model reads."""
    return predicate.replace("_", " ")
