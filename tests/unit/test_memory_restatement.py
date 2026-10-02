"""A turn restated to stand on its own (``modules.memory.restatement``, ADR 0027): what the
model is asked, which of its lines are kept, and how the restatement joins the index key."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.memory.restatement import USE, grounded, restate
from memory_service.modules.rag.indexer import memory_index_text
from memory_service.ports.intelligence import MemoryCandidate

pytestmark = pytest.mark.unit

SAID = datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
BEFORE = {"source_id": "m1", "speaker": "melanie", "text": "Did you go camping last weekend?"}
TURN = "Yes, with my kids at the lake, we loved it."


class _Assist:
    def __init__(self, output: Any, *, wanted: bool = True) -> None:
        self.output, self.wanted, self.calls = output, wanted, []

    def wants(self, use: str) -> bool:
        return self.wanted and use == USE

    async def structured(self, use: str, **kwargs: Any) -> Any:
        self.calls.append((use, kwargs))
        return self.output


async def test_the_model_sees_the_turn_the_turn_before_it_and_the_day() -> None:
    assist = _Assist({"restatement": "", "facts": []})
    await restate(assist, text=TURN, speaker="caroline", said_at=SAID, before=BEFORE)  # type: ignore[arg-type]
    [(use, kwargs)] = assist.calls
    sent = json.loads(kwargs["user"])
    assert use == USE and kwargs["schema"]["required"] == ["restatement", "facts"]
    assert sent["said_on"] == "2023-05-08" and sent["weekday"] == "Monday"
    assert sent["previous_turn"] == {"speaker": "melanie", "text": BEFORE["text"]}
    assert sent["turn"] == {"speaker": "caroline", "text": TURN}


async def test_grounded_lines_are_kept_and_invented_ones_dropped() -> None:
    assist = _Assist(
        {
            "restatement": "Caroline went camping at the lake with her kids around 2023-05-06.",
            "facts": [
                "Caroline's kids loved camping at the lake.",
                "Caroline went camping with Jonathan.",  # nobody named Jonathan was mentioned
                "Caroline camped for 3 nights.",  # no number was said
            ],
        }
    )
    said = await restate(assist, text=TURN, speaker="caroline", said_at=SAID, before=BEFORE)  # type: ignore[arg-type]
    assert said == (
        "Caroline went camping at the lake with her kids around 2023-05-06. "
        "Caroline's kids loved camping at the lake."
    )


async def test_no_model_no_output_and_nothing_to_say_are_none() -> None:
    off = _Assist({"restatement": "x", "facts": []}, wanted=False)
    assert await restate(off, text=TURN, speaker="c", said_at=SAID, before=None) is None  # type: ignore[arg-type]
    assert off.calls == []
    failed = _Assist(None)
    assert await restate(failed, text=TURN, speaker="c", said_at=SAID, before=None) is None  # type: ignore[arg-type]
    empty = _Assist({"restatement": "", "facts": []})
    assert await restate(empty, text="Thanks!", speaker="c", said_at=SAID, before=None) is None  # type: ignore[arg-type]


def test_a_computed_date_is_grounded_but_an_invented_number_is_not() -> None:
    source = "caroline: I went to a support group yesterday"
    assert grounded("Caroline went to a support group on 2023-05-07, a Sunday.", source)
    assert not grounded("Caroline went to 2 support groups.", source)
    assert not grounded("Caroline went with Melanie to a support group.", source)


def test_the_restatement_is_appended_to_the_turns_own_key() -> None:
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="caroline", workspace_id="ws")
    candidate = MemoryCandidate(
        content=TURN,
        memory_type=MemoryType.EPISODIC,
        lifetime=Lifetime.LONG_TERM,
        category="verbatim_turn",
        evidence=[EvidenceRef(source_type="message", source_id="m2", observed_at=SAID)],
        restatement="Caroline went camping at the lake with her kids.",
    )
    memory = build_memory(candidate, ctx, now=SAID)
    assert memory.content == TURN  # the turn itself is kept verbatim
    key = memory_index_text(memory)
    assert key.endswith(f"{TURN}\nCaroline went camping at the lake with her kids.")
    plain = build_memory(candidate.model_copy(update={"restatement": None}), ctx, now=SAID)
    assert memory_index_text(plain).endswith(TURN)
