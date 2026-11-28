"""``benchmark.locomo_rescore`` against the model client the service builds: the scorer calls
only what ``EndpointLLM`` has (it called a ``model_for`` the client no longer had, and every
rescore stopped on its first row)."""

from __future__ import annotations

from unittest.mock import create_autospec

import benchmark.locomo_rescore as rescorer
import pytest

from memory_service.adapters.models.llm import EndpointLLM


@pytest.mark.asyncio
async def test_a_rescore_names_its_judge_with_the_clients_own_call(monkeypatch) -> None:
    llm = create_autospec(EndpointLLM, instance=True)
    llm.enabled = True
    llm.resolve_model.return_value = "judge-model"

    class Container:
        def __init__(self) -> None:
            self.llm = llm

        async def close(self) -> None:
            pass

    async def build_container(*args, **kwargs):
        return Container()

    async def judge(llm, question, gold, got, *, ruler="strict"):
        return {"correct": gold in got, "abstained": False, "reason": ""}

    monkeypatch.setattr(rescorer, "build_container", build_container)
    monkeypatch.setattr(rescorer, "_judge", judge)
    result = {
        "judged": True,
        "records": [
            {
                "category": "single_hop",
                "question": "Dog?",
                "answer": "Biscuit",
                "judged": {"produced": "Biscuit."},
            },
        ],
    }
    out = await rescorer.rescore(result, "strict", 1e9)
    assert out["judge_model"] == "judge-model" and out["records"][0]["judge_model"] == "judge-model"
    assert out["by_category"] == {"single_hop": {"n": 1, "score": 1.0}}
