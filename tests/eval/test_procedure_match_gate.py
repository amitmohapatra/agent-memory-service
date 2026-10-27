"""The learned-skill match on the frozen multilingual encoder (``models``): a task worded
otherwise finds the procedure learned for it, an unrelated one does not.

The capability eval's MFA skill (``08_procedural_skill.json``): the procedure each language
learns from its own task, matched against that language's ten phrasings of the request and
against the 45 queries of the other procedural scenarios, the ten phrasings each. The floor is
``FROZEN_MODELS.dense_ml.procedure_cosine``; the counts are what it was calibrated at.
"""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_service.adapters.models.sparse import Bm25SparseEncoder
from memory_service.config.constants import FROZEN_MODELS, local_model_path
from memory_service.domain.tools import StoredProcedure
from memory_service.modules.rag.indexer import Indexer
from memory_service.modules.rag.spaces import DenseSpaces
from memory_service.modules.tools.matching import ProcedureMatcher
from memory_service.modules.tools.patterns import task_pattern
from memory_service.ports.search import VectorName
from tests.support_models import NO_WEIGHTS, requires_onnxruntime

pytestmark = [pytest.mark.eval, pytest.mark.models]

DATA = Path(__file__).resolve().parents[2] / "benchmark/capabilities/data/08_procedural_skill.json"
SKILL = "procedure-it-mfa-skill"
#: phrasings of ten that the floor keeps, per language, as calibrated
MATCHED_AT_LEAST = {"en": 9, "es": 8, "hi": 7, "ar": 6, "de": 2}
UNRELATED = {
    "en": ["reset a store's POS terminal", "order more stock for S12"],
    "es": ["reinicia el terminal punto de venta de la tienda", "pide más stock para S12"],
    "hi": ["स्टोर का POS टर्मिनल रीसेट करो", "S12 के लिए और स्टॉक मँगाओ"],
    "ar": ["أعد تشغيل جهاز نقطة البيع في المتجر", "اطلب مخزونًا إضافيًا لـ S12"],
    "de": ["Setz das Kassenterminal der Filiale zurück", "Bestell mehr Ware für S12"],
}


@pytest.fixture(scope="module")
def matcher():
    requires_onnxruntime()
    spec = FROZEN_MODELS.dense_ml
    if local_model_path(spec.local_dir) is None and spec.model_path is None:
        pytest.skip(f"{NO_WEIGHTS} ({spec.local_dir} missing)")
    from memory_service.adapters.models.embeddings import load_dense

    encoder = load_dense(spec, threads=1)
    spaces = DenseSpaces.single(encoder, VectorName.DENSE_ML)
    yield ProcedureMatcher(Indexer(None, None, spaces, Bm25SparseEncoder(), None))  # type: ignore[arg-type]
    encoder.close()


def _scenarios() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))["scenarios"]


@pytest.mark.parametrize("lang", list(MATCHED_AT_LEAST))
async def test_the_skill_is_found_for_its_rewordings_and_not_for_another_task(
    matcher: ProcedureMatcher, lang: str
) -> None:
    skill = next(s for s in _scenarios() if s["id"] == SKILL)
    task = skill["sessions"][0]["tool_runs"][0]["task"][lang]
    procedure = StoredProcedure(
        procedure_id="proc_mfa", tenant_id="t", scope_key="k", pattern=task_pattern(task)
    )
    phrasings = [p["text"][lang] for p in skill["queries"][0]["phrasings"]]
    matched = [t for t in phrasings if await matcher.rank(t, [procedure], 3) == [procedure]]
    assert len(matched) >= MATCHED_AT_LEAST[lang], set(phrasings) - set(matched)
    others = [
        p["text"][lang]
        for s in _scenarios()
        if s["id"] != SKILL
        for q in s["queries"]
        for p in q["phrasings"]
        if lang in p["text"]
    ] + UNRELATED[lang]
    wrong = [t for t in others if await matcher.rank(t, [procedure], 3)]
    assert wrong == []
