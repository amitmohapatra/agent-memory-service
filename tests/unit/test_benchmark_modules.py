"""The embedding / reranker benchmark modules: candidate matrices, verdict selection on
synthetic rows, and one end-to-end run of the embedding benchmark with the hash stand-in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark import embedding, reranker

from tests.conftest import PG_AVAILABLE


def _row(recall: float, egr: float, **extra: object) -> dict[str, object]:
    return {
        "quality": {"critical_recall_at_k": recall, "critical_evidence_group_recall": egr},
        **extra,
    }


def _emb_row(recall: float, egr: float, *, q_p95: float, dim: int) -> dict[str, object]:
    return _row(recall, egr, embed_ms={"query_p95": q_p95}, dimension=dim)


def _rr_row(recall: float, egr: float, *, k: int, p95: float) -> dict[str, object]:
    return _row(recall, egr, candidate_k=k, rerank_ms={"p95": p95})


@pytest.mark.unit
class TestVerdicts:
    def test_embedding_quality_first_then_speed(self) -> None:
        rows = {
            "large/torch": _emb_row(1.0, 1.0, q_p95=30.0, dim=768),
            "small/torch": _emb_row(1.0, 1.0, q_p95=12.0, dim=384),
            "small/onnx": _emb_row(0.9, 1.0, q_p95=4.0, dim=384),
            "large/openvino": {"skipped": "DependencyUnavailable: no openvino"},
        }
        verdict = embedding.pick_default(rows, embedding.embedding_cost)
        assert verdict["default"] == "small/torch"
        assert verdict["eligible"] == ["small/torch", "large/torch"]
        assert verdict["best"] == {
            "critical_recall_at_k": 1.0,
            "critical_evidence_group_recall": 1.0,
        }

    def test_egr_breaks_recall_ties_before_speed(self) -> None:
        rows = {
            "fast-incomplete": _emb_row(1.0, 0.8, q_p95=1.0, dim=64),
            "slow-complete": _emb_row(1.0, 1.0, q_p95=50.0, dim=768),
        }
        assert embedding.pick_default(rows, embedding.embedding_cost)["default"] == "slow-complete"

    def test_dimension_breaks_latency_ties(self) -> None:
        rows = {
            "b-768": _emb_row(1.0, 1.0, q_p95=10.0, dim=768),
            "a-384": _emb_row(1.0, 1.0, q_p95=10.0, dim=384),
        }
        assert embedding.pick_default(rows, embedding.embedding_cost)["default"] == "a-384"

    def test_all_skipped_has_no_default(self) -> None:
        rows = {"x": {"skipped": "ImportError: torch"}, "y": {"skipped": "OSError: weights"}}
        verdict = embedding.pick_default(rows, embedding.embedding_cost)
        assert verdict["default"] is None
        assert verdict["eligible"] == []

    def test_reranker_cheapest_k_keeping_best_quality(self) -> None:
        rows = {
            "sentence_transformers@k15": _rr_row(1.0, 0.9, k=15, p95=20.0),
            "sentence_transformers@k20": _rr_row(1.0, 1.0, k=20, p95=25.0),
            "sentence_transformers@k25": _rr_row(1.0, 1.0, k=25, p95=30.0),
            "onnx@k20": _rr_row(1.0, 1.0, k=20, p95=15.0),
            "lexical@k15": _rr_row(0.8, 0.8, k=15, p95=0.5),
            "disabled@k15": _rr_row(0.7, 0.7, k=15, p95=0.0),
        }
        verdict = embedding.pick_default(rows, reranker.reranker_cost)
        assert verdict["default"] == "onnx@k20"
        assert verdict["eligible"] == [
            "onnx@k20",
            "sentence_transformers@k20",
            "sentence_transformers@k25",
        ]


@pytest.mark.unit
class TestCandidates:
    def test_embedding_matrix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MEMORY_MODELS_DIR", str(tmp_path))
        full = embedding.candidates(threads=3)
        # derived from the catalogue, so adding a candidate model does not break the test
        assert len(full) == len(embedding.MODELS) * len(embedding.BACKENDS)
        assert {c.provider for c in full.values()} == set(embedding.BACKENDS)
        assert {c.dimension for c in full.values()} == {d for _, d in embedding.MODELS.values()}
        assert all(c.threads == 3 for c in full.values())
        large = full["granite-embedding-english-r2/onnx"]
        assert large.model == "ibm-granite/granite-embedding-english-r2"
        assert large.model_path == str(tmp_path / "granite-embedding-english-r2")

        quick = embedding.candidates(quick=True)
        assert len(quick) == len(embedding.MODELS)
        assert {c.provider for c in quick.values()} == {"sentence_transformers"}

        stand_in = embedding.candidates(stand_in=True)
        assert list(stand_in) == [embedding.STAND_IN]
        assert stand_in[embedding.STAND_IN].provider == "hash"

    def test_reranker_matrix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MEMORY_MODELS_DIR", str(tmp_path))
        full = reranker.candidates()
        assert len(full) == 12
        assert sorted({c.candidate_k for c in full.values()}) == [15, 20, 25]
        ce = full["onnx@k25"]
        assert ce.provider == "onnx"
        assert ce.candidate_k == 25
        assert ce.model_path == str(tmp_path / "ms-marco-MiniLM-L6-v2")
        assert full["lexical@k15"].model_path is None
        assert {c.provider for c in reranker.candidates(stand_in=True).values()} == {
            "lexical",
            "disabled",
        }

    def test_sample_documents_cycles_to_batch_size(self) -> None:
        from memory_service.modules.evaluation.golden import GoldenSet

        docs = embedding.sample_documents(GoldenSet.load(embedding.GOLDEN))
        assert len(docs) == embedding.BATCH_DOCUMENTS
        assert all(len(d) > 40 for d in docs)

    def test_with_models_replaces_only_the_given_sections(self) -> None:
        from memory_service.config.settings import EmbeddingSettings, Settings

        base = Settings(models={"reranker": {"provider": "lexical", "candidate_k": 25}})
        cfg = EmbeddingSettings(provider="onnx", model="m", model_path="/w/m", dimension=768)
        merged = embedding.with_models(base, embedding=cfg)
        assert merged.models.embedding == cfg
        assert merged.models.reranker.provider == "lexical"
        assert merged.models.reranker.candidate_k == 25

    def test_stand_in_base_keeps_infrastructure_and_swaps_models(self) -> None:
        from memory_service.config.settings import Settings

        base = Settings(
            search={"provider": "qdrant", "qdrant_url": "http://q:6333"},
            models={"embedding": {"provider": "onnx", "dimension": 768}},
        )
        stand_in = embedding.stand_in_base(base)
        assert stand_in.search.provider == "qdrant"
        assert stand_in.models.embedding.provider == "hash"
        assert stand_in.models.embedding.dimension == 64
        assert stand_in.models.reranker.provider == "lexical"


@pytest.mark.integration
@pytest.mark.skipif(not PG_AVAILABLE, reason="PostgreSQL not reachable at MEMORY__DATABASE__URL")
def test_embedding_benchmark_stand_in_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for section in ("SEARCH", "CACHE", "AUTHORIZATION", "TASKS", "BLOB"):
        monkeypatch.setenv(f"MEMORY__{section}__PROVIDER", "memory")
    out = tmp_path / "embedding.json"
    embedding.main(["--quick", "--stand-in", "--copies", "1", "--batches", "1", "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == {"quick": True, "stand_in": True}
    assert payload["providers"]["representative"] is False
    row = payload["candidates"][embedding.STAND_IN]
    assert row["fingerprint"] == "hash-v1-d64"
    assert row["dimension"] == 64
    assert row["indexed_points"] > 0
    assert row["embed_ms"]["batch_size"] == embedding.BATCH_DOCUMENTS
    assert row["embed_ms"]["batches"] == 1
    assert 0.0 <= row["quality"]["critical_recall_at_k"] <= 1.0
    assert row["quality_per_ms"] >= 0.0
    assert row["representative"] is False
    assert payload["verdict"]["default"] == embedding.STAND_IN
    assert "git_commit" in payload["provenance"]

    printed = capsys.readouterr().out
    assert f"wrote {out}" in printed
    assert embedding.STAND_IN in printed
    assert f"default -> {embedding.STAND_IN}" in printed
