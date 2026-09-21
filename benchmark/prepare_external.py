"""Download the external benchmark corpus into ``benchmark/data`` (git-ignored).

Kept out of the repository on purpose: benchmark data is not ours to vendor, and a corpus
checked in beside the code is one more thing that can drift from its upstream. The result
file records the dataset name, split and source so a score can always be traced back.

    uv run python -m benchmark.prepare_external
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "benchmark" / "data" / "scifact.json"
#: BEIR SciFact: 5,183 abstracts, 300 test queries with graded relevance judgements.
DATASET = "BeIR/scifact"
QRELS = "BeIR/scifact-qrels"


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.stderr.write("the `datasets` package is required: uv sync --all-extras --dev\n")
        return 2

    corpus = load_dataset(DATASET, "corpus", split="corpus")
    queries = {q["_id"]: q["text"] for q in load_dataset(DATASET, "queries", split="queries")}
    relevant: dict[str, list[str]] = {}
    for row in load_dataset(QRELS, split="test"):
        if int(row["score"]) > 0:
            relevant.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))

    payload = {
        "name": DATASET,
        "split": "test",
        "source": f"https://huggingface.co/datasets/{DATASET}",
        "licence": "CC BY-NC 4.0 — benchmark use",
        "corpus": [{"id": d["_id"], "title": d["title"], "text": d["text"]} for d in corpus],
        "queries": [
            {"id": qid, "text": queries[qid], "relevant": docs}
            for qid, docs in sorted(relevant.items())
            if qid in queries
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    sys.stdout.write(
        f"{len(payload['corpus']):,} documents and {len(payload['queries'])} queries -> {OUT}\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
