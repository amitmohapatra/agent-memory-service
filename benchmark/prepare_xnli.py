"""Prepare a deterministic, aligned XNLI screen before looking at model predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark.common import file_sha256

LABELS = ("entailment", "neutral", "contradiction")
SELECTION_SEED = "xnli-screen-v1"


def select_rows(rows: list[dict], per_label: int) -> list[int]:
    if per_label < 1:
        raise ValueError("per_label must be positive")
    selected = []
    for label in range(len(LABELS)):
        indices = [index for index, row in enumerate(rows) if row["label"] == label]
        if len(indices) < per_label:
            raise ValueError(f"Insufficient examples for {LABELS[label]}")
        indices.sort(
            key=lambda index: hashlib.sha256(f"{SELECTION_SEED}:{index}".encode()).digest()
        )
        selected.extend(indices[:per_label])
    return sorted(selected)


def aligned_pairs(rows: list[dict], selected: list[int]) -> list[dict]:
    records = []
    for index in selected:
        row = rows[index]
        hypotheses = dict(
            zip(row["hypothesis"]["language"], row["hypothesis"]["translation"], strict=True)
        )
        if set(hypotheses) != set(row["premise"]) or "en" not in hypotheses:
            raise ValueError("Premise and hypothesis language sets must align and include English")
        for language in sorted(hypotheses):
            modes = ("same",) if language == "en" else ("same", "cross")
            for mode in modes:
                records.append(
                    {
                        "id": f"{index}:{language}:{mode}",
                        "language": language,
                        "mode": mode,
                        "premise": row["premise"][language],
                        "hypothesis": hypotheses[language if mode == "same" else "en"],
                        "label": LABELS[row["label"]],
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--per-label", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    source = pq.ParquetFile(args.parquet)
    metadata = json.loads(source.schema_arrow.metadata[b"huggingface"])
    if tuple(metadata["info"]["features"]["label"]["names"]) != LABELS:
        raise ValueError("Unexpected XNLI label order")
    rows = source.read(use_threads=False).to_pylist()
    selected = select_rows(rows, args.per_label)
    output = {
        "dataset": "facebook/xnli",
        "revision": args.revision,
        "split": "test",
        "source_sha256": file_sha256(args.parquet),
        "total_original_rows": len(rows),
        "selection": f"{args.per_label} examples per label, smallest SHA256({SELECTION_SEED}:<row index>), selected before model predictions",
        "selected_ids": selected,
        "records": aligned_pairs(rows, selected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
