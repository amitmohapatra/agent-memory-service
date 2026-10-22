"""The ReAct arm asks the questions the h.wrap arm was asked, of the same store: its rows are
the h.wrap arm's retrieval rows without their answers.

    python -m benchmark.memvs.react_rows OUT DATASET
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmark.memvs.model import SystemName

ANSWER_FIELDS = (
    "answer",
    "answer_prompt_tokens",
    "answer_completion_tokens",
    "answer_latency_ms",
    "judge",
)


def main() -> None:
    out, dataset = Path(sys.argv[1]), sys.argv[2]
    source = out / f"{SystemName.TRELLIS.value}.{dataset}"
    target = out / f"{SystemName.TRELLIS_REACT.value}.{dataset}"
    rows = [
        json.loads(line)
        for line in source.with_name(source.name + ".rows.jsonl").read_text().splitlines()
    ]
    with target.with_name(target.name + ".rows.jsonl").open("w") as f:
        for row in rows:
            fresh = row | dict.fromkeys(ANSWER_FIELDS) | {"system": SystemName.TRELLIS_REACT.value}
            f.write(json.dumps(fresh, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
