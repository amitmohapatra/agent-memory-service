"""Download LoCoMo into ``benchmark/data`` (git-ignored).

LoCoMo (Maharana et al.) is the benchmark that matches what this service *is*: very long
multi-session conversations with questions whose answers are spread across sessions, plus
annotated evidence turns. Every other gate here measures document retrieval; this one
measures conversational memory — observe, extract, consolidate, recall.

    uv run python -m benchmark.prepare_locomo
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "benchmark" / "data" / "locomo10.json"
SOURCE = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(SOURCE, timeout=120) as response:  # noqa: S310 - fixed https URL
            payload = json.loads(response.read())
    except Exception as exc:
        sys.stderr.write(f"could not fetch LoCoMo: {type(exc).__name__}: {exc}\n")
        return 1
    OUT.write_text(json.dumps(payload))
    questions = sum(len(c.get("qa", [])) for c in payload)
    sys.stdout.write(f"{len(payload)} conversations, {questions} questions -> {OUT}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
