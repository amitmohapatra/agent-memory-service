"""Public benchmarks: BEIR subset through the retrieval engine; LongMemEval and LoCoMo
through the memory pipeline with Bifrost-generated answers and the benchmarks' own graders.
See ``benchmark.public.cli`` for the command line."""

from benchmark.public.cli import main

__all__ = ["main"]
