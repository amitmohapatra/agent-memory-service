#!/usr/bin/env bash
# One system on one workload: ingest, then (a new process, after a restart) query, then answer.
#
#   benchmark/memvs/bench.sh <system> <dataset> [answer|react|answer-only]
#
# ``react`` (this service only): answer through h.wrap, then again through h.wrap(ReAct) from
# the same store. ``answer-only`` (any other system): answer from the rows an earlier query
# phase wrote; the store is not needed.
#
# Every library runs in the interpreter that has it (MEMVS_PY_HARNESS: agent-harness's, with the
# Agents SDK and Deep Agents' dependencies; MEMVS_PY_LIBS: Mem0, LangMem, LangGraph's SqliteStore,
# Graphiti; MEMVS_PY_CREWAI: CrewAI's). This service (trellis, trellis_react) runs on a store of
# its own (benchmark/capabilities/isolated_stack.sh): its API and worker are stopped after the
# ingest and started again on the same stores before the questions (the upgrade path, to this
# same checkout), and the store is dropped when the run ends.
#
# Environment: MEMVS_OUT (results), MEMVS_STORES (per-system stores, removed after the run),
# MEMVS_WORKLOADS, MEMVS_EMBED_URL and MEMVS_EMBED_PID (the shared encoder), MEMVS_LLM_URL,
# MEMVS_LLM_KEY, MEMVS_LLM_MODEL (the gateway, for answers and model-backed systems).
set -euo pipefail
SYSTEM=${1:?system}; DATASET=${2:?dataset}; ANSWER=${3:-}
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
: "${MEMVS_OUT:?}" "${MEMVS_STORES:?}" "${MEMVS_WORKLOADS:?}" "${MEMVS_EMBED_URL:?}"
WORKLOAD="$MEMVS_WORKLOADS/$DATASET.json"
STORE="$MEMVS_STORES/$SYSTEM.$DATASET"
export PYTHONPATH="${MEMVS_PYTHONPATH:+$MEMVS_PYTHONPATH:}$HERE"
export MEM0_TELEMETRY=False GRAPHITI_TELEMETRY_ENABLED=false CREWAI_DISABLE_TELEMETRY=true \
  CREWAI_TRACING_ENABLED=false OTEL_SDK_DISABLED=true ANONYMIZED_TELEMETRY=False

case "$SYSTEM" in
  trellis|trellis_react|openai_session) PY=${MEMVS_PY_HARNESS:?} ;;
  crewai|crewai_llm) PY=${MEMVS_PY_CREWAI:?} ;;
  *) PY=${MEMVS_PY_LIBS:?} ;;
esac
phase() { "$PY" -m benchmark.memvs.run "$1" --system "${SYSTEM}" --workload "$WORKLOAD" --out "$MEMVS_OUT" --store "$STORE"; }
export -f phase
export PY SYSTEM WORKLOAD STORE DATASET

if [ "$ANSWER" = answer-only ]; then
  phase answer
  exit 0
fi
rm -rf "$STORE"
case "$SYSTEM" in
  trellis|trellis_react)
    # the service's processes are measured with the run: API, worker and Qdrant (from the stack's
    # manifest; the worker and API are new pids after the restart)
    export CAPEVAL_WORK=${CAPEVAL_WORK:-/var/tmp/capeval_memvs}
    pids='a=$(grep ^api_pid= $CAPEVAL_WORK.manifest | tail -1 | cut -d= -f2);
          w=$(grep ^worker_pid= $CAPEVAL_WORK.manifest | tail -1 | cut -d= -f2);
          q=$(grep ^qdrant_pid= $CAPEVAL_WORK.manifest | tail -1 | cut -d= -f2);
          export MEMVS_PIDS="api=$a,worker=$w,qdrant=$q"'
    # ReAct answers from the same store: the rows of the h.wrap system, without its answers
    react='"$PY" -m benchmark.memvs.react_rows "$MEMVS_OUT" "$DATASET" &&
           SYSTEM=trellis_react phase answer'
    after="$pids; phase query${ANSWER:+ && phase answer}"
    [ "$ANSWER" = react ] && after="$after && $react"
    CAPEVAL_UPGRADE_ROOT="$HERE" CAPEVAL_UPGRADE_CMD="$after" \
      "$HERE/benchmark/capabilities/isolated_stack.sh" bash -c "$pids; phase ingest"
    ;;
  *)
    export MEMVS_PIDS="embed=${MEMVS_EMBED_PID:?}"
    phase ingest
    phase query
    [ -n "$ANSWER" ] && phase answer
    ;;
esac
rm -rf "$STORE"
