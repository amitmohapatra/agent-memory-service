#!/usr/bin/env bash
# The upgrade path: a store written by one code tree, then served by another, in place.
#
#   benchmark/capabilities/stress/run_upgrade.sh <name> <old checkout> <new checkout> <out dir> \
#       [backfill shell command] [-- run.py args...]
#
# 1. the old checkout serves a private stack and the earlier runs are written (--phase prior);
# 2. its API and worker stop; the SAME database is migrated to the new checkout's head;
# 3. the backfill command runs (optional; in the new checkout, with the service's environment:
#    MEMORY__DATABASE__URL, MEMORY__SEARCH__QDRANT_URL, ... point at the stack);
# 4. the new checkout serves the same database, Qdrant, Redis and OpenFGA; the final run is
#    written and its questions asked (--phase final), scored as run_isolated.sh scores them.
#
# Writes <out dir>/<name>.json (the final phase: records + summary, meta.upgraded_from),
# <name>.md, <name>.prior.json and <name>.log. Same defaults as run_isolated.sh (run date
# 2026-10-18, seed, 6 earlier runs), so <name>.json pairs with the run_isolated.sh baselines.
set -euo pipefail
NAME=${1:?name}; OLD=$(realpath "${2:?old checkout}"); NEW=$(realpath "${3:?new checkout}")
OUT=$(realpath -m "${4:?out dir}"); shift 4
BACKFILL=""
if [ $# -gt 0 ] && [ "$1" != "--" ]; then BACKFILL=$1; shift; fi
[ "${1:-}" = "--" ] && shift
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$OUT"
export CAPEVAL_ROOT=$OLD CAPEVAL_UPGRADE_ROOT=$NEW CAPEVAL_UPGRADE_BACKFILL=$BACKFILL
export CAPEVAL_PY=${CAPEVAL_PY:-/home/user/agent-memory-service/.venv/bin/python}
export CAPEVAL_DB=${CAPEVAL_DB:-capeval_stress}
export CAPEVAL_WORK=${CAPEVAL_WORK:-/var/tmp/capeval_stress}
export CAPEVAL_LOGS=${CAPEVAL_LOGS:-/var/tmp/capeval_stress_logs_$NAME}
export CAPEVAL_REDIS_DB=${CAPEVAL_REDIS_DB:-6}
export CAPEVAL_API_PORT=${CAPEVAL_API_PORT:-8491} CAPEVAL_METRICS_PORT=${CAPEVAL_METRICS_PORT:-9491}
export CAPEVAL_QDRANT_HTTP=${CAPEVAL_QDRANT_HTTP:-6833} CAPEVAL_QDRANT_GRPC=${CAPEVAL_QDRANT_GRPC:-6834}
export CAPEVAL_FGA_HTTP=${CAPEVAL_FGA_HTTP:-8591} CAPEVAL_FGA_GRPC=${CAPEVAL_FGA_GRPC:-8593}
RUN_DATE=${RUN_DATE:-2026-10-18}
ARGS=$(printf ' %q' --run-date "$RUN_DATE" "$@")
export CAPEVAL_UPGRADE_CMD="$CAPEVAL_PY -m benchmark.capabilities.stress.run --phase final --out $(printf %q "$OUT/$NAME.json")$ARGS"
cd "$HERE"
t0=$(date +%s)
set +e
benchmark/capabilities/isolated_stack.sh "$CAPEVAL_PY" -m benchmark.capabilities.stress.run --phase prior \
  --run-date "$RUN_DATE" --out "$OUT/$NAME.prior.json" "$@" > "$OUT/$NAME.log" 2>&1
rc=$?
set -e
echo "$NAME rc=$rc old=$(git -C "$OLD" rev-parse --short HEAD) new=$(git -C "$NEW" rev-parse --short HEAD)$(git -C "$NEW" diff --quiet || echo +dirty) backfill=${BACKFILL:-none} wall=$(( $(date +%s) - t0 ))s $(grep -E '^wrote .*pass' "$OUT/$NAME.log" | tail -1)"
exit $rc
