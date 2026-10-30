#!/usr/bin/env bash
# Run a command against a memory service of its own, then remove what it created.
#
#   benchmark/capabilities/isolated_stack.sh <command...>
#   (make eval-capabilities-isolated MODE=nollm SET=core)
#
# The capability eval writes tens of thousands of memories per run. On a shared stack they
# accumulate: the service has no tenant purge, and forgetting them one by one through the API
# costs an index job each. So a gate run gets a store of its own and drops it when it ends:
#
#   PostgreSQL   a database of its own (CAPEVAL_DB, must start with "capeval"): created,
#                migrated, dropped
#   Qdrant       an instance of its own (the service has no collection-prefix setting) on its
#                own port and storage path; its collections are deleted through the API, then
#                the instance is stopped and the path removed
#   OpenFGA      an in-memory instance of its own
#   Redis        a database number of its own (CAPEVAL_REDIS_DB): must be empty at start,
#                flushed at the end
#   blobs        a directory under WORK
#   the service  memory-api + memory-worker from this checkout's src/, on their own ports
#
# Safety, before anything is created: the database must match ^capeval; the Qdrant ports must
# not be the live ones (6333/6334) nor answer already; WORK must resolve under
# /var/tmp/capeval*; the Redis db must not be 0 nor a reserved one (CAPEVAL_REDIS_RESERVED)
# and must be empty. One run at a time (an exclusive lock beside WORK). The service starts
# with a clean environment (`env -i`: only the variables set here, no .env file: it runs from
# WORK), with both database URLs (MEMORY__DATABASE__URL and __DIRECT_URL, the one migrations,
# the job queue and graph sessions use) on the eval database. Every resource created is
# written to a manifest, and teardown removes only what the manifest names; a process is
# signalled only if its command line is still the one started here.
#
# The command runs with MEMORY_URL, MEMORY_WORKER_METRICS_URL and CAPEVAL_ADMIN_KEY pointing at
# the stack. Teardown runs on any exit. The disk is recorded before and after.
set -euo pipefail

# the harness checkout (this script's); the service runs from CAPEVAL_ROOT when set (another
# branch measured with this harness), else from the harness checkout itself
HARNESS="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$(realpath "${CAPEVAL_ROOT:-$HARNESS}")"
# upgrade path (optional, CAPEVAL_UPGRADE_ROOT): when the command succeeds, stop the service,
# migrate the SAME database to CAPEVAL_UPGRADE_ROOT's head, run CAPEVAL_UPGRADE_BACKFILL (a shell
# command, run in the new checkout with the service's environment) if set, serve the same
# database, Qdrant, Redis and OpenFGA from the new checkout, and run CAPEVAL_UPGRADE_CMD (a shell
# command) against it - a deployed store upgraded in place
UPGRADE_ROOT="${CAPEVAL_UPGRADE_ROOT:+$(realpath "$CAPEVAL_UPGRADE_ROOT")}"
PY="${CAPEVAL_PY:-$ROOT/.venv/bin/python}"
QDRANT_BIN="${QDRANT_BIN:-/var/tmp/qdrant}"
OPENFGA_BIN="${OPENFGA_BIN:-/var/tmp/fga/openfga}"
DB="${CAPEVAL_DB:-capeval}"
REDIS_DB="${CAPEVAL_REDIS_DB:-14}"
REDIS_RESERVED="${CAPEVAL_REDIS_RESERVED:-0 7 8 11 12 13}"
API_PORT="${CAPEVAL_API_PORT:-8191}"
METRICS_PORT="${CAPEVAL_METRICS_PORT:-9481}"
QDRANT_HTTP="${CAPEVAL_QDRANT_HTTP:-6533}"
QDRANT_GRPC="${CAPEVAL_QDRANT_GRPC:-6534}"
FGA_HTTP="${CAPEVAL_FGA_HTTP:-8281}"
FGA_GRPC="${CAPEVAL_FGA_GRPC:-8283}"
READY_SECONDS="${CAPEVAL_READY_SECONDS:-300}"
WORK="$(realpath -m "${CAPEVAL_WORK:-/var/tmp/capeval_stack}")"
LOGS="$(realpath -m "${CAPEVAL_LOGS:-$WORK.logs}")"
MANIFEST="$WORK.manifest"
PGHOST_="${PGHOST:-localhost}" PGUSER_="${PGUSER:-memory}" PGPASSWORD_="${PGPASSWORD:-memory}"

log() { echo "[isolated-stack $(date +%T)] $*" >&2; }
refuse() { log "refused: $*"; exit 2; }
answers() { curl -s -m 2 -o /dev/null "http://127.0.0.1:$1/" 2>/dev/null; }
psql_() { PGPASSWORD="$PGPASSWORD_" psql -h "$PGHOST_" -U "$PGUSER_" -d postgres -Atq "$@"; }
# redis_ DB CMD: one argument-free command (dbsize, flushdb) against local Redis, through
# redis-cli where it is installed, else the protocol over bash's /dev/tcp (CI's runner talks
# to a Redis service container and has no redis-cli)
redis_() {
  if command -v redis-cli >/dev/null 2>&1; then redis-cli -n "$1" "$2"; return; fi
  local reply
  exec 9<>/dev/tcp/127.0.0.1/6379 || return 1
  printf '*2\r\n$6\r\nSELECT\r\n$%d\r\n%s\r\n*1\r\n$%d\r\n%s\r\n' "${#1}" "$1" "${#2}" "$2" >&9
  read -r reply <&9 && [ "${reply%$'\r'}" = +OK ] && read -r reply <&9
  exec 9<&-
  reply="${reply%$'\r'}"
  case "$reply" in :*|+*) echo "${reply:1}" ;; *) return 1 ;; esac
}

# ------------------------------------------------------------------ refuse before creating
[[ "$DB" =~ ^capeval[a-z0-9_]*$ ]] || refuse "database '$DB' does not match ^capeval"
# every path this script may delete, resolved (no "..", no symlink games) before the check
for path in "$WORK" "$LOGS"; do
  case "$path" in /var/tmp/capeval*) ;; *) refuse "path '$path' is not under /var/tmp/capeval*" ;; esac
done
# LOGS is removed and rewritten: never inside (or equal to) the WORK of any isolated run, this
# one's included - a WORK is any /var/tmp/capeval* path with a .lock or .manifest beside it
for marker in /var/tmp/capeval*.lock /var/tmp/capeval*.manifest; do
  [ -e "$marker" ] || continue
  other="$(realpath -m "${marker%.*}")"
  case "$LOGS/" in "$other"/*) refuse "LOGS '$LOGS' is inside the WORK of an isolated run ($other)" ;; esac
done
case "$LOGS/" in "$WORK"/*) refuse "LOGS '$LOGS' is inside this run's WORK" ;; esac
# numbers as numbers: 00 is db 0, 06333 is port 6333
for v in REDIS_DB API_PORT METRICS_PORT QDRANT_HTTP QDRANT_GRPC FGA_HTTP FGA_GRPC READY_SECONDS; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || refuse "$v '${!v}' is not a number"
  printf -v "$v" '%d' "$((10#${!v}))"
done
# the eval database lives on this machine unless a remote one is explicitly allowed
case "$PGHOST_" in
  localhost|127.0.0.1|::1) ;;
  *) [ "${CAPEVAL_ALLOW_REMOTE_PG:-0}" = 1 ] || refuse "PGHOST '$PGHOST_' is not local (CAPEVAL_ALLOW_REMOTE_PG=1 to allow)" ;;
esac
for p in "$QDRANT_HTTP" "$QDRANT_GRPC"; do
  if [ "$p" = 6333 ] || [ "$p" = 6334 ]; then refuse "Qdrant port $p is the live instance's"; fi
done
for r in $REDIS_RESERVED; do
  if [ "$REDIS_DB" = "$r" ]; then refuse "Redis db $REDIS_DB is reserved"; fi
done
for p in "$QDRANT_HTTP" "$API_PORT" "$METRICS_PORT" "$FGA_HTTP"; do
  if answers "$p"; then refuse "port $p already answers: another service is there"; fi
done

if [ "${CAPEVAL_GATEWAY:-0}" = 1 ] && { [ -z "${LLM_BASE_URL:-}" ] || [ -z "${LLM_API_KEY:-}" ]; }; then
  refuse "CAPEVAL_GATEWAY=1 needs LLM_BASE_URL and LLM_API_KEY"
fi

# one isolated stack at a time
exec 8>"$WORK.lock"
flock -n 8 || refuse "another isolated stack holds $WORK.lock"

disk() {
  echo "df_avail_mb=$(df -BM --output=avail / | tail -1 | tr -d ' M')"\
" work_mb=$(du -sm "$WORK" 2>/dev/null | cut -f1 || echo 0)"\
" db_exists=$(psql_ -c "select count(*) from pg_database where datname='$DB'" 2>/dev/null || echo ?)"\
" redis_keys=$(redis_ "$REDIS_DB" dbsize 2>/dev/null || echo ?)"
}
stop() {  # pid marker: SIGTERM, then SIGKILL if it has not exited (a worker draining a job
  # may never exit on its own, and would keep the stack's port for the next run)
  ours "$1" "$2" || return 0
  kill "$1" 2>/dev/null
  for _ in $(seq 1 "${CAPEVAL_KILL_WAIT:-20}"); do ours "$1" "$2" || return 0; sleep 1; done
  ours "$1" "$2" && kill -9 "$1" 2>/dev/null
  return 0
}
ours() {  # pid, a marker its command line or its environment must still carry
  [ -n "${1:-}" ] && [ -r "/proc/$1/cmdline" ] || return 1
  { tr '\0' ' ' < "/proc/$1/cmdline"; tr '\0' ' ' < "/proc/$1/environ" 2>/dev/null; } | grep -qF -- "$2"
}
note() { echo "$1=$2" >> "$MANIFEST"; }
get() { grep "^$1=" "$MANIFEST" 2>/dev/null | tail -1 | cut -d= -f2- || true; }

teardown() {
  local errexit=0
  [[ $- == *e* ]] && errexit=1
  set +e
  log "teardown"
  for name in worker api fga; do
    local pid
    pid="$(get "${name}_pid")"
    stop "$pid" "$(get "${name}_marker")"
  done
  local qpid qhttp
  qpid="$(get qdrant_pid)" qhttp="$(get qdrant_http)"
  if ours "$qpid" "$(get qdrant_marker)"; then
    for c in $(curl -s "http://127.0.0.1:$qhttp/collections" \
        | "$PY" -c "import json,sys; print(' '.join(c['name'] for c in json.load(sys.stdin)['result']['collections']))" 2>/dev/null); do
      curl -s -X DELETE "http://127.0.0.1:$qhttp/collections/$c" >/dev/null
    done
    stop "$qpid" "$(get qdrant_marker)"
  fi
  sleep 2
  local db rdb work
  db="$(get db_created)" rdb="$(get redis_db)" work="$(get work_created)"
  [[ "$db" =~ ^capeval ]] && psql_ -c "drop database if exists $db with (force)" 2>/dev/null
  [ -n "$rdb" ] && redis_ "$rdb" flushdb >/dev/null 2>&1
  work="$(realpath -m "${work:-/nonexistent}")"
  if [[ "$work" == /var/tmp/capeval* ]] && [[ "$LOGS" == /var/tmp/capeval* ]]; then
    mkdir -p "$LOGS" && cp "$work"/*.log "$LOGS"/ 2>/dev/null
    rm -rf "$work"
  fi
  rm -f "$MANIFEST"
  log "disk after: $(disk); logs in $LOGS"
  if [ "$errexit" = 1 ]; then set -e; fi
  return 0
}

if [ -e "$MANIFEST" ]; then
  log "a previous run of this script left $MANIFEST: removing what it created"
  teardown
fi
[ -e "$WORK" ] && refuse "$WORK exists but no manifest names it: not ours to remove"
[ "$(psql_ -c "select count(*) from pg_database where datname='$DB'")" = 0 ] \
  || refuse "database $DB exists but no manifest names it: not ours to drop"
[ "$(redis_ "$REDIS_DB" dbsize)" = 0 ] || refuse "Redis db $REDIS_DB is not empty"

log "disk before: $(disk)"
# Memory this script's processes may still take, in MB: the host's MemAvailable, or less when a
# memory cgroup (v1 or v2, this one or an ancestor) caps them below the host - a container's
# cgroup is often smaller than the machine, and its OOM killer acts on the cgroup's limit. A
# level's room is its limit less what reclaim cannot give back: anonymous memory, shmem and
# mapped files (the page cache beyond that is reclaimed before anything is killed).
mem_avail_mb() {
  local best root rel dir lim held h
  best=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  root=${CAPEVAL_CGROUP_ROOT:-/sys/fs/cgroup}
  for rel in $(awk -F: '$2=="memory"{print "memory" $3} $1=="0"{print $3}' /proc/self/cgroup); do
    dir=$root/${rel#/}
    while [[ $dir == "$root"/* ]]; do
      dir=${dir%/}
      lim=$(cat "$dir/memory.limit_in_bytes" "$dir/memory.max" 2>/dev/null | head -1)
      # v1 (hierarchical totals) or v2 names
      held=$(awk '$1~/^(total_rss|total_shmem|total_mapped_file|anon|shmem|file_mapped)$/{n+=$2; f=1}
        END{if (f) print n}' "$dir/memory.stat" 2>/dev/null)
      # "max" (v2) and the near-2^63 v1 default mean no limit at this level
      if [[ $lim =~ ^[0-9]+$ && $held =~ ^[0-9]+$ ]] && [ "${#lim}" -lt 19 ]; then
        h=$(( (lim - held) / 1048576 ))
        [ "$h" -lt "$best" ] && best=$h
      fi
      dir=$(dirname "$dir")
    done
  done
  echo "$best"
}

# the API and the worker each load the encoders (about 2 GB apiece): wait, bounded, for room
avail=$(mem_avail_mb)
for _ in $(seq 1 "${CAPEVAL_MEM_WAITS:-60}"); do
  [ "$avail" -ge "${CAPEVAL_MIN_MEM_MB:-5000}" ] && break
  log "waiting for memory: ${avail} MB available"
  sleep 10
  avail=$(mem_avail_mb)
done
[ "$avail" -ge "${CAPEVAL_MIN_MEM_MB:-5000}" ] \
  || refuse "only ${avail} MB of memory available: starting the service would invite the OOM killer"

trap teardown EXIT
trap 'log "failed: $BASH_COMMAND"' ERR
if [[ "$LOGS" == /var/tmp/capeval* ]]; then rm -rf "$LOGS"; fi
: > "$MANIFEST"
mkdir -p "$WORK/qdrant" "$WORK/blob"
note work_created "$WORK"
# the service reads its frozen model set from ./models: the checkout's, through a link
ln -s "$ROOT/models" "$WORK/models"
note redis_db "$REDIS_DB"

wait_for() {  # url, seconds, the pid that must stay alive meanwhile
  for _ in $(seq 1 "$2"); do
    curl -sf -m 2 "$1" >/dev/null && return 0
    kill -0 "$3" 2>/dev/null || { log "exited before ready: $1"; return 1; }
    sleep 1
  done
  log "not ready: $1"
  return 1
}

# Qdrant and OpenFGA of its own, started with a clean environment
env -i PATH=/usr/bin:/bin HOME="$WORK" \
  QDRANT__STORAGE__STORAGE_PATH="$WORK/qdrant" QDRANT__STORAGE__SNAPSHOTS_PATH="$WORK/qdrant/snapshots" \
  QDRANT__SERVICE__HTTP_PORT="$QDRANT_HTTP" QDRANT__SERVICE__GRPC_PORT="$QDRANT_GRPC" \
  "$QDRANT_BIN" --disable-telemetry > "$WORK/qdrant.log" 2>&1 &
note qdrant_pid $!
note qdrant_marker "QDRANT__STORAGE__STORAGE_PATH=$WORK/qdrant"
note qdrant_http "$QDRANT_HTTP"
env -i PATH=/usr/bin:/bin HOME="$WORK" "$OPENFGA_BIN" run --datastore-engine memory \
  --http-addr "127.0.0.1:$FGA_HTTP" --grpc-addr "127.0.0.1:$FGA_GRPC" \
  --playground-enabled=false --metrics-enabled=false --log-level warn > "$WORK/openfga.log" 2>&1 &
note fga_pid $!
note fga_marker "--http-addr 127.0.0.1:$FGA_HTTP"
wait_for "http://127.0.0.1:$QDRANT_HTTP/readyz" 60 "$(get qdrant_pid)"
wait_for "http://127.0.0.1:$FGA_HTTP/healthz" 60 "$(get fga_pid)"
# the instance answering on the port is the one started here
ours "$(get qdrant_pid)" "$(get qdrant_marker)" || refuse "the Qdrant on :$QDRANT_HTTP is not the one started here"

psql_ -c "create database $DB"
note db_created "$DB"
URL="postgresql+psycopg://$PGUSER_:$PGPASSWORD_@$PGHOST_:5432/$DB"
# the model gateway only when asked for (llm mode): its URL and key from the caller's env, and
# a model-call timeout and a query-expansion deadline the gateway's model can meet
# (benchmark/capabilities/stack_service.py: the 30 s and 250 ms constants are sized for a hosted
# model; Granite 2B on this CPU writes ~8 tokens/s)
ENTRY="from memory_service.__main__ import run_api; run_api()"
WORKER_ENTRY="from memory_service.__main__ import run_worker; run_worker()"
if [ "${CAPEVAL_GATEWAY:-0}" = 1 ]; then
  LLM_TIMEOUT="${CAPEVAL_LLM_TIMEOUT_SECONDS:-600}"
  [[ "$LLM_TIMEOUT" =~ ^[0-9]+$ ]] || refuse "CAPEVAL_LLM_TIMEOUT_SECONDS '$LLM_TIMEOUT' is not a number"
  EXPANSION_MS="${CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS:-60000}"
  [[ "$EXPANSION_MS" =~ ^[0-9]+$ ]] || refuse "CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS '$EXPANSION_MS' is not a number"
  ENTRY="from benchmark.capabilities.stack_service import main; main('api')"
  WORKER_ENTRY="from benchmark.capabilities.stack_service import main; main('worker')"
  export CAPEVAL_LLM_TIMEOUT_SECONDS="$LLM_TIMEOUT" CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS="$EXPANSION_MS"
fi
service_env() {  # root: the service's clean environment, serving that checkout's src/
  SERVICE_ENV=(
    PATH=/usr/bin:/bin HOME="$HOME" LANG=C.UTF-8 PYTHONPATH="$1/src:$1:$HARNESS"
    MEMORY__SERVICE__ENVIRONMENT=dev MEMORY__SERVICE__PORT="$API_PORT" MEMORY__SERVICE__WORKERS=1
    MEMORY__SERVICE__LOG_JSON=false MEMORY__SERVICE__LOG_LEVEL=WARNING
    MEMORY__AUTHENTICATION__TRUSTED_DEV_API_KEYS='["capeval-dev-key"]'
    MEMORY__DATABASE__URL="$URL" MEMORY__DATABASE__DIRECT_URL="$URL"
    MEMORY__CACHE__URL="redis://localhost:6379/$REDIS_DB"
    MEMORY__BLOB__FILESYSTEM_ROOT="$WORK/blob" MEMORY__TASKS__METRICS_PORT="$METRICS_PORT"
    MEMORY__SEARCH__QDRANT_URL="http://127.0.0.1:$QDRANT_HTTP"
    MEMORY__SEARCH__QDRANT_GRPC_PORT="$QDRANT_GRPC"
    MEMORY__AUTHORIZATION__OPENFGA_API_URL="http://127.0.0.1:$FGA_HTTP"
  )
  if [ "${CAPEVAL_GATEWAY:-0}" = 1 ]; then
    SERVICE_ENV+=(LLM_BASE_URL="$LLM_BASE_URL" LLM_API_KEY="$LLM_API_KEY"
      LLM_GATEWAY="${LLM_GATEWAY:-openai-compatible}" LLM_AUTH="${LLM_AUTH:-bearer}"
      CAPEVAL_LLM_TIMEOUT_SECONDS="$LLM_TIMEOUT" CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS="$EXPANSION_MS")
    [ -n "${CAPEVAL_WORKER_CONCURRENCY:-}" ] && SERVICE_ENV+=(MEMORY__TASKS__WORKER_CONCURRENCY="$CAPEVAL_WORKER_CONCURRENCY")
  fi
  return 0
}
migrate() {  # root: the eval database to that checkout's head
  local root=$1
  service_env "$root"
  # the service reads its frozen model set from ./models: the checkout's, through a link
  ln -sfn "$root/models" "$WORK/models"
  # migrations from WORK (no .env there), with the eval database named explicitly
  sed -e "s|^script_location = .*|script_location = $root/migrations|" \
      -e "s|^prepend_sys_path = .*|prepend_sys_path = $root/src\nsqlalchemy.url = $URL|" \
      "$root/alembic.ini" > "$WORK/alembic.ini"
  (cd "$WORK" && env -i "${SERVICE_ENV[@]}" "$PY" -m alembic -c "$WORK/alembic.ini" upgrade head >> "$WORK/migrate.log" 2>&1)
  (cd "$WORK" && env -i "${SERVICE_ENV[@]}" "$PY" -m memory_service.adapters.tasks.procrastinate_schema >> "$WORK/migrate.log" 2>&1) || true
}
start_service() {  # root: that checkout's API and worker
  local root=$1
  service_env "$root"
  (cd "$WORK" && exec env -i "${SERVICE_ENV[@]}" "$PY" -c "$ENTRY  # $API_MARK" >> "$WORK/api.log" 2>&1) &
  note api_pid $!
  (cd "$WORK" && exec env -i "${SERVICE_ENV[@]}" "$PY" -c "$WORKER_ENTRY  # $WORKER_MARK" >> "$WORK/worker.log" 2>&1) &
  note worker_pid $!
  wait_for "http://127.0.0.1:$API_PORT/health/ready" "$READY_SECONDS" "$(get api_pid)"
  wait_for "http://127.0.0.1:$METRICS_PORT/metrics" "$READY_SECONDS" "$(get worker_pid)"
  log "ready: memory on :$API_PORT ($root @ $(git -C "$root" rev-parse --short HEAD 2>/dev/null)), worker metrics on :$METRICS_PORT"
}
API_MARK="capeval-api-$API_PORT"
WORKER_MARK="capeval-worker-$METRICS_PORT"
note api_marker "$API_MARK"
note worker_marker "$WORKER_MARK"
migrate "$ROOT"
start_service "$ROOT"

export MEMORY_URL="http://127.0.0.1:$API_PORT"
export MEMORY_WORKER_METRICS_URL="http://127.0.0.1:$METRICS_PORT/metrics"
export CAPEVAL_ADMIN_KEY=capeval-dev-key CAPEVAL_ROOT="$ROOT"
unset TRELLIS_ADMIN_KEY TRELLIS_API_KEY
set +e
"$@"
rc=$?
set -e
if [ -n "$UPGRADE_ROOT" ] && [ "$rc" = 0 ]; then
  log "upgrade: stopping the service of $ROOT, serving the same stores from $UPGRADE_ROOT"
  stop "$(get worker_pid)" "$WORKER_MARK"
  stop "$(get api_pid)" "$API_MARK"
  migrate "$UPGRADE_ROOT"
  if [ -n "${CAPEVAL_UPGRADE_BACKFILL:-}" ]; then
    service_env "$UPGRADE_ROOT"
    log "upgrade: backfill: $CAPEVAL_UPGRADE_BACKFILL"
    set +e
    (cd "$UPGRADE_ROOT" && env -i "${SERVICE_ENV[@]}" bash -c "$CAPEVAL_UPGRADE_BACKFILL") > "$WORK/backfill.log" 2>&1
    brc=$?
    set -e
    log "upgrade: backfill rc=$brc"
    [ "$brc" = 0 ] || exit "$brc"
  fi
  start_service "$UPGRADE_ROOT"
  export CAPEVAL_ROOT="$UPGRADE_ROOT" CAPEVAL_UPGRADED_FROM="$ROOT"
  set +e
  bash -c "${CAPEVAL_UPGRADE_CMD:?CAPEVAL_UPGRADE_ROOT needs CAPEVAL_UPGRADE_CMD}"
  rc=$?
  set -e
fi
exit "$rc"
