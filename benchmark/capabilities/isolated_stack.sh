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

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
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

# ------------------------------------------------------------------ refuse before creating
[[ "$DB" =~ ^capeval[a-z0-9_]*$ ]] || refuse "database '$DB' does not match ^capeval"
# every path this script may delete, resolved (no "..", no symlink games) before the check
for path in "$WORK" "$LOGS"; do
  case "$path" in /var/tmp/capeval*) ;; *) refuse "path '$path' is not under /var/tmp/capeval*" ;; esac
done
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

if [ "${CAPEVAL_GATEWAY:-0}" = 1 ] && { [ -z "${BIFROST_URL:-}" ] || [ -z "${BIFROST_VIRTUAL_KEY:-}" ]; }; then
  refuse "CAPEVAL_GATEWAY=1 needs BIFROST_URL and BIFROST_VIRTUAL_KEY"
fi

# one isolated stack at a time
exec 8>"$WORK.lock"
flock -n 8 || refuse "another isolated stack holds $WORK.lock"

disk() {
  echo "df_avail_mb=$(df -BM --output=avail / | tail -1 | tr -d ' M')"\
" work_mb=$(du -sm "$WORK" 2>/dev/null | cut -f1 || echo 0)"\
" db_exists=$(psql_ -c "select count(*) from pg_database where datname='$DB'" 2>/dev/null || echo ?)"\
" redis_keys=$(redis-cli -n "$REDIS_DB" dbsize 2>/dev/null || echo ?)"
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
get() { grep -m1 "^$1=" "$MANIFEST" 2>/dev/null | cut -d= -f2- || true; }

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
  [ -n "$rdb" ] && redis-cli -n "$rdb" flushdb >/dev/null 2>&1
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
[ "$(redis-cli -n "$REDIS_DB" dbsize)" = 0 ] || refuse "Redis db $REDIS_DB is not empty"

log "disk before: $(disk)"
# the API and the worker each load the encoders (about 2 GB apiece): wait, bounded, for room
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -ge "${CAPEVAL_MIN_MEM_MB:-5000}" ] && break
  log "waiting for memory: ${avail} MB available"
  sleep 10
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
SERVICE_ENV=(
  PATH=/usr/bin:/bin HOME="$HOME" LANG=C.UTF-8 PYTHONPATH="$ROOT/src:$ROOT"
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
# the model gateway only when asked for (llm mode): its URL and key from the caller's env
if [ "${CAPEVAL_GATEWAY:-0}" = 1 ]; then
  SERVICE_ENV+=(BIFROST_URL="$BIFROST_URL" BIFROST_VIRTUAL_KEY="$BIFROST_VIRTUAL_KEY")
fi
# migrations from WORK (no .env there), with the eval database named explicitly
sed -e "s|^script_location = .*|script_location = $ROOT/migrations|" \
    -e "s|^prepend_sys_path = .*|prepend_sys_path = $ROOT/src\nsqlalchemy.url = $URL|" \
    "$ROOT/alembic.ini" > "$WORK/alembic.ini"
(cd "$WORK" && env -i "${SERVICE_ENV[@]}" "$PY" -m alembic -c "$WORK/alembic.ini" upgrade head > "$WORK/migrate.log" 2>&1)
(cd "$WORK" && env -i "${SERVICE_ENV[@]}" "$PY" -m memory_service.adapters.tasks.procrastinate_schema >> "$WORK/migrate.log" 2>&1) || true
API_MARK="capeval-api-$API_PORT"
WORKER_MARK="capeval-worker-$METRICS_PORT"
(cd "$WORK" && exec env -i "${SERVICE_ENV[@]}" "$PY" -c "from memory_service.__main__ import run_api; run_api()  # $API_MARK" > "$WORK/api.log" 2>&1) &
note api_pid $!
note api_marker "$API_MARK"
(cd "$WORK" && exec env -i "${SERVICE_ENV[@]}" "$PY" -c "from memory_service.__main__ import run_worker; run_worker()  # $WORKER_MARK" > "$WORK/worker.log" 2>&1) &
note worker_pid $!
note worker_marker "$WORKER_MARK"
wait_for "http://127.0.0.1:$API_PORT/health/ready" "$READY_SECONDS" "$(get api_pid)"
wait_for "http://127.0.0.1:$METRICS_PORT/metrics" "$READY_SECONDS" "$(get worker_pid)"
log "ready: memory on :$API_PORT, worker metrics on :$METRICS_PORT"

export MEMORY_URL="http://127.0.0.1:$API_PORT"
export MEMORY_WORKER_METRICS_URL="http://127.0.0.1:$METRICS_PORT/metrics"
export CAPEVAL_ADMIN_KEY=capeval-dev-key
unset TRELLIS_ADMIN_KEY TRELLIS_API_KEY
set +e
"$@"
rc=$?
set -e
exit "$rc"
