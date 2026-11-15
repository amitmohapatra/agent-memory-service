#!/bin/bash
# memh_lo.sh <label> <tree> <conversation> <usd cap>: one memh_lo.py ingest on its own database on
# 55957, in memory slot 2, stopped when the answered calls' spend in its log passes the cap.
label=$1; tree=$2; conv=$3; cap=$4
# the benchmark resets only a database whose name starts with memory_tests
D=/var/tmp/cons/memh-tmp; DB=memory_tests_memh_lo_$label; LOG=$D/lo_${label}_$conv.log; PY=/var/tmp/mx/ams/.venv/bin/python
mkdir /var/tmp/cons/memslot2 || { echo "slot 2 taken"; exit 1; }
trap 'rmdir /var/tmp/cons/memslot2 2>/dev/null' EXIT
export TMPDIR=$D/t HF_HUB_OFFLINE=1 BENCH_EMBEDDING=frozen PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$tree/src:$tree:/var/tmp/cons/full/bifrost-sdk/src:/var/tmp/cons/full/agent-contracts/src
export MEMORY__DATABASE__URL=postgresql+psycopg://postgres@127.0.0.1:55957/$DB MEMORY__DATABASE__DIRECT_URL=postgresql+psycopg://postgres@127.0.0.1:55957/$DB
export BENCH_LLM=on LLM_BASE_URL=https://api.anthropic.com/v1/ BENCH_LLM_MODEL=claude-haiku-5-5 MEMORY__MODELS__LLM__PROFILE__TEMPERATURE=false
unset LLM_GATEWAY
export LLM_API_KEY="$(cat /tmp/claude-0/-home-user/dce2e0ac-5a20-555f-aac7-8b600b986a51/scratchpad/.anthropic_key)"
psql -h 127.0.0.1 -p 55957 -U postgres -Atc "drop database if exists $DB" >/dev/null
createdb -h 127.0.0.1 -p 55957 -U postgres $DB && ( cd $tree && $PY -m alembic upgrade head > $D/alembic_$label.log 2>&1 )
( cd $D/t && $PY $D/memh_lo.py $label $conv > $LOG 2>&1 ) &
pid=$!
while kill -0 $pid 2>/dev/null; do
  usd=$($PY $D/memh_lo_count.py $LOG | $PY -c "import json,sys; print(json.load(sys.stdin)['answered_usd'])")
  if $PY -c "import sys; sys.exit(0 if $usd > $cap else 1)"; then
    echo "STOPPED at answered spend $usd > cap $cap" >> $LOG; pkill -P $pid; kill $pid; sleep 2; kill -9 $pid 2>/dev/null
  fi
  sleep 15
done
psql -h 127.0.0.1 -p 55957 -U postgres -Atc "drop database if exists $DB" >/dev/null
echo "memh_lo $label ended $(date +%H:%M)" >> $D/lo.out
