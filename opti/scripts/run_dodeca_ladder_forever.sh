#!/bin/bash
# run_dodeca_ladder_forever.sh — keep the 2.5-minute dodecahedron ladder
# running unattended, in batches, resuming from the last completed rung each
# time.  Each batch is one invocation of run_dodeca_ladder.sh (RUNGS rungs),
# which is itself resumable, so a crash or a stop costs at most the rung in
# flight and never the chain.
#
# Exclusive by construction: the batch runs under flock, so a second
# invocation of this script (or a manual run_dodeca_ladder.sh) waits for the
# lock rather than reading the same "highest rung on disk" and racing to write
# the same outdir.
#
# Failure policy: three consecutive batch failures stop the run so the cause
# can be inspected; a single failure is retried after a pause, resuming from
# the last good rung.  A batch that reports a full cover stops everything and
# audits the certificate.
#
# Usage:  HOURS=12 ./run_dodeca_ladder_forever.sh
#         HOURS=8 BATCH=40 ./run_dodeca_ladder_forever.sh
set -uo pipefail
cd "$(dirname "$0")"
HOURS=${HOURS:-12}
BATCH=${BATCH:-20}          # rungs per batch (~3 min each)
LOCK=${LOCK:-/tmp/dodeca_ladder.lock}
mkdir -p logs

deadline=$(( $(date +%s) + HOURS * 3600 ))
echo "ladder-forever: starting $(date -u +%H:%M:%SZ), running until $(date -u -d @$deadline +%H:%M:%SZ), $BATCH rungs per batch"
batch=0; fails=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  batch=$(( batch + 1 ))
  blog="logs/batch$(printf '%03d' "$batch").log"
  echo "=== batch $batch  $(date -u +%H:%M:%SZ)  -> $blog"
  # -w 600: wait up to 10 min for a lock another ladder may hold, rather than
  # failing or interleaving rungs with it.
  flock -w 600 "$LOCK" env RUNGS="$BATCH" ./run_dodeca_ladder.sh 2>&1 | tee "$blog"
  rc=${PIPESTATUS[0]}
  if grep -q "FULL COVER" "$blog"; then
    echo "*** a rung covered all 7776 facets — stopping the ladder to audit"
    exit 0
  fi
  if [ "$rc" -ne 0 ]; then
    fails=$(( fails + 1 ))
    echo "!!! batch $batch exited $rc (consecutive failures: $fails)"
    if [ "$fails" -ge 3 ]; then
      echo "!!! three consecutive failures — stopping so the cause can be inspected"
      exit "$rc"
    fi
    sleep 60
  else
    fails=0
  fi
done
echo "ladder-forever: deadline reached at $(date -u +%H:%M:%SZ) after $batch batches"
