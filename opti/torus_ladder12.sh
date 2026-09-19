#!/bin/bash
# torus_ladder12.sh — the remaining 10-domain hybrid splits: (7,3), (8,2), (5,5), 10 min each.  Waits for pid $1.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
hyb() { m=$1; k=$2; seed=$3
  echo "=== $(date -u +%FT%TZ) hybrid m=$m k=$k seed=$seed 600 s ==="
  "$PY" -u hybrid_cover.py torus results/torus_starcover_opt/summary.json $m $k results/torus_hybrid_m${m}_k${k} \
      --time-limit 600 --seed $seed 2>&1 | grep --line-buffered -v Warning
}
hyb 7 3 51
hyb 8 2 52
hyb 5 5 53
echo "=== LADDER12 DONE $(date -u +%FT%TZ) ==="
